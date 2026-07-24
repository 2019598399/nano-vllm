#!/usr/bin/env python3
"""Reproducible mixed prefill/decode benchmark for local and PD execution."""

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

MODES = ("local-tp1", "local-tp2", "pd")


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_requests(records, request_ids):
    ttft = []
    e2e = []
    itl = []
    tpot = []
    for seq_id in request_ids:
        record = records[seq_id]
        times = record["token_times"]
        if not times:
            continue
        ttft.append(times[0] - record["submitted"])
        e2e.append(times[-1] - record["submitted"])
        intervals = [right - left for left, right in zip(times, times[1:])]
        itl.extend(intervals)
        if intervals:
            tpot.append(sum(intervals) / len(intervals))
    return {
        "mean_ttft_ms": statistics.fmean(ttft) * 1000 if ttft else 0.0,
        "p95_ttft_ms": percentile(ttft, 95) * 1000,
        "mean_itl_ms": statistics.fmean(itl) * 1000 if itl else 0.0,
        "p95_itl_ms": percentile(itl, 95) * 1000,
        "p99_itl_ms": percentile(itl, 99) * 1000,
        "mean_tpot_ms": statistics.fmean(tpot) * 1000 if tpot else 0.0,
        "mean_e2e_ms": statistics.fmean(e2e) * 1000 if e2e else 0.0,
        "p95_e2e_ms": percentile(e2e, 95) * 1000,
        "itl_samples": len(itl),
    }


def make_prompt(rng, length, vocab_size):
    return [rng.randrange(1, vocab_size) for _ in range(length)]


def drain(engine):
    while not engine.is_finished():
        engine.step_with_events()


def run_once(engine, args, seed):
    import random
    import torch
    from nanovllm import SamplingParams

    random_source = random.Random(seed)
    torch.manual_seed(seed)
    vocab_size = engine.config.hf_config.vocab_size
    foreground_ids = []
    background_ids = []
    records = {}

    foreground_params = SamplingParams(
        temperature=1.0,
        max_tokens=args.foreground_output_len,
        ignore_eos=True,
    )
    background_params = SamplingParams(
        temperature=1.0,
        max_tokens=args.background_output_len,
        ignore_eos=True,
    )

    proxy = engine.engine_proxy
    transfer = getattr(proxy, "transfer_stats", None)
    counters_before = {
        "graph_replays": getattr(proxy, "graph_replays", getattr(getattr(proxy, "runner", None), "graph_replay_count", 0)),
        "kv_transfers": getattr(transfer, "transfers", 0),
        "kv_tokens": getattr(transfer, "tokens", 0),
        "kv_bytes": getattr(transfer, "bytes", 0),
        "kv_transfer_s": getattr(proxy, "transfer_seconds", 0.0),
    }
    started = perf_counter()
    for _ in range(args.foreground_requests):
        prompt = make_prompt(random_source, args.foreground_input_len, vocab_size)
        submitted = perf_counter()
        seq_id = engine.add_request(prompt, foreground_params)
        foreground_ids.append(seq_id)
        records[seq_id] = {"submitted": submitted, "token_times": [], "kind": "foreground"}

    wave = 0
    milestones = args.injection_milestones
    finished_at = started
    while not engine.is_finished() or wave < len(milestones):
        if engine.is_finished():
            raise RuntimeError("foreground completed before all background waves were injected")
        output = engine.step_with_events()
        for event in output.token_events:
            if event.seq_id in records:
                records[event.seq_id]["token_times"].append(event.timestamp)
        for seq_id, _ in output.finished:
            if seq_id in records:
                records[seq_id]["finished"] = perf_counter()
        foreground_progress = min(len(records[seq_id]["token_times"]) for seq_id in foreground_ids)
        while wave < len(milestones) and foreground_progress >= milestones[wave]:
            for _ in range(args.background_requests_per_wave):
                prompt = make_prompt(random_source, args.background_input_len, vocab_size)
                submitted = perf_counter()
                seq_id = engine.add_request(prompt, background_params)
                background_ids.append(seq_id)
                records[seq_id] = {
                    "submitted": submitted,
                    "token_times": [],
                    "kind": "background",
                    "wave": wave,
                }
            wave += 1
        finished_at = perf_counter()

    for seq_id, record in records.items():
        expected = args.foreground_output_len if record["kind"] == "foreground" else args.background_output_len
        actual = len(record["token_times"])
        if actual != expected:
            raise RuntimeError(f"request {seq_id} emitted {actual} tokens, expected {expected}")

    all_output_tokens = sum(len(record["token_times"]) for record in records.values())
    transfer = getattr(proxy, "transfer_stats", None)
    result = {
        "seed": seed,
        "duration_s": finished_at - started,
        "output_tokens": all_output_tokens,
        "output_throughput_tok_s": all_output_tokens / (finished_at - started),
        "foreground": summarize_requests(records, foreground_ids),
        "background": summarize_requests(records, background_ids),
        "graph_replays": getattr(proxy, "graph_replays", getattr(getattr(proxy, "runner", None), "graph_replay_count", 0)) - counters_before["graph_replays"],
        "kv_transfers": getattr(transfer, "transfers", 0) - counters_before["kv_transfers"],
        "kv_tokens": getattr(transfer, "tokens", 0) - counters_before["kv_tokens"],
        "kv_bytes": getattr(transfer, "bytes", 0) - counters_before["kv_bytes"],
        "kv_transfer_s": getattr(proxy, "transfer_seconds", 0.0) - counters_before["kv_transfer_s"],
        "requests": {str(key): value for key, value in records.items()},
    }
    return result


def worker(mode, args):
    import random
    import torch
    from nanovllm import LLM, SamplingParams

    tp_size = 2 if mode == "local-tp2" else 1
    llm = LLM(
        args.model,
        pd_disaggregation=mode == "pd",
        tensor_parallel_size=tp_size,
        enforce_eager=False,
        enable_prefix_cache=False,
        enable_chunked_prefill=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    try:
        for index in range(args.warmups):
            rng = random.Random(args.seed + 10000 + index)
            params = SamplingParams(temperature=1.0, max_tokens=8, ignore_eos=True)
            for _ in range(2):
                llm.add_request(make_prompt(rng, 128, llm.config.hf_config.vocab_size), params)
            drain(llm)
        runs = [run_once(llm, args, args.seed + index) for index in range(args.repeats)]
        return {
            "mode": mode,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "config": benchmark_config(args),
            "runs": runs,
            "aggregate": aggregate_runs(runs),
        }
    finally:
        llm.exit()


def aggregate_runs(runs):
    fields = (
        ("duration_s",),
        ("output_throughput_tok_s",),
        ("foreground", "mean_ttft_ms"),
        ("foreground", "p95_itl_ms"),
        ("foreground", "p99_itl_ms"),
        ("foreground", "mean_tpot_ms"),
        ("foreground", "mean_e2e_ms"),
        ("kv_transfer_s",),
    )
    output = {}
    for path in fields:
        values = []
        for run in runs:
            value = run
            for key in path:
                value = value[key]
            values.append(value)
        output["_".join(path)] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    return output


def benchmark_config(args):
    return {
        "model": str(Path(args.model).expanduser()),
        "seed": args.seed,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "foreground_requests": args.foreground_requests,
        "foreground_input_len": args.foreground_input_len,
        "foreground_output_len": args.foreground_output_len,
        "background_requests_per_wave": args.background_requests_per_wave,
        "background_input_len": args.background_input_len,
        "background_output_len": args.background_output_len,
        "injection_milestones": args.injection_milestones,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "prefix_cache": False,
        "chunked_prefill": True,
        "cuda_graph": True,
    }


def command_output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def metadata():
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "gpus": command_output([
            "nvidia-smi", "--query-gpu=index,name,memory.total,pci.bus_id", "--format=csv,noheader"
        ]),
        "topology": command_output(["nvidia-smi", "topo", "-m"]),
    }


def print_report(results):
    print("\nMode comparison (median across measured runs)")
    print(f"{'mode':<12} {'P99 ITL ms':>12} {'P95 ITL ms':>12} {'TTFT ms':>12} {'E2E ms':>12} {'tok/s':>12}")
    for mode in MODES:
        if mode not in results:
            continue
        aggregate = results[mode]["aggregate"]
        value = lambda key: aggregate[key]["median"]
        print(
            f"{mode:<12} {value('foreground_p99_itl_ms'):>12.2f} "
            f"{value('foreground_p95_itl_ms'):>12.2f} "
            f"{value('foreground_mean_ttft_ms'):>12.2f} "
            f"{value('foreground_mean_e2e_ms'):>12.2f} "
            f"{value('output_throughput_tok_s'):>12.2f}"
        )
    if "pd" in results and "local-tp2" in results:
        pd_itl = results["pd"]["aggregate"]["foreground_p99_itl_ms"]["median"]
        baseline = results["local-tp2"]["aggregate"]["foreground_p99_itl_ms"]["median"]
        reduction = (baseline - pd_itl) / baseline * 100 if baseline else 0.0
        verdict = "PASS" if reduction >= 20 else "FAIL"
        print(f"\nPrimary acceptance: {verdict} — PD P99 ITL reduction vs local-tp2: {reduction:.2f}%")


def markdown_report(results):
    lines = [
        "# PD Benchmark Results",
        "",
        "Median values across measured runs. P99 ITL is the primary acceptance metric.",
        "",
        "| Mode | P99 ITL (ms) | P95 ITL (ms) | Mean TTFT (ms) | Mean E2E (ms) | Output tok/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        if mode not in results:
            continue
        aggregate = results[mode]["aggregate"]
        value = lambda key: aggregate[key]["median"]
        lines.append(
            f"| {mode} | {value('foreground_p99_itl_ms'):.2f} | "
            f"{value('foreground_p95_itl_ms'):.2f} | "
            f"{value('foreground_mean_ttft_ms'):.2f} | "
            f"{value('foreground_mean_e2e_ms'):.2f} | "
            f"{value('output_throughput_tok_s'):.2f} |"
        )
    if "pd" in results and "local-tp2" in results:
        pd_itl = results["pd"]["aggregate"]["foreground_p99_itl_ms"]["median"]
        baseline = results["local-tp2"]["aggregate"]["foreground_p99_itl_ms"]["median"]
        reduction = (baseline - pd_itl) / baseline * 100 if baseline else 0.0
        verdict = "PASS" if reduction >= 20 else "FAIL"
        lines.extend([
            "",
            f"Primary acceptance: **{verdict}** — PD reduces median P99 ITL by "
            f"{reduction:.2f}% versus local-tp2 (required: 20%).",
            "",
            "See `results.json` and per-mode JSON files for raw token timestamps and environment metadata.",
        ])
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    parser.add_argument("--mode", choices=("all",) + MODES, default="all")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--foreground-requests", type=int, default=4)
    parser.add_argument("--foreground-input-len", type=int, default=128)
    parser.add_argument("--foreground-output-len", type=int, default=96)
    parser.add_argument("--background-requests-per-wave", type=int, default=1)
    parser.add_argument("--background-input-len", type=int, default=1024)
    parser.add_argument("--background-output-len", type=int, default=1)
    parser.add_argument("--injection-milestones", type=int, nargs="+", default=[8, 24, 40, 56])
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--worker", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser.parse_args()


def forwarded_args(args):
    values = [
        "--model", str(Path(args.model).expanduser()),
        "--seed", str(args.seed),
        "--warmups", str(args.warmups),
        "--repeats", str(args.repeats),
        "--foreground-requests", str(args.foreground_requests),
        "--foreground-input-len", str(args.foreground_input_len),
        "--foreground-output-len", str(args.foreground_output_len),
        "--background-requests-per-wave", str(args.background_requests_per_wave),
        "--background-input-len", str(args.background_input_len),
        "--background-output-len", str(args.background_output_len),
        "--max-model-len", str(args.max_model_len),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--max-num-seqs", str(args.max_num_seqs),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--injection-milestones", *map(str, args.injection_milestones),
    ]
    return values


def main():
    args = parse_args()
    args.model = str(Path(args.model).expanduser())
    if args.worker:
        result = worker(args.worker, args)
        Path(args.worker_output).write_text(json.dumps(result, indent=2))
        print(f"completed {args.worker}: {args.worker_output}")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"benchmark_results/pd_compare_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = MODES if args.mode == "all" else (args.mode,)
    results = {}
    for mode in selected:
        output = output_dir / f"{mode}.json"
        devices = "0" if mode == "local-tp1" else "0,1"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = devices
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker", mode,
            "--worker-output", str(output.resolve()),
            *forwarded_args(args),
        ]
        print(f"\nRunning {mode} on CUDA_VISIBLE_DEVICES={devices}", flush=True)
        subprocess.run(command, check=True, env=env)
        results[mode] = json.loads(output.read_text())

    report = {"metadata": metadata(), "config": benchmark_config(args), "results": results}
    combined = output_dir / "results.json"
    combined.write_text(json.dumps(report, indent=2))
    (output_dir / "report.md").write_text(markdown_report(results))
    print_report(results)
    print(f"Raw results: {combined}")


if __name__ == "__main__":
    main()
