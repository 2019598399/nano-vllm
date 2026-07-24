#!/usr/bin/env python3
"""Reproducible serving benchmark for local, replicated, and PD execution."""

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from harness.metrics import aggregate_runs, bootstrap_ci, summarize_records
from harness.service import run_trace
from harness.workload import PROFILES, load_sharegpt, save_trace, synthetic_trace

MODES = ("local-tp1", "local-tp2", "local-dp2", "pd")
RUN_SCHEMA_VERSION = 2


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_hash(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def command_output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def source_hash():
    digest = hashlib.sha256()
    files = command_output(["git", "ls-files"]).splitlines()
    files += [
        str(path)
        for root, patterns in (
            ("benchmarks", ("*.py",)),
            ("nanovllm", ("*.py",)),
            ("tests", ("*.py",)),
            ("benchmark_profiles", ("*.json",)),
        )
        for pattern in patterns
        for path in Path(root).rglob(pattern)
        if str(path) not in files
    ]
    for name in sorted(set(files)):
        path = Path(name)
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_manifest(args, suite_path):
    model_config = Path(args.model) / "config.json"
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "source_sha256": source_hash(),
        "suite_config_sha256": file_hash(suite_path),
        "model": str(Path(args.model).resolve()),
        "model_config_sha256": file_hash(model_config),
        "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"]),
        "driver": command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "gpus": command_output(["nvidia-smi", "--query-gpu=index,name,memory.total,pci.bus_id", "--format=csv,noheader"]),
        "topology": command_output(["nvidia-smi", "topo", "-m"]),
        "cpu": command_output(["lscpu"]),
    }



def validate_suite(suite, tier_name):
    if not isinstance(suite, dict) or tier_name not in suite:
        raise ValueError(f"tier {tier_name!r} is not present in the suite config")
    tier = suite[tier_name]
    required = {"profiles", "requests", "factors", "paired"}
    if not isinstance(tier, dict) or not required.issubset(tier):
        raise ValueError(f"tier {tier_name!r} must define {sorted(required)}")
    unknown = [name for name in tier["profiles"] if name not in PROFILES]
    if unknown:
        raise ValueError(f"unknown workload profiles: {unknown}")
    if not tier["profiles"]:
        raise ValueError("profiles must not be empty")
    if not isinstance(tier["requests"], int) or tier["requests"] <= 0:
        raise ValueError("requests must be a positive integer")
    if not isinstance(tier["paired"], int) or tier["paired"] <= 0:
        raise ValueError("paired must be a positive integer")
    if not tier["factors"] or any(not isinstance(value, (int, float)) or value <= 0 for value in tier["factors"]):
        raise ValueError("factors must contain positive numbers")
    return tier


def run_fingerprint(trace, mode, config, context):
    return canonical_hash({
        "schema_version": RUN_SCHEMA_VERSION,
        "source_sha256": source_hash(),
        "trace_sha256": trace.sha256,
        "mode": mode,
        "engine_config": config,
        **context,
    })


def load_resumable(path, expected_fingerprint):
    result = json.loads(Path(path).read_text())
    actual = result.get("benchmark_fingerprint")
    if actual != expected_fingerprint:
        raise RuntimeError(
            f"cannot resume {path}: benchmark fingerprint mismatch "
            f"(expected {expected_fingerprint}, found {actual or 'legacy result'})"
        )
    return result


def validate_resume_manifest(path, expected):
    current = json.loads(Path(path).read_text())
    keys = ("schema_version", "source_sha256", "suite_config_sha256", "model_config_sha256")
    mismatch = [key for key in keys if current.get(key) != expected.get(key)]
    if mismatch:
        raise RuntimeError(f"cannot resume: manifest mismatch in {', '.join(mismatch)}")

def engine_config(args, seed, overrides=None):
    result = {
        "model": str(Path(args.model).resolve()),
        "seed": seed,
        "warmup_rounds": args.warmup_rounds,
        "cuda_graph": args.cuda_graph,
        "prefix_cache": args.prefix_cache,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    result.update(overrides or {})
    return result


def model_vocab_size(model):
    config = json.loads((Path(model) / "config.json").read_text())
    return config["vocab_size"]


def calibration_qps(profile, args, output_dir, suite_sha256):
    if args.qps:
        return args.qps
    count = 8 if args.tier == "smoke" else 16
    trace = synthetic_trace(profile, count, 1e6, args.seed - 1, model_vocab_size(args.model), args.max_model_len)
    path = output_dir / f"calibration-{profile}.json"
    config = engine_config(args, args.seed - 1)
    fingerprint = run_fingerprint(
        trace, "local-dp2", config,
        {"kind": "calibration", "profile": profile, "suite_sha256": suite_sha256},
    )
    if args.resume and path.exists():
        return max(load_resumable(path, fingerprint)["summary"]["request_s"], 0.1)
    result = run_trace(
        "local-dp2", trace, config, telemetry=False,
        startup_timeout_s=args.startup_timeout_s,
        no_progress_timeout_s=args.no_progress_timeout_s,
    )
    result["benchmark_fingerprint"] = fingerprint
    path.write_text(json.dumps(result, indent=2))
    return max(result["summary"]["request_s"], 0.1)


def mode_order(modes, seed):
    modes = list(modes)
    random.Random(seed).shuffle(modes)
    return modes


def derive_slos(runs, args):
    profiles = sorted({run["profile"] for run in runs})
    output = {}
    for profile in profiles:
        selected = [run for run in runs if run["profile"] == profile]
        if args.ttft_slo_ms and args.tpot_slo_ms:
            output[profile] = {"ttft_ms": args.ttft_slo_ms, "tpot_ms": args.tpot_slo_ms, "source": "cli"}
            continue
        ttft = [run["summary"]["ttft_ms"]["p90"] for run in selected]
        tpot = [run["summary"]["tpot_ms"]["p90"] for run in selected]
        multiplier = {"strict": (2.0, 1.5), "moderate": (5.0, 2.0), "relaxed": (10.0, 3.0)}[args.slo]
        output[profile] = {
            "ttft_ms": min(ttft) * multiplier[0],
            "tpot_ms": max(min(tpot) * multiplier[1], 0.1),
            "source": f"best-low-load-p90-{args.slo}",
        }
    return output


def apply_slo(run, slo):
    gpu_count = 1 if run["mode"] == "local-tp1" else 2
    run["summary"] = summarize_records(
        {int(key): value for key, value in run["records"].items()},
        gpu_count,
        slo["ttft_ms"],
        slo["tpot_ms"],
    )
    attained = run["summary"]["slo_attainment"] >= 0.90
    run["summary"]["goodput_request_s"] = run["offered_qps"] if attained else 0.0
    run["summary"]["per_gpu_goodput_request_s"] = run["offered_qps"] / gpu_count if attained else 0.0


def experiment_matrix(args, tier):
    profiles = args.profiles or tier["profiles"]
    base = [("baseline", profile, {}) for profile in profiles]
    if args.tier != "full" or args.no_ablations:
        return base
    return base + [
        ("cuda-graph-off", "decode_heavy", {"cuda_graph": False}),
        ("prefix-cache-on", "prefix_reuse", {"prefix_cache": True}),
        *[(f"chunk-{budget}", "mixed_chat", {"max_num_batched_tokens": budget}) for budget in (128, 256, 512, 1024)],
    ]


def write_events(run, path):
    with path.open("w") as handle:
        for request_id, record in run["records"].items():
            for index, timestamp in enumerate(record["token_times_ns"]):
                handle.write(json.dumps({
                    "request_id": int(request_id),
                    "token_index": index,
                    "timestamp_ns": timestamp,
                    "phase": record["phases"][index],
                    "arrival_ns": record["arrival_ns"],
                    "admitted_ns": record["admitted_ns"],
                }) + "\n")


def aggregate_groups(runs):
    grouped = {}
    for run in runs:
        key = "|".join((run["experiment"], run["profile"], str(run["rate_factor"]), run["mode"]))
        grouped.setdefault(key, []).append(run)
    return {key: aggregate_runs(values) for key, values in grouped.items()}


def best_goodput(groups):
    result = {}
    for group_key, aggregate in groups.items():
        experiment, profile, _, mode = group_key.split("|")
        key = "|".join((experiment, profile, mode))
        if aggregate["slo_attainment"]["median"] < 0.90:
            continue
        value = aggregate["per_gpu_goodput_request_s"]["median"]
        result[key] = max(result.get(key, 0.0), value)
    return result


def markdown_report(summary):
    lines = [
        "# Serving Benchmark Report", "",
        f"Tier: `{summary['tier']}`. SLOs are calibrated per workload and shared by all modes.", "",
    ]
    for profile, slo in sorted(summary["slos"].items()):
        lines.append(f"- `{profile}`: TTFT ≤ {slo['ttft_ms']:.2f} ms, TPOT ≤ {slo['tpot_ms']:.2f} ms")
    lines.extend(["",
        "| Experiment | Profile | Load | Mode | P99 ITL ms | P95 TTFT ms | Output tok/s | SLO attainment | Per-GPU goodput req/s |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ])
    for key, aggregate in sorted(summary["groups"].items()):
        experiment, profile, factor, mode = key.split("|")
        med = lambda name: aggregate[name]["median"]
        attainment_values = [run["summary"]["slo_attainment"] for run in summary["runs"] if "|".join((run["experiment"], run["profile"], str(run["rate_factor"]), run["mode"])) == key]
        lines.append(
            f"| {experiment} | {profile} | {factor} | {mode} | {med('p99_itl_ms'):.2f} | "
            f"{med('p95_ttft_ms'):.2f} | {med('output_tok_s'):.2f} | "
            f"{statistics.median(attainment_values):.3f} | {med('per_gpu_goodput_request_s'):.3f} |"
        )
    lines.extend(["", "## Best tested SLO goodput", ""])
    for key, value in sorted(summary["best_per_gpu_goodput"].items()):
        lines.append(f"- `{key}`: {value:.3f} req/s/GPU")
    lines.extend(["", "Results are workload- and SLO-specific; no universal PD throughput claim is made."])
    return "\n".join(lines) + "\n"


def write_csv(summary, path):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "profile", "rate_factor", "mode", "p99_itl_ms", "p95_ttft_ms", "output_tok_s", "per_gpu_goodput_request_s"])
        for key, aggregate in sorted(summary["groups"].items()):
            experiment, profile, factor, mode = key.split("|")
            writer.writerow([experiment, profile, factor, mode,
                             aggregate["p99_itl_ms"]["median"], aggregate["p95_ttft_ms"]["median"],
                             aggregate["output_tok_s"]["median"], aggregate["per_gpu_goodput_request_s"]["median"]])


def _run_group_key(run):
    return "|".join((run["experiment"], run["profile"], str(run["rate_factor"]), run["mode"]))


def _run_metric(run, metric):
    paths = {
        "output_tok_s": ("summary", "output_tok_s"),
        "per_gpu_goodput_request_s": ("summary", "per_gpu_goodput_request_s"),
        "p99_itl_ms": ("summary", "itl_ms", "p99"),
        "p95_ttft_ms": ("summary", "ttft_ms", "p95"),
    }
    value = run
    for key in paths[metric]:
        value = value[key]
    return value


def compare_results(current, baseline, strict=False):
    regressions = []
    previous = baseline.get("groups", {})
    current_runs = {(_run_group_key(run), run["repeat"]): run for run in current.get("runs", [])}
    baseline_runs = {(_run_group_key(run), run["repeat"]): run for run in baseline.get("runs", [])}
    for key, metrics in current["groups"].items():
        if key not in previous:
            continue
        for metric, limit, lower_better in (("output_tok_s", 0.05, False), ("per_gpu_goodput_request_s", 0.05, False), ("p99_itl_ms", 0.10, True), ("p95_ttft_ms", 0.10, True)):
            old = previous[key][metric]["median"]
            new = metrics[metric]["median"]
            if not old:
                continue
            paired_changes = []
            repeats = sorted(repeat for group, repeat in current_runs if group == key and (key, repeat) in baseline_runs)
            for repeat in repeats:
                old_value = _run_metric(baseline_runs[(key, repeat)], metric)
                new_value = _run_metric(current_runs[(key, repeat)], metric)
                if old_value:
                    paired_changes.append((new_value - old_value) / old_value)
            if len(paired_changes) < 2:
                continue
            change = (new - old) / old
            change_ci = bootstrap_ci(paired_changes)
            if lower_better:
                failed = change > limit and change_ci[0] > limit
            else:
                failed = change < -limit and change_ci[1] < -limit
            if failed:
                regressions.append({"group": key, "metric": metric, "old": old, "new": new,
                                    "change": change, "paired_change_ci95": change_ci,
                                    "paired_repeats": len(paired_changes)})
    if regressions:
        print("Performance regressions:")
        for item in regressions:
            print(json.dumps(item))
        if strict:
            raise SystemExit(2)
    return regressions


def finalize_results(output_dir, runs, args, tier):
    low_factor = min(tier["factors"])
    low_load = [run for run in runs if run["rate_factor"] == low_factor]
    slos = derive_slos(low_load, args)
    for run in runs:
        apply_slo(run, slos[run["profile"]])
    groups = aggregate_groups(runs)
    summary = {
        "schema_version": 1,
        "report_source_sha256": source_hash(),
        "tier": args.tier,
        "slos": slos,
        "runs": runs,
        "groups": groups,
        "best_per_gpu_goodput": best_goodput(groups),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "report.md").write_text(markdown_report(summary))
    write_csv(summary, output_dir / "summary.csv")
    if args.compare:
        baseline = json.loads(Path(args.compare).read_text())
        summary["regressions"] = compare_results(summary, baseline, args.strict)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    parser.add_argument("--tier", default="standard")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES))
    parser.add_argument("--output-dir")
    parser.add_argument("--suite-config", default="benchmark_profiles/defaults.json")
    parser.add_argument("--trace")
    parser.add_argument("--qps", type=float)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--paired-repeats", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--slo", choices=("strict", "moderate", "relaxed"), default="moderate")
    parser.add_argument("--ttft-slo-ms", type=float)
    parser.add_argument("--tpot-slo-ms", type=float)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--telemetry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--telemetry-interval-s", type=float, default=1.0)
    parser.add_argument("--startup-timeout-s", type=float, default=300.0)
    parser.add_argument("--no-progress-timeout-s", type=float, default=120.0)
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument("--compare")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only")
    return parser.parse_args()


def main():
    args = parse_args()
    args.model = str(Path(args.model).expanduser())
    suite_path = Path(args.suite_config)
    suite = json.loads(suite_path.read_text())
    tier = validate_suite(suite, args.tier)
    request_count = args.requests or tier["requests"]
    paired = args.paired_repeats or tier["paired"]
    if request_count <= 0 or paired <= 0:
        raise ValueError("requests and paired repeats must be positive")
    if min(args.startup_timeout_s, args.no_progress_timeout_s, args.telemetry_interval_s) <= 0:
        raise ValueError("timeout and telemetry intervals must be positive")
    if args.summarize_only:
        output_dir = Path(args.summarize_only)
        runs = [json.loads(path.read_text()) for path in sorted((output_dir / "runs").glob("*.json"))]
        finalize_results(output_dir, runs, args, tier)
        print(f"Report: {output_dir / 'report.md'}")
        return
    if args.resume and not args.output_dir:
        raise ValueError("--resume requires --output-dir")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"benchmark_results/serve_{args.tier}_{stamp}")
    runs_dir = output_dir / "runs"
    traces_dir = output_dir / "traces"
    runs_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = environment_manifest(args, suite_path)
    if args.resume:
        if not manifest_path.exists():
            raise RuntimeError("cannot resume: manifest.json is missing")
        validate_resume_manifest(manifest_path, manifest)
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    suite_sha256 = file_hash(suite_path)
    vocab_size = model_vocab_size(args.model)
    runs = []
    experiments = experiment_matrix(args, tier)
    for experiment, profile, overrides in experiments:
        base_qps = calibration_qps(profile, args, output_dir, suite_sha256)
        for factor in tier["factors"]:
            repetitions = paired if factor == 0.75 else 1
            for repeat in range(repetitions):
                seed = args.seed + repeat
                qps = base_qps * factor
                if args.trace:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
                    trace = load_sharegpt(args.trace, tokenizer, request_count, qps, seed, args.max_model_len)
                else:
                    trace = synthetic_trace(profile, request_count, qps, seed, vocab_size, args.max_model_len)
                trace_name = f"{experiment}-{profile}-{factor}-{repeat}.json"
                save_trace(trace, traces_dir / trace_name)
                for mode in mode_order(args.modes, seed + int(factor * 100)):
                    run_name = f"{experiment}-{profile}-{factor}-{repeat}-{mode}"
                    run_path = runs_dir / f"{run_name}.json"
                    config = engine_config(args, seed, overrides)
                    context = {
                        "experiment": experiment, "profile": profile,
                        "rate_factor": factor, "repeat": repeat,
                        "offered_qps": qps, "suite_sha256": suite_sha256,
                    }
                    fingerprint = run_fingerprint(trace, mode, config, context)
                    if args.resume and run_path.exists():
                        runs.append(load_resumable(run_path, fingerprint))
                        continue
                    print(
                        f"Running {experiment}/{profile} load={factor} "
                        f"repeat={repeat} mode={mode}", flush=True,
                    )
                    run = run_trace(
                        mode, trace, config, telemetry=args.telemetry,
                        telemetry_interval_s=args.telemetry_interval_s,
                        startup_timeout_s=args.startup_timeout_s,
                        no_progress_timeout_s=args.no_progress_timeout_s,
                    )
                    run.update({**context, "engine_config": config,
                                "benchmark_fingerprint": fingerprint})
                    write_events(run, runs_dir / f"{run_name}.events.jsonl")
                    run_path.write_text(json.dumps(run, indent=2))
                    runs.append(run)
    finalize_results(output_dir, runs, args, tier)
    print(f"Report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
