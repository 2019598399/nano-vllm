import multiprocessing as mp
from multiprocessing.connection import wait
import os
import socket
import subprocess
import threading
import time
import traceback

from harness.metrics import summarize_records


def _proxy_stats(llm):
    proxy = llm.engine_proxy
    transfer = getattr(proxy, "transfer_stats", None)
    prefill = getattr(proxy, "prefill_scheduler", None)
    cache_stats = getattr(prefill, "stats", None)
    return {
        "graph_replays": getattr(
            proxy, "graph_replays",
            getattr(getattr(proxy, "runner", None), "graph_replay_count", 0),
        ),
        "kv_transfers": getattr(transfer, "transfers", 0),
        "kv_tokens": getattr(transfer, "tokens", 0),
        "kv_bytes": getattr(transfer, "bytes", 0),
        "kv_transfer_s": getattr(proxy, "transfer_seconds", 0.0),
        "prefix_matched_blocks": getattr(cache_stats, "matched_blocks", 0),
        "prefill_computed_tokens": getattr(cache_stats, "computed_tokens", 0),
        "prefill_chunks": getattr(cache_stats, "chunks", 0),
    }


def _engine_worker(connection, mode, devices, config):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        import random
        import torch
        from nanovllm import LLM, SamplingParams

        worker_seed = config["seed"] + int(devices.split(",")[0])
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed_all(worker_seed)
        tp_size = 2 if mode == "local-tp2" else 1
        llm = LLM(
            config["model"],
            pd_disaggregation=mode == "pd",
            tensor_parallel_size=tp_size,
            enforce_eager=not config["cuda_graph"],
            enable_prefix_cache=config["prefix_cache"],
            enable_chunked_prefill=True,
            max_model_len=config["max_model_len"],
            max_num_batched_tokens=config["max_num_batched_tokens"],
            max_num_seqs=config["max_num_seqs"],
            gpu_memory_utilization=config["gpu_memory_utilization"],
            distributed_port=config["distributed_port"],
            pd_distributed_port=config["pd_distributed_port"],
        )
        rng = random.Random(worker_seed)
        for _ in range(config["warmup_rounds"]):
            params = SamplingParams(temperature=1.0, max_tokens=8, ignore_eos=True)
            for _ in range(2):
                prompt = [rng.randrange(1, llm.config.hf_config.vocab_size) for _ in range(128)]
                llm.add_request(prompt, params)
            while not llm.is_finished():
                llm.step_with_events()
        mapping = {}
        connection.send(("ready", {
            "stats": _proxy_stats(llm),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "worker_seed": worker_seed,
        }))
        stopping = False
        while not stopping or not llm.is_finished():
            if llm.is_finished() and not connection.poll():
                command, payload = connection.recv()
                commands = [(command, payload)]
            else:
                commands = []
                while connection.poll():
                    commands.append(connection.recv())
            for command, payload in commands:
                if command == "add":
                    params = SamplingParams(temperature=1.0, max_tokens=payload["output_tokens"], ignore_eos=True)
                    seq_id = llm.add_request(payload["prompt_token_ids"], params)
                    mapping[seq_id] = payload["request_id"]
                    connection.send(("admitted", {
                        "request_id": payload["request_id"],
                        "timestamp_ns": time.monotonic_ns(),
                    }))
                elif command == "stop":
                    stopping = True
                else:
                    raise RuntimeError(f"unknown benchmark worker command: {command}")
            if not llm.is_finished():
                output = llm.step_with_events()
                events = []
                for event in output.token_events:
                    events.append({
                        "request_id": mapping[event.seq_id],
                        "token_id": event.token_id,
                        "timestamp_ns": int(event.timestamp * 1e9),
                        "phase": event.phase.name.lower(),
                    })
                finished = []
                for seq_id, _ in output.finished:
                    finished.append(mapping.pop(seq_id))
                connection.send(("step", {
                    "events": events,
                    "finished": finished,
                    "stats": _proxy_stats(llm),
                }))
        llm.exit()
        connection.send(("stopped", None))
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        finally:
            connection.close()
        raise


class ServiceClient:
    def __init__(self, mode, devices, config, startup_timeout_s=300):
        self.mode = mode
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        self.connection = parent
        self.process = ctx.Process(target=_engine_worker, args=(child, mode, devices, config))
        self.process.start()
        if not self.connection.poll(startup_timeout_s):
            self._terminate()
            raise TimeoutError(
                f"{mode} worker pid={self.process.pid} did not become ready "
                f"within {startup_timeout_s}s"
            )
        try:
            kind, payload = self.recv()
        except BaseException:
            self._terminate()
            self.connection.close()
            raise
        if kind != "ready":
            self._terminate()
            self.connection.close()
            raise RuntimeError(payload)
        self.initial_stats = payload["stats"]
        self.last_stats = payload["stats"]
        self.runtime = payload
        self.outstanding_work = 0

    def add(self, request):
        try:
            self.connection.send(("add", request.to_dict()))
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise RuntimeError(self.failure_message("failed to submit request")) from exc
        self.outstanding_work += len(request.prompt_token_ids) + request.output_tokens

    def failure_message(self, message):
        return (
            f"{self.mode} worker pid={self.process.pid} {message}; "
            f"alive={self.process.is_alive()} exitcode={self.process.exitcode}"
        )

    def recv(self):
        try:
            return self.connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(self.failure_message("closed its control pipe")) from exc

    def _terminate(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=5)

    def stop(self):
        try:
            if self.process.is_alive():
                try:
                    self.connection.send(("stop", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
                deadline = time.monotonic() + 30
                while self.process.is_alive() and time.monotonic() < deadline:
                    if self.connection.poll(0.2):
                        try:
                            kind, _ = self.connection.recv()
                        except (EOFError, OSError):
                            break
                        if kind == "stopped":
                            break
            self.process.join(timeout=5)
            if self.process.is_alive():
                self._terminate()
        finally:
            self.connection.close()


class TelemetrySampler:
    def __init__(self, enabled=True, interval=1.0):
        self.enabled = enabled
        self.interval = interval
        self.samples = []
        self.failures = 0
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        command = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                rows = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).splitlines()
                timestamp_ns = time.monotonic_ns()
                for row in rows:
                    values = [part.strip() for part in row.split(",")]
                    self.samples.append({
                        "timestamp_ns": timestamp_ns,
                        "gpu": int(values[0]),
                        "utilization_pct": float(values[1]),
                        "memory_mib": float(values[2]),
                        "power_w": float(values[3]),
                        "temperature_c": float(values[4]),
                        "sm_clock_mhz": float(values[5]),
                    })
            except (OSError, subprocess.CalledProcessError, ValueError):
                self.failures += 1
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self.samples


def summarize_telemetry(samples):
    grouped = {}
    for sample in samples:
        grouped.setdefault(sample["gpu"], []).append(sample)
    output = {}
    for gpu, rows in grouped.items():
        output[str(gpu)] = {}
        for key in ("utilization_pct", "memory_mib", "power_w", "temperature_c", "sm_clock_mhz"):
            values = [row[key] for row in rows]
            output[str(gpu)][key] = {"mean": sum(values) / len(values), "max": max(values)}
    return output


def _delta_stats(clients):
    output = {}
    for client in clients:
        for key, value in client.last_stats.items():
            output[key] = output.get(key, 0) + value - client.initial_stats.get(key, 0)
    return output


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _with_ports(config):
    result = dict(config)
    result["distributed_port"] = _free_port()
    result["pd_distributed_port"] = _free_port()
    return result


def run_trace(
    mode, trace, config, ttft_slo_ms=None, tpot_slo_ms=None, telemetry=True,
    telemetry_interval_s=1.0, startup_timeout_s=300, no_progress_timeout_s=120,
):
    clients = []
    try:
        if mode == "local-dp2":
            clients = [
                ServiceClient("local-tp1", "0", _with_ports(config), startup_timeout_s),
                ServiceClient("local-tp1", "1", _with_ports(config), startup_timeout_s),
            ]
            gpu_count = 2
        else:
            devices = "0" if mode == "local-tp1" else "0,1"
            clients = [ServiceClient(mode, devices, _with_ports(config), startup_timeout_s)]
            gpu_count = 1 if mode == "local-tp1" else 2
    except BaseException:
        for client in clients:
            client.stop()
        raise
    records = {
        request.request_id: {
            "arrival_ns": 0,
            "admitted_ns": None,
            "token_times_ns": [],
            "phases": [],
            "input_tokens": len(request.prompt_token_ids),
            "expected_output_tokens": request.output_tokens,
            "worker": None,
        }
        for request in trace.requests
    }
    sampler = TelemetrySampler(telemetry, telemetry_interval_s)
    sampler.start()
    start_ns = time.monotonic_ns()
    next_request = 0
    completed = 0
    last_progress = time.monotonic()
    try:
        while completed < len(trace.requests):
            now_ns = time.monotonic_ns()
            while next_request < len(trace.requests):
                request = trace.requests[next_request]
                arrival_ns = start_ns + int(request.arrival_s * 1e9)
                if arrival_ns > now_ns:
                    break
                client_index = 0
                if len(clients) > 1:
                    client_index = min(range(len(clients)), key=lambda i: (clients[i].outstanding_work, i))
                records[request.request_id]["arrival_ns"] = arrival_ns
                records[request.request_id]["worker"] = client_index
                clients[client_index].add(request)
                next_request += 1
            active = [client.connection for client in clients]
            timeout = min(no_progress_timeout_s, 1.0)
            if next_request < len(trace.requests):
                deadline_ns = start_ns + int(trace.requests[next_request].arrival_s * 1e9)
                until_arrival = max((deadline_ns - time.monotonic_ns()) / 1e9, 0)
                timeout = min(timeout, until_arrival)
            ready = wait(active, timeout)
            if not ready:
                dead = [client for client in clients if not client.process.is_alive()]
                if dead:
                    raise RuntimeError(dead[0].failure_message("exited before completing the trace"))
                has_outstanding_work = any(client.outstanding_work for client in clients)
                if has_outstanding_work and time.monotonic() - last_progress >= no_progress_timeout_s:
                    raise TimeoutError(
                        f"{mode} made no progress for {no_progress_timeout_s}s "
                        f"({completed}/{len(trace.requests)} requests complete)"
                    )
            for connection in ready:
                client = next(client for client in clients if client.connection is connection)
                while connection.poll():
                    kind, payload = client.recv()
                    last_progress = time.monotonic()
                    if kind == "error":
                        raise RuntimeError(payload)
                    if kind == "admitted":
                        records[payload["request_id"]]["admitted_ns"] = payload["timestamp_ns"]
                    elif kind == "step":
                        client.last_stats = payload["stats"]
                        for event in payload["events"]:
                            record = records[event["request_id"]]
                            record["token_times_ns"].append(event["timestamp_ns"])
                            record["phases"].append(event["phase"])
                        for request_id in payload["finished"]:
                            request = trace.requests[request_id]
                            client.outstanding_work -= len(request.prompt_token_ids) + request.output_tokens
                            completed += 1
        for request_id, record in records.items():
            actual = len(record["token_times_ns"])
            if actual != record["expected_output_tokens"]:
                expected = record["expected_output_tokens"]
                raise RuntimeError(
                    f"request {request_id} emitted {actual} tokens, expected {expected}"
                )
            if record["token_times_ns"] != sorted(record["token_times_ns"]):
                raise RuntimeError(f"request {request_id} has non-monotonic token timestamps")
        summary = summarize_records(records, gpu_count, ttft_slo_ms, tpot_slo_ms)
        telemetry_samples = sampler.stop()
        return {
            "mode": mode,
            "trace": {"name": trace.name, "seed": trace.seed, "sha256": trace.sha256},
            "summary": summary,
            "engine_stats": _delta_stats(clients),
            "workers": [client.runtime for client in clients],
            "telemetry": summarize_telemetry(telemetry_samples),
            "telemetry_status": {
                "enabled": telemetry,
                "interval_s": telemetry_interval_s,
                "samples": len(telemetry_samples),
                "failures": sampler.failures,
            },
            "telemetry_samples": telemetry_samples,
            "records": {str(key): value for key, value in records.items()},
        }
    finally:
        sampler.stop()
        for client in clients:
            client.stop()
