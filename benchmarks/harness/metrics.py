import math
import random
import statistics


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def describe(values):
    if not values:
        return {key: 0.0 for key in ("mean", "p50", "p90", "p95", "p99", "max")}
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def summarize_records(records, gpu_count, ttft_slo_ms=None, tpot_slo_ms=None):
    ttft = []
    admission_delay = []
    tpot = []
    e2e = []
    itl = []
    input_tokens = output_tokens = 0
    completed = []
    for record in records.values():
        times = record["token_times_ns"]
        if not times:
            continue
        submitted = record["arrival_ns"]
        if record.get("admitted_ns") is not None:
            admission_delay.append((record["admitted_ns"] - submitted) / 1e6)
        request_ttft = (times[0] - submitted) / 1e6
        intervals = [(right - left) / 1e6 for left, right in zip(times, times[1:])]
        request_tpot = statistics.fmean(intervals) if intervals else 0.0
        request_e2e = (times[-1] - submitted) / 1e6
        ttft.append(request_ttft)
        tpot.append(request_tpot)
        e2e.append(request_e2e)
        itl.extend(intervals)
        input_tokens += record["input_tokens"]
        output_tokens += len(times)
        completed.append((request_ttft, request_tpot))
    starts = [record["arrival_ns"] for record in records.values()]
    ends = [record["token_times_ns"][-1] for record in records.values() if record["token_times_ns"]]
    duration_s = (max(ends) - min(starts)) / 1e9 if starts and ends else 0.0
    attainment = None
    if ttft_slo_ms is not None and tpot_slo_ms is not None and completed:
        attainment = sum(a <= ttft_slo_ms and b <= tpot_slo_ms for a, b in completed) / len(completed)
    request_rate = len(completed) / duration_s if duration_s else 0.0
    return {
        "requests": len(completed),
        "duration_s": duration_s,
        "ttft_ms": describe(ttft),
        "admission_delay_ms": describe(admission_delay),
        "tpot_ms": describe(tpot),
        "itl_ms": describe(itl),
        "e2e_ms": describe(e2e),
        "input_tok_s": input_tokens / duration_s if duration_s else 0.0,
        "output_tok_s": output_tokens / duration_s if duration_s else 0.0,
        "total_tok_s": (input_tokens + output_tokens) / duration_s if duration_s else 0.0,
        "request_s": request_rate,
        "per_gpu_request_s": request_rate / gpu_count,
        "slo_attainment": attainment,
        "goodput_request_s": request_rate if attainment is not None and attainment >= 0.90 else 0.0,
        "per_gpu_goodput_request_s": request_rate / gpu_count if attainment is not None and attainment >= 0.90 else 0.0,
        "itl_samples": len(itl),
    }


def bootstrap_ci(values, seed=0, samples=2000, confidence=0.95):
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [rng.choice(values) for _ in values]
        estimates.append(statistics.median(draw))
    tail = (1 - confidence) / 2
    return [percentile(estimates, tail * 100), percentile(estimates, (1 - tail) * 100)]


def aggregate_runs(runs):
    paths = {
        "p99_itl_ms": ("summary", "itl_ms", "p99"),
        "p95_ttft_ms": ("summary", "ttft_ms", "p95"),
        "p95_tpot_ms": ("summary", "tpot_ms", "p95"),
        "p95_e2e_ms": ("summary", "e2e_ms", "p95"),
        "output_tok_s": ("summary", "output_tok_s"),
        "request_s": ("summary", "request_s"),
        "per_gpu_request_s": ("summary", "per_gpu_request_s"),
        "slo_attainment": ("summary", "slo_attainment"),
        "per_gpu_goodput_request_s": ("summary", "per_gpu_goodput_request_s"),
    }
    result = {}
    for name, path in paths.items():
        values = []
        for run in runs:
            value = run
            for key in path:
                value = value[key]
            values.append(value)
        result[name] = {
            "median": statistics.median(values),
            "iqr": [percentile(values, 25), percentile(values, 75)],
            "range": [min(values), max(values)],
            "ci95": bootstrap_ci(values),
            "values": values,
        }
    return result
