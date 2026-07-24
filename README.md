<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM

Nano-vLLM is a compact vLLM-style inference engine intended for learning and
experimentation. It supports Qwen3, paged KV cache, tensor parallelism, CUDA
Graph replay, prefix caching, chunked prefill, and an educational single-host
prefill/decode (PD) disaggregation path.

## Installation

Python 3.10–3.12 and a CUDA-capable system are required. For local development:

```bash
python -m venv .venv
./.venv/bin/python -m pip install -e .
```

Download the example model if it is not already available:

```bash
huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/
```

## Quick Start

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-0.6B", enforce_eager=True)
params = SamplingParams(temperature=0.6, max_tokens=128)
output = llm.generate(["Hello, Nano-vLLM."], params)
print(output[0]["text"])
```

Run the complete example with `./.venv/bin/python example.py`.

## Prefill/Decode Disaggregation

PD mode is deliberately small and readable. A CPU-side `PDProxy` routes each
request through two independent TP1 workers on one host:

```text
request → prefill worker (visible GPU 0)
        → NCCL P2P transfer of missing prompt KV
        → decode worker (visible GPU 1) → token events → proxy
```

Each worker holds a full model replica. Prefix caches are independent on P and
D; the connector transfers only the range missing on D. Prefill and decode can
overlap when both queues have work, while each KV handoff is synchronous.

```bash
CUDA_VISIBLE_DEVICES=0,1 ./.venv/bin/python example_pd.py
```

The same options are available through the Python API:

```python
llm = LLM(
    model_path,
    pd_disaggregation=True,
    enforce_eager=False,          # decode CUDA Graph replay
    enable_prefix_cache=True,
    enable_chunked_prefill=True,
    max_num_batched_tokens=1024,  # per-step prefill token budget
)
```

Current limitations: single host, exactly two visible GPUs, TP1 per PD worker,
and NCCL P2P KV transfer. This is a teaching implementation, not a production
fault-tolerant serving stack.

## Testing

CPU-safe unit tests cover scheduling, block ownership, proxy routing, benchmark
statistics, timeout handling, and resume validation:

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

GPU-specific behavior should be checked with the smoke benchmark below.

## Reproducible Serving Benchmark

`bench.py` is the original offline throughput benchmark, while
`benchmarks/pd_compare.py` is a focused PD interference microbenchmark. The
main harness compares four serving modes:

- `local-tp1`: one GPU; useful as a per-GPU reference.
- `local-tp2`: one tensor-parallel instance across two GPUs.
- `local-dp2`: two routed TP1 replicas; the primary resource-matched baseline.
- `pd`: one prefill GPU plus one decode GPU.

```bash
# Fast four-mode integration check
./.venv/bin/python benchmarks/serve_compare.py --tier smoke

# Paired multi-workload suite
./.venv/bin/python benchmarks/serve_compare.py --tier standard \
  --output-dir benchmark_results/my-standard-run

# Safely continue the exact same experiment
./.venv/bin/python benchmarks/serve_compare.py --tier standard \
  --output-dir benchmark_results/my-standard-run --resume

# Rebuild reports without loading a model
./.venv/bin/python benchmarks/serve_compare.py --tier standard \
  --summarize-only benchmark_results/my-standard-run

# Compare against a saved summary; exit 2 on a paired, significant regression
./.venv/bin/python benchmarks/serve_compare.py --tier standard \
  --compare path/to/baseline-summary.json --strict
```

`--resume` validates a fingerprint covering source, suite, model configuration,
trace, mode, and engine parameters. A mismatch fails instead of mixing results.
Worker startup and no-progress timeouts default to 300 and 120 seconds and can
be changed with `--startup-timeout-s` and `--no-progress-timeout-s`.

The suite reports TTFT (arrival to first token), TPOT (mean time per output
token), tail ITL (gap between consecutive tokens), throughput, SLO attainment,
and per-GPU goodput. Workloads and load factors live in
`benchmark_profiles/defaults.json`. Raw runs, traces, events, manifests, and
telemetry are written under ignored `benchmark_results/` directories.

Results are hardware-, model-, workload-, SLO-, and scheduler-specific. PD is
expected to isolate decode and improve tail ITL, but KV transfer and dedicated
P/D capacity can increase TTFT or reduce throughput. It must not be presented
as universally faster than replication or advanced chunked-prefill scheduling.
See the [curated 2× RTX 4090 report](docs/benchmarks/2026-07-24-pd-standard-2x4090.md).

## Project Layout

- `nanovllm/engine/`: proxies, schedulers, runners, workers, and KV connector.
- `nanovllm/layers/` and `nanovllm/models/`: CUDA-aware layers and model code.
- `benchmarks/`: serving harness, workload generation, and metrics.
- `tests/`: deterministic CPU-safe unit tests.
- `docs/benchmarks/`: small, reviewable experiment summaries; raw data stays local.

## Legacy Offline Result

The original `bench.py` comparison on an RTX 4070 Laptop with Qwen3-0.6B
reported 1,434 output tokens/s for Nano-vLLM and 1,362 output tokens/s for
vLLM over 256 sequences. Treat this as historical context, not a current
cross-version performance guarantee.
