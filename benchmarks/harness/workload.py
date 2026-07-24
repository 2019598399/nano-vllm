from dataclasses import asdict, dataclass
import hashlib
import json
import random
from pathlib import Path


@dataclass(frozen=True)
class TraceRequest:
    request_id: int
    arrival_s: float
    prompt_token_ids: list[int]
    output_tokens: int
    group: str = "default"

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class WorkloadTrace:
    name: str
    seed: int
    requests: list[TraceRequest]
    source: str = "synthetic"

    def to_dict(self):
        return {
            "name": self.name,
            "seed": self.seed,
            "source": self.source,
            "requests": [request.to_dict() for request in self.requests],
        }

    @property
    def sha256(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


PROFILES = {
    "prefill_interference": {
        "inputs": ([1024], [1.0]),
        "outputs": ([1], [1.0]),
        "shared_prefix": 0,
    },
    "mixed_chat": {
        "inputs": ([128, 256, 512, 1024], [0.20, 0.35, 0.30, 0.15]),
        "outputs": ([32, 64, 128, 256], [0.15, 0.40, 0.35, 0.10]),
        "shared_prefix": 0,
    },
    "decode_heavy": {
        "inputs": ([64, 128, 256], [0.25, 0.50, 0.25]),
        "outputs": ([256, 384, 512], [0.35, 0.40, 0.25]),
        "shared_prefix": 0,
    },
    "long_context": {
        "inputs": ([1024, 1536, 2048, 3072], [0.20, 0.35, 0.30, 0.15]),
        "outputs": ([32, 64, 128], [0.25, 0.50, 0.25]),
        "shared_prefix": 0,
    },
    "prefix_reuse": {
        "inputs": ([544, 640, 768], [0.35, 0.40, 0.25]),
        "outputs": ([64, 128], [0.60, 0.40]),
        "shared_prefix": 512,
    },
    "prefill_interference_output32": {
        "inputs": ([1024], [1.0]), "outputs": ([32], [1.0]), "shared_prefix": 0,
    },
    "prefill_interference_output128": {
        "inputs": ([1024], [1.0]), "outputs": ([128], [1.0]), "shared_prefix": 0,
    },
    "handoff_128": {"inputs": ([128], [1.0]), "outputs": ([64], [1.0]), "shared_prefix": 0},
    "handoff_512": {"inputs": ([512], [1.0]), "outputs": ([64], [1.0]), "shared_prefix": 0},
    "handoff_1024": {"inputs": ([1024], [1.0]), "outputs": ([64], [1.0]), "shared_prefix": 0},
    "handoff_2048": {"inputs": ([2048], [1.0]), "outputs": ([64], [1.0]), "shared_prefix": 0},
    "no_interference": {
        "inputs": ([128], [1.0]),
        "outputs": ([96], [1.0]),
        "shared_prefix": 0,
    },
}


def _weighted(rng, values, weights):
    return rng.choices(values, weights=weights, k=1)[0]


def synthetic_trace(name, request_count, qps, seed, vocab_size, max_model_len=4096):
    if name not in PROFILES:
        raise ValueError(f"unknown workload profile: {name}")
    profile = PROFILES[name]
    rng = random.Random(seed)
    if name.startswith("prefill_interference"):
        foreground = max(4, request_count // 4)
        background_output = profile["outputs"][0][0]
        requests = []
        for request_id in range(foreground):
            prompt = [rng.randrange(1, vocab_size) for _ in range(128)]
            requests.append(TraceRequest(request_id, 0.0, prompt, 256, "foreground"))
        arrival = 0.02
        for request_id in range(foreground, request_count):
            arrival += rng.expovariate(qps)
            prompt = [rng.randrange(1, vocab_size) for _ in range(1024)]
            requests.append(TraceRequest(request_id, arrival, prompt, background_output, "background"))
        return WorkloadTrace(name, seed, requests)
    shared_len = profile["shared_prefix"]
    shared = [rng.randrange(1, vocab_size) for _ in range(shared_len)]
    requests = []
    arrival = 0.0
    for request_id in range(request_count):
        input_len = _weighted(rng, *profile["inputs"])
        output_len = _weighted(rng, *profile["outputs"])
        if input_len + output_len > max_model_len:
            input_len = max_model_len - output_len
        suffix = [rng.randrange(1, vocab_size) for _ in range(input_len - shared_len)]
        if request_id:
            arrival += rng.expovariate(qps)
        requests.append(TraceRequest(request_id, arrival, shared + suffix, output_len, name))
    return WorkloadTrace(name, seed, requests)


def load_sharegpt(path, tokenizer, request_count, qps, seed, max_model_len=4096, max_output_tokens=256):
    source_path = Path(path)
    data = json.loads(source_path.read_text())
    rng = random.Random(seed)
    rng.shuffle(data)
    requests = []
    arrival = 0.0
    for item in data:
        conversations = item.get("conversations", [])
        if not conversations:
            continue
        prompt = conversations[0].get("value", "")
        answer = conversations[1].get("value", "") if len(conversations) > 1 else ""
        prompt_ids = tokenizer.encode(prompt)
        output_len = min(max(len(tokenizer.encode(answer)), 1), max_output_tokens)
        if not prompt_ids or len(prompt_ids) + output_len > max_model_len:
            continue
        request_id = len(requests)
        if request_id:
            arrival += rng.expovariate(qps)
        requests.append(TraceRequest(request_id, arrival, prompt_ids, output_len, "trace"))
        if len(requests) == request_count:
            break
    if len(requests) < request_count:
        raise ValueError(f"trace contains only {len(requests)} usable requests")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()[:12]
    return WorkloadTrace(f"sharegpt-{digest}", seed, requests, str(source_path.resolve()))


def save_trace(trace, path):
    Path(path).write_text(json.dumps(trace.to_dict(), indent=2))
