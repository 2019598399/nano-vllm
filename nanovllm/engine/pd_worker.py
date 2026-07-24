import torch
import torch.distributed as dist

from nanovllm.engine.kv_connector import KVConnectorRole, NCCLP2PConnector
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.layers.sampler import Sampler
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
from nanovllm.utils.parallel import set_tp_group


class PDLocalModelRunner(ModelRunner):
    """A full-model runner inside one PD worker process."""

    def __init__(self, config, role: str, device: int):
        self.config = config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager if role == "decode" else True
        self.world_size = 1
        self.rank = 0
        self.event = None
        self.graph_replay_count = 0

        torch.cuda.set_device(device)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(config.hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)


class PrefillModelRunner(PDLocalModelRunner):
    def __init__(self, config, device: int = 0):
        super().__init__(config, "prefill", device)


class DecodeModelRunner(PDLocalModelRunner):
    def __init__(self, config, device: int = 1):
        super().__init__(config, "decode", device)


def _init_process_group(rank: int, port: int):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", f"tcp://localhost:{port}", world_size=2, rank=rank)
    groups = [dist.new_group([0], backend="nccl"), dist.new_group([1], backend="nccl")]
    set_tp_group(groups[rank])


def pd_worker_main(config, role: str, connection):
    rank = 0 if role == "prefill" else 1
    _init_process_group(rank, config.pd_distributed_port)
    runner = PrefillModelRunner(config, rank) if rank == 0 else DecodeModelRunner(config, rank)
    connector_role = KVConnectorRole.PRODUCER if rank == 0 else KVConnectorRole.CONSUMER
    connector = NCCLP2PConnector(connector_role)
    connector.bind_kv_cache(runner.kv_cache)
    connection.send({"num_blocks": config.num_kvcache_blocks, "role": role})

    while True:
        command, payload = connection.recv()
        if command == "run":
            seqs, is_prefill = payload
            tokens = runner.run(seqs, is_prefill)
            connection.send({"tokens": tokens, "graph_replays": runner.graph_replay_count})
        elif command == "send_kv":
            connector.send(payload)
            connection.send(connector.stats)
        elif command == "recv_kv":
            connector.recv(payload)
            connection.send(connector.stats)
        elif command == "exit":
            break
        else:
            raise RuntimeError(f"unknown PD worker command: {command}")

    connection.close()
    torch.cuda.synchronize()
    dist.barrier(device_ids=[rank])
    dist.destroy_process_group()
