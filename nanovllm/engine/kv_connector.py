from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import torch
import torch.distributed as dist


class KVConnectorRole(str, Enum):
    PRODUCER = "producer"
    CONSUMER = "consumer"


@dataclass
class KVTransferMetadata:
    request_ids: list[int]
    source_slots: list[int]
    destination_slots: list[int]
    token_counts: list[int]

    @property
    def total_tokens(self) -> int:
        return sum(self.token_counts)


@dataclass
class KVTransferStats:
    transfers: int = 0
    tokens: int = 0
    bytes: int = 0


class KVConnectorBase(ABC):
    def __init__(self, role: KVConnectorRole):
        self.role = role
        self.kv_cache: torch.Tensor | None = None
        self.stats = KVTransferStats()

    def bind_kv_cache(self, kv_cache: torch.Tensor):
        self.kv_cache = kv_cache

    @abstractmethod
    def send(self, metadata: KVTransferMetadata):
        raise NotImplementedError

    @abstractmethod
    def recv(self, metadata: KVTransferMetadata):
        raise NotImplementedError


class NCCLP2PConnector(KVConnectorBase):
    """Synchronous, single-host NCCL connector used by the teaching PD path."""

    def send(self, metadata: KVTransferMetadata):
        assert self.role == KVConnectorRole.PRODUCER and self.kv_cache is not None
        cache = self.kv_cache
        flat = cache.view(cache.size(0), cache.size(1), -1, cache.size(4), cache.size(5))
        slots = torch.tensor(metadata.source_slots, dtype=torch.long, device="cuda")
        packed = flat.index_select(2, slots).contiguous()
        dist.send(packed, dst=1)
        torch.cuda.synchronize()
        self._record(metadata, packed)

    def recv(self, metadata: KVTransferMetadata):
        assert self.role == KVConnectorRole.CONSUMER and self.kv_cache is not None
        cache = self.kv_cache
        shape = (cache.size(0), cache.size(1), metadata.total_tokens, cache.size(4), cache.size(5))
        packed = torch.empty(shape, dtype=cache.dtype, device="cuda")
        dist.recv(packed, src=0)
        flat = cache.view(cache.size(0), cache.size(1), -1, cache.size(4), cache.size(5))
        slots = torch.tensor(metadata.destination_slots, dtype=torch.long, device="cuda")
        flat.index_copy_(2, slots, packed)
        torch.cuda.synchronize()
        self._record(metadata, packed)

    def _record(self, metadata: KVTransferMetadata, packed: torch.Tensor):
        self.stats.transfers += 1
        self.stats.tokens += metadata.total_tokens
        self.stats.bytes += packed.numel() * packed.element_size()
