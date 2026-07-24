from collections import deque
from dataclasses import dataclass

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.interfaces import BaseScheduler, ExecutionPhase, SchedulerOutput
from nanovllm.engine.sequence import Sequence, SequenceStatus


@dataclass
class CacheStats:
    matched_blocks: int = 0
    computed_tokens: int = 0
    chunks: int = 0


class PrefillScheduler(BaseScheduler):
    def __init__(self, config, num_blocks: int):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.block_size = config.kvcache_block_size
        self.enable_prefix_cache = config.enable_prefix_cache
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.block_manager = BlockManager(num_blocks, self.block_size)
        self.waiting: deque[Sequence] = deque()
        self.transferring: dict[int, Sequence] = {}
        self.stats = CacheStats()

    def is_finished(self):
        return not self.waiting and not self.transferring

    def add(self, seq: Sequence):
        if not self.enable_chunked_prefill and len(seq) > self.max_num_batched_tokens:
            raise ValueError(
                f"prompt requires chunked prefill ({len(seq)} > {self.max_num_batched_tokens})"
            )
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.num_cached_tokens = 0
        seq.num_scheduled_tokens = 0
        seq.block_table.clear()
        self.waiting.append(seq)

    def _allocate(self, seq: Sequence) -> bool:
        if self.enable_prefix_cache:
            matched = self.block_manager.can_allocate(seq)
        else:
            matched = 0 if len(self.block_manager.free_block_ids) >= seq.num_blocks else -1
        if matched == -1:
            return False
        self.block_manager.allocate(seq, matched)
        self.stats.matched_blocks += matched
        return True

    def schedule(self) -> SchedulerOutput:
        scheduled = []
        num_tokens = 0
        while self.waiting and len(scheduled) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_tokens
            if remaining == 0:
                break
            if not seq.block_table and not self._allocate(seq):
                break
            needed = len(seq) - seq.num_cached_tokens
            if needed > remaining and scheduled:
                break
            if needed > remaining and not self.enable_chunked_prefill:
                raise ValueError("chunked prefill is disabled")
            seq.num_scheduled_tokens = min(needed, remaining)
            num_tokens += seq.num_scheduled_tokens
            self.stats.computed_tokens += seq.num_scheduled_tokens
            self.stats.chunks += 1
            if seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq):
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.transferring[seq.seq_id] = seq
            scheduled.append(seq)
        if not scheduled:
            raise RuntimeError("prefill scheduler has work but cannot allocate KV blocks")
        return SchedulerOutput(scheduled, ExecutionPhase.PREFILL, num_tokens)

    def postprocess(self, output: SchedulerOutput, token_ids: list[int]) -> list[Sequence]:
        ready = []
        for seq, token_id in zip(output.sequences, token_ids):
            if self.enable_prefix_cache:
                self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.num_cached_tokens < len(seq):
                continue
            seq.handoff_context_tokens = len(seq)
            seq.append_token(token_id)
            ready.append(seq)
        return ready

    def release(self, seq: Sequence, source_block_table: list[int]):
        destination_block_table = seq.block_table
        seq.block_table = source_block_table
        self.block_manager.deallocate(seq)
        seq.block_table = destination_block_table
        self.transferring.pop(seq.seq_id, None)


class DecodeScheduler(BaseScheduler):
    def __init__(self, config, num_blocks: int):
        self.max_num_seqs = config.max_num_seqs
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.enable_prefix_cache = config.enable_prefix_cache
        self.block_manager = BlockManager(num_blocks, self.block_size)
        self.running: deque[Sequence] = deque()
        self.loading: dict[int, Sequence] = {}
        self.stats = CacheStats()

    def is_finished(self):
        return not self.running and not self.loading

    def add(self, seq: Sequence):
        self.running.append(seq)

    def _try_allocate(self, seq: Sequence) -> int:
        if self.enable_prefix_cache:
            matched = self.block_manager.can_allocate(seq)
        else:
            matched = 0 if len(self.block_manager.free_block_ids) >= seq.num_blocks else -1
        if matched != -1:
            self.block_manager.allocate(seq, matched)
            self.stats.matched_blocks += matched
        return matched

    def prepare_handoff(self, seq: Sequence) -> tuple[int, list[Sequence]]:
        seq.block_table = []
        seq.num_cached_tokens = 0
        preempted = []
        matched = self._try_allocate(seq)
        while matched == -1 and self.running:
            victim = self.running.pop()
            self.preempt(victim)
            preempted.append(victim)
            matched = self._try_allocate(seq)
        if matched == -1:
            raise RuntimeError("decode KV cache is too small for this request")
        self.loading[seq.seq_id] = seq
        return min(matched * self.block_size, seq.handoff_context_tokens), preempted

    def finish_handoff(self, seq: Sequence, cached_tokens: int):
        transferred = seq.handoff_context_tokens - cached_tokens
        seq.num_cached_tokens = cached_tokens
        seq.num_scheduled_tokens = transferred
        if self.enable_prefix_cache:
            self.block_manager.hash_blocks(seq)
        seq.num_cached_tokens = seq.handoff_context_tokens
        seq.num_scheduled_tokens = 0
        seq.is_prefill = False
        seq.status = SequenceStatus.RUNNING
        self.loading.pop(seq.seq_id, None)
        self.running.append(seq)

    def schedule(self) -> SchedulerOutput:
        scheduled = []
        preempted = []
        while self.running and len(scheduled) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    victim = self.running.pop()
                    self.preempt(victim)
                    preempted.append(victim)
                else:
                    self.preempt(seq)
                    preempted.append(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                self.block_manager.may_append(seq)
                scheduled.append(seq)
        self.running.extendleft(reversed(scheduled))
        return SchedulerOutput(scheduled, ExecutionPhase.DECODE, -len(scheduled), preempted)

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)

    def postprocess(self, output: SchedulerOutput, token_ids: list[int]) -> list[Sequence]:
        finished = []
        for seq, token_id in zip(output.sequences, token_ids):
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            seq.append_token(token_id)
            if ((not seq.ignore_eos and token_id == self.eos) or
                    seq.num_completion_tokens == seq.max_tokens):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                finished.append(seq)
        return finished
