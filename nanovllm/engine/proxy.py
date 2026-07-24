from multiprocessing.connection import wait
from time import perf_counter

import torch
import torch.multiprocessing as mp

from nanovllm.engine.interfaces import (
    EngineProxy,
    EngineStepOutput,
    ExecutionPhase,
    TokenEvent,
)
from nanovllm.engine.kv_connector import KVTransferMetadata, KVTransferStats
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.pd_scheduler import DecodeScheduler, PrefillScheduler
from nanovllm.engine.pd_worker import pd_worker_main
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def _slots(block_table: list[int], start: int, end: int, block_size: int) -> list[int]:
    return [block_table[pos // block_size] * block_size + pos % block_size for pos in range(start, end)]


class LocalEngineProxy(EngineProxy):
    def __init__(self, config):
        self.closed = False
        self.processes = []
        self.events = []
        ctx = mp.get_context("spawn")
        for rank in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, rank, event))
            process.start()
            self.processes.append(process)
            self.events.append(event)
        self.runner = ModelRunner(config, 0, self.events)
        self.scheduler = Scheduler(config)

    def add(self, seq: Sequence):
        self.scheduler.add(seq)

    def step(self) -> EngineStepOutput:
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.runner.call("run", seqs, is_prefill)
        timestamp = perf_counter()
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        finished = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        phase = ExecutionPhase.PREFILL if is_prefill else ExecutionPhase.DECODE
        events = [
            TokenEvent(seq.seq_id, seq.last_token, timestamp, phase)
            for seq in seqs
            if not is_prefill or seq.num_completion_tokens > 0
        ]
        return EngineStepOutput(finished, num_tokens, events)

    def is_finished(self):
        return self.scheduler.is_finished()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.runner.call("exit")
        for process in self.processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


class _PDWorkerClient:
    def __init__(self, ctx, config, role: str):
        parent, child = ctx.Pipe()
        self.connection = parent
        self.process = ctx.Process(target=pd_worker_main, args=(config, role, child))
        self.process.start()

    def ready(self):
        return self.connection.recv()

    def send(self, command: str, payload=None):
        self.connection.send((command, payload))

    def recv(self):
        return self.connection.recv()

    def poll(self):
        return self.connection.poll()

    def call(self, command: str, payload=None):
        self.send(command, payload)
        return self.recv()


class PDProxy(EngineProxy):
    """CPU-only top-level router for the two internal PD engines."""

    def __init__(self, config):
        if torch.cuda.device_count() < 2:
            raise RuntimeError("PD mode requires at least two visible CUDA devices")
        ctx = mp.get_context("spawn")
        self.prefill_worker = _PDWorkerClient(ctx, config, "prefill")
        self.decode_worker = _PDWorkerClient(ctx, config, "decode")
        prefill_ready = self.prefill_worker.ready()
        decode_ready = self.decode_worker.ready()
        self.prefill_scheduler = PrefillScheduler(config, prefill_ready["num_blocks"])
        self.decode_scheduler = DecodeScheduler(config, decode_ready["num_blocks"])
        self.transfer_stats = KVTransferStats()
        self.transfer_seconds = 0.0
        self.graph_replays = 0
        self.prefill_inflight = None
        self.decode_inflight = None
        self.pending_ready = []
        self.closed = False
        print("PD proxy: GPU 0 = prefill engine, GPU 1 = decode engine")

    def add(self, seq: Sequence):
        self.prefill_scheduler.add(seq)

    def is_finished(self):
        return self.prefill_scheduler.is_finished() and self.decode_scheduler.is_finished()

    def _build_metadata(self, entries) -> KVTransferMetadata:
        source_slots = []
        destination_slots = []
        request_ids = []
        token_counts = []
        block_size = self.prefill_scheduler.block_size
        for seq, source_table, cached_tokens in entries:
            end = seq.handoff_context_tokens
            request_ids.append(seq.seq_id)
            token_counts.append(end - cached_tokens)
            source_slots.extend(_slots(source_table, cached_tokens, end, block_size))
            destination_slots.extend(_slots(seq.block_table, cached_tokens, end, block_size))
        return KVTransferMetadata(request_ids, source_slots, destination_slots, token_counts)

    def _route_ready(self, ready: list[Sequence]) -> list[tuple[int, list[int]]]:
        finished = []
        entries = []
        source_tables = {}
        for seq in ready:
            source_tables[seq.seq_id] = list(seq.block_table)
            if seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.prefill_scheduler.block_manager.deallocate(seq)
                self.prefill_scheduler.transferring.pop(seq.seq_id, None)
                finished.append((seq.seq_id, seq.completion_token_ids))
                continue
            cached_tokens, preempted = self.decode_scheduler.prepare_handoff(seq)
            for victim in preempted:
                self.prefill_scheduler.add(victim)
            entries.append((seq, source_tables[seq.seq_id], cached_tokens))

        if entries:
            metadata = self._build_metadata(entries)
            if metadata.total_tokens:
                started = perf_counter()
                self.decode_worker.send("recv_kv", metadata)
                self.prefill_worker.send("send_kv", metadata)
                producer_stats = self.prefill_worker.recv()
                self.decode_worker.recv()
                self.transfer_seconds = getattr(self, "transfer_seconds", 0.0) + perf_counter() - started
                self.transfer_stats = producer_stats
            for seq, source_table, cached_tokens in entries:
                self.prefill_scheduler.release(seq, source_table)
                self.decode_scheduler.finish_handoff(seq, cached_tokens)
        return finished

    def step(self) -> EngineStepOutput:
        finished = []
        events = []
        prefill_tokens = 0
        decode_tokens = 0

        while True:
            # Timestamp decode first when both workers complete together.
            if self.decode_inflight is not None and self.decode_worker.poll():
                output = self.decode_inflight
                result = self.decode_worker.recv()
                timestamp = perf_counter()
                self.decode_inflight = None
                self.graph_replays = result["graph_replays"]
                finished_seqs = self.decode_scheduler.postprocess(output, result["tokens"])
                finished.extend((seq.seq_id, seq.completion_token_ids) for seq in finished_seqs)
                events.extend(
                    TokenEvent(seq.seq_id, seq.last_token, timestamp, ExecutionPhase.DECODE)
                    for seq in output.sequences
                )
                decode_tokens += len(output.sequences)

            if self.prefill_inflight is not None and self.prefill_worker.poll():
                output = self.prefill_inflight
                result = self.prefill_worker.recv()
                timestamp = perf_counter()
                self.prefill_inflight = None
                ready = self.prefill_scheduler.postprocess(output, result["tokens"])
                self.pending_ready.extend(ready)
                prefill_tokens += output.num_tokens

            # Handoff allocation can preempt decode sequences. Wait until both
            # workers are idle and prioritize the transfer to avoid starvation.
            if (
                self.pending_ready
                and self.prefill_inflight is None
                and self.decode_inflight is None
            ):
                ready, self.pending_ready = self.pending_ready, []
                finished.extend(self._route_ready(ready))
                timestamp = perf_counter()
                events.extend(
                    TokenEvent(seq.seq_id, seq.last_token, timestamp, ExecutionPhase.PREFILL)
                    for seq in ready
                )

            if (
                self.prefill_inflight is None
                and not self.pending_ready
                and self.prefill_scheduler.waiting
            ):
                output = self.prefill_scheduler.schedule()
                self.prefill_worker.send("run", (output.sequences, True))
                self.prefill_inflight = output

            if (
                self.decode_inflight is None
                and not self.pending_ready
                and self.decode_scheduler.running
            ):
                output = self.decode_scheduler.schedule()
                for seq in output.preempted:
                    self.prefill_scheduler.add(seq)
                if output.sequences:
                    self.decode_worker.send("run", (output.sequences, False))
                    self.decode_inflight = output

            if events or finished:
                num_tokens = -decode_tokens if decode_tokens else prefill_tokens
                return EngineStepOutput(finished, num_tokens, events)

            connections = []
            if self.decode_inflight is not None:
                connections.append(self.decode_worker.connection)
            if self.prefill_inflight is not None:
                connections.append(self.prefill_worker.connection)
            if connections:
                wait(connections)
                continue
            if self.is_finished():
                return EngineStepOutput([], 0, [])
            raise RuntimeError("PD proxy has pending work but cannot make progress")

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.prefill_worker.send("exit")
        self.decode_worker.send("exit")
        for worker in (self.prefill_worker, self.decode_worker):
            worker.process.join(timeout=30)
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(timeout=5)
            worker.connection.close()
