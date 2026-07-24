import unittest
from unittest.mock import patch
from types import SimpleNamespace

from nanovllm.engine.interfaces import ExecutionPhase, SchedulerOutput
from nanovllm.engine.kv_connector import KVTransferStats
from nanovllm.engine.proxy import PDProxy
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class _Worker:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def send(self, command, metadata):
        self.events.append((self.name, command, metadata.total_tokens))

    def recv(self):
        self.events.append((self.name, "done"))
        return KVTransferStats(1, 44, 100)


class ProxyRoutingTest(unittest.TestCase):
    def test_routes_prefill_metadata_connector_decode(self):
        Sequence.block_size = 256
        events = []
        proxy = PDProxy.__new__(PDProxy)
        proxy.prefill_worker = _Worker("P", events)
        proxy.decode_worker = _Worker("D", events)
        proxy.transfer_stats = KVTransferStats()

        class Prefill:
            block_size = 256

            def release(self, seq, source):
                events.append(("P-scheduler", "release", source))

        class Decode:
            def prepare_handoff(self, seq):
                events.append(("D-scheduler", "allocate"))
                seq.block_table = [5, 6]
                return 256, []

            def finish_handoff(self, seq, cached):
                events.append(("D-scheduler", "ready", cached))

        proxy.prefill_scheduler = Prefill()
        proxy.decode_scheduler = Decode()
        seq = Sequence(list(range(300)), SamplingParams(max_tokens=2, ignore_eos=True))
        seq.block_table = [1, 2]
        seq.handoff_context_tokens = 300
        seq.append_token(7)

        self.assertEqual(proxy._route_ready([seq]), [])
        self.assertEqual(events[0], ("D-scheduler", "allocate"))
        self.assertEqual(events[1], ("D", "recv_kv", 44))
        self.assertEqual(events[2], ("P", "send_kv", 44))
        self.assertEqual(events[-2], ("P-scheduler", "release", [1, 2]))
        self.assertEqual(events[-1], ("D-scheduler", "ready", 256))


class _AsyncWorker:
    def __init__(self, ready):
        self.ready = ready
        self.connection = object()
        self.sent = []

    def send(self, command, payload=None):
        self.sent.append((command, payload))

    def poll(self):
        return self.ready and bool(self.sent)

    def recv(self):
        self.ready = False
        return {"tokens": [9], "graph_replays": 1}


class AsyncProxyTest(unittest.TestCase):
    def test_dispatches_prefill_and_decode_concurrently(self):
        Sequence.block_size = 256
        prefill_seq = Sequence([1, 2], SamplingParams(max_tokens=2, ignore_eos=True))
        decode_seq = Sequence([3, 4], SamplingParams(max_tokens=2, ignore_eos=True))

        class Prefill:
            def __init__(self):
                self.waiting = [prefill_seq]
                self.transferring = {}

            def schedule(self):
                self.waiting.clear()
                return SchedulerOutput([prefill_seq], ExecutionPhase.PREFILL, 2)

            def postprocess(self, output, tokens):
                return []

            def is_finished(self):
                return False

        class Decode:
            def __init__(self):
                self.running = [decode_seq]
                self.loading = {}

            def schedule(self):
                self.running.clear()
                return SchedulerOutput([decode_seq], ExecutionPhase.DECODE, -1)

            def postprocess(self, output, tokens):
                decode_seq.append_token(tokens[0])
                return []

            def is_finished(self):
                return True

        proxy = PDProxy.__new__(PDProxy)
        proxy.prefill_worker = _AsyncWorker(ready=False)
        proxy.decode_worker = _AsyncWorker(ready=True)
        proxy.prefill_scheduler = Prefill()
        proxy.decode_scheduler = Decode()
        proxy.prefill_inflight = None
        proxy.decode_inflight = None
        proxy.pending_ready = []
        proxy.graph_replays = 0

        with patch("nanovllm.engine.proxy.wait", return_value=None):
            output = proxy.step()

        self.assertEqual(proxy.prefill_worker.sent[0][0], "run")
        self.assertEqual(proxy.decode_worker.sent[0][0], "run")
        self.assertIsNotNone(proxy.prefill_inflight)
        self.assertIsNone(proxy.decode_inflight)
        self.assertEqual(output.num_tokens, -1)
        self.assertEqual([event.seq_id for event in output.token_events], [decode_seq.seq_id])
        self.assertEqual(output.token_events[0].phase, ExecutionPhase.DECODE)


if __name__ == "__main__":
    unittest.main()
