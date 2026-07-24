import unittest
from types import SimpleNamespace

from nanovllm.engine.pd_scheduler import DecodeScheduler, PrefillScheduler
from nanovllm.engine.proxy import _slots
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def config(budget=256, prefix=True, chunked=True):
    return SimpleNamespace(
        max_num_seqs=8,
        max_num_batched_tokens=budget,
        eos=-1,
        kvcache_block_size=256,
        enable_prefix_cache=prefix,
        enable_chunked_prefill=chunked,
    )


class PDSchedulersTest(unittest.TestCase):
    def setUp(self):
        Sequence.block_size = 256

    def test_slots_follow_paged_blocks(self):
        slots = _slots([3, 7], 0, 258, 256)
        self.assertEqual(slots[:2], [768, 769])
        self.assertEqual(slots[255:258], [1023, 1792, 1793])

    def test_chunked_prefill_only_handoffs_after_last_chunk(self):
        scheduler = PrefillScheduler(config(), 8)
        seq = Sequence(list(range(600)), SamplingParams(max_tokens=2, ignore_eos=True))
        scheduler.add(seq)
        chunk_sizes = []
        ready = []
        while scheduler.waiting:
            output = scheduler.schedule()
            chunk_sizes.append(output.num_tokens)
            ready.extend(scheduler.postprocess(output, [7]))
        self.assertEqual(chunk_sizes, [256, 256, 88])
        self.assertEqual(ready, [seq])
        self.assertEqual(seq.handoff_context_tokens, 600)

    def test_prefix_cache_hits_on_both_sides(self):
        cfg = config(budget=1024)
        prefill = PrefillScheduler(cfg, 12)
        first = Sequence(list(range(600)), SamplingParams(max_tokens=2, ignore_eos=True))
        prefill.add(first)
        output = prefill.schedule()
        prefill.postprocess(output, [7])
        source = list(first.block_table)
        first.block_table = []
        prefill.release(first, source)

        second = Sequence(list(range(600)), SamplingParams(max_tokens=2, ignore_eos=True))
        prefill.add(second)
        output = prefill.schedule()
        self.assertEqual(output.num_tokens, 88)
        self.assertEqual(second.num_cached_tokens, 512)
        prefill.postprocess(output, [7])

        decode = DecodeScheduler(cfg, 12)
        second.block_table = []
        cached, _ = decode.prepare_handoff(second)
        self.assertEqual(cached, 0)
        decode.finish_handoff(second, cached)
        decode.block_manager.deallocate(second)
        decode.running.clear()

        repeated = Sequence(list(range(600)), SamplingParams(max_tokens=2, ignore_eos=True))
        repeated.handoff_context_tokens = 600
        repeated.append_token(7)
        cached, _ = decode.prepare_handoff(repeated)
        self.assertEqual(cached, 512)

    def test_decode_does_not_duplicate_boundary_block(self):
        scheduler = DecodeScheduler(config(budget=1024), 8)
        seq = Sequence(list(range(256)), SamplingParams(max_tokens=3, ignore_eos=True))
        seq.handoff_context_tokens = 256
        seq.append_token(7)
        cached, _ = scheduler.prepare_handoff(seq)
        scheduler.finish_handoff(seq, cached)
        self.assertEqual(len(seq.block_table), 2)
        output = scheduler.schedule()
        self.assertEqual(len(output.sequences[0].block_table), 2)

    def test_prefix_cache_can_be_disabled(self):
        scheduler = PrefillScheduler(config(budget=1024, prefix=False), 8)
        for _ in range(2):
            seq = Sequence(list(range(300)), SamplingParams(max_tokens=1, ignore_eos=True))
            scheduler.add(seq)
            output = scheduler.schedule()
            self.assertEqual(output.num_tokens, 300)
            scheduler.postprocess(output, [7])
            scheduler.block_manager.deallocate(seq)
            scheduler.transferring.pop(seq.seq_id)
        self.assertEqual(scheduler.stats.matched_blocks, 0)

    def test_rejects_long_prompt_when_chunking_disabled(self):
        scheduler = PrefillScheduler(config(chunked=False), 8)
        with self.assertRaisesRegex(ValueError, "chunked prefill"):
            scheduler.add(Sequence(list(range(257)), SamplingParams()))


if __name__ == "__main__":
    unittest.main()
