import atexit
from dataclasses import fields
from time import perf_counter

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.proxy import LocalEngineProxy, PDProxy
from nanovllm.engine.sequence import Sequence


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.engine_proxy = PDProxy(config) if config.pd_disaggregation else LocalEngineProxy(config)
        self.closed = False
        atexit.register(self.exit)

    def exit(self):
        if self.closed:
            return
        self.closed = True
        self.engine_proxy.close()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if len(prompt) + sampling_params.max_tokens > self.config.max_model_len:
            raise ValueError(
                f"request exceeds max_model_len: {len(prompt)} + "
                f"{sampling_params.max_tokens} > {self.config.max_model_len}"
            )
        seq = Sequence(prompt, sampling_params)
        self.engine_proxy.add(seq)
        return seq.seq_id

    def step(self):
        output = self.step_with_events()
        return output.finished, output.num_tokens

    def step_with_events(self):
        return self.engine_proxy.step()

    def is_finished(self):
        return self.engine_proxy.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
