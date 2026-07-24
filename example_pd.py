import os

from nanovllm import LLM, SamplingParams


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(
        path,
        pd_disaggregation=True,
        enforce_eager=False,
        enable_prefix_cache=True,
        enable_chunked_prefill=True,
    )
    outputs = llm.generate(
        ["Hello, Nano-vLLM.", "Name three prime numbers."],
        SamplingParams(temperature=0.6, max_tokens=16),
    )
    for output in outputs:
        print(output["text"])
    llm.exit()


if __name__ == "__main__":
    main()
