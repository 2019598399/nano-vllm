# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `nanovllm/`. Public APIs are exposed through
`nanovllm/__init__.py`, while `llm.py`, `config.py`, and `sampling_params.py`
define the main user-facing behavior. Runtime orchestration is under
`nanovllm/engine/`; reusable CUDA-aware components are in `nanovllm/layers/`;
model implementations belong in `nanovllm/models/`; and loading or context
helpers live in `nanovllm/utils/`.

Use `example.py` for a minimal inference run and `bench.py` for performance
comparisons. Static project images are stored in `assets/`. Automated tests live under `tests/`; mirror source module names when adding
new coverage.

## Build, Test, and Development Commands

The project supports Python 3.10–3.12 and requires a CUDA-capable environment.

```bash
python -m venv .venv
./.venv/bin/python -m pip install -e .
./.venv/bin/python example.py
./.venv/bin/python bench.py
```

The editable install makes local package changes immediately available.
`example.py` loads Qwen3-0.6B from `~/huggingface/Qwen3-0.6B/`; update that
path locally if model weights are elsewhere. The benchmark is GPU-intensive,
so record hardware, model, and sampling settings with results.

No test runner is configured yet. For new tests, prefer `pytest`, place files
under `tests/`, and run `./.venv/bin/python -m pytest`.

## Coding Style & Naming Conventions

Follow standard Python conventions: four-space indentation, `snake_case` for
functions and variables, `PascalCase` for classes, and `UPPER_CASE` for
constants. Keep modules focused and preserve the compact, readable style of
the existing implementation. Add type hints where they clarify public APIs or
tensor shapes. No formatter or linter is enforced; avoid unrelated formatting
changes and ensure imports are grouped consistently.

## Testing Guidelines

Cover scheduler state transitions, block allocation, sequence lifecycle, and
sampling behavior with small deterministic unit tests. Name files
`test_<module>.py` and tests `test_<behavior>()`. GPU-specific tests should
skip cleanly when CUDA is unavailable and document model or memory
requirements.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries such as `fix cache hit`, with
optional scopes like `fix(scheduler): ...`. Keep each commit focused. Pull
requests should explain the behavior change, motivation, verification
commands, and GPU/model configuration when performance or numerical output is
affected. Link relevant issues and include benchmark comparisons for
optimization work.
