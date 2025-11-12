# Repository Guidelines

## Project Structure & Module Organization
- Core package lives in `src/hiddenlayer_strands/`, with `HiddenlayerStrands` wrapping the Strands agent event loop to broker HiddenLayer moderation (`__init__.py`).
- Tests sit under `tests/`, mirroring key behaviours (blocking, redaction, streaming). Use them as templates when adding new coverage.
- Tooling and dependency metadata are maintained in `pyproject.toml` and `uv.lock`; no code should bypass this configuration.

## Build, Test & Development Commands
- `uv sync --dev` installs runtime and dev dependencies into the active environment; run after editing dependencies or cloning the repo.
- `uv run pytest` executes the full suite, including async streaming scenarios powered by `pytest-asyncio`.
- `uv run pytest tests/test_hiddenlayer.py -k streaming` focuses on SSE streaming regressions before shipping UX changes.
- `uv build` produces sdist/wheel artifacts via the `uv_build` backend; verify the build when you touch packaging metadata.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and keep type hints in public APIs (the package is typed via `py.typed`).
- Preserve existing prefixes (`hl_` for HiddenLayer context, `agent` for Strands objects) so new parameters read consistently.
- Keep side effects explicit: prefer returning sanitized content to mutating shared state, and replace `print` debugging with structured logging when possible.

## Testing Guidelines
- Add `pytest` tests alongside the existing file, naming them `test_<feature>()`. Async flows should use `@pytest.mark.asyncio`.
- Extend fixtures instead of duplicating agent setup; ensure both blocking and benign paths stay covered.
- When mocking HiddenLayer responses, assert on both the agent output and any redacted message mutations to avoid regressions.

## Commit & Pull Request Guidelines
- Write imperative, concise commit summaries (e.g., `Add structured output redaction tests`), mirroring the current history.
- Reference linked issues and list verification commands (`uv run pytest ...`) in PR descriptions.
- Provide reviewers with example prompts and expected transcripts when altering `HiddenlayerStrands.hl_event_loop_cycle`.

## Security & Configuration Tips
- Inject `model`, `client_id`, and `client_secret` via environment variables or secrets managers; never hard-code credentials.
- Avoid logging raw user or agent content—redact sensitive strings before emitting telemetry or debug output.
