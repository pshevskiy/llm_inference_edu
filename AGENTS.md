# Project instructions for Codex

## Purpose and scope

Build a small, understandable Rust inference engine. The owner's main goal is to learn inference internals deeply and maintain the code without an AI coding agent. Performance and low memory use matter, but must be demonstrated.

Read `docs/PLAN.md` before editing. It is the authoritative requirements, roadmap, and progress record. It supersedes the older integration-first plan. Target the existing M1 Pro with 16 GB first; iPhone 14 is a later physical test device. CPU reference first, Core ML/Neural Engine qualification later, GPU/Metal only after explicit authorization.

## One task at a time

Execute only the ticket explicitly requested. When told to start this project, begin with T00. When told to continue, inspect the evidence and choose only the next prerequisite-satisfied ticket. Never implement the entire roadmap or advance to a second ticket in the same run without explicit authorization.

For each ticket: briefly explain the mechanism, implement a small working slice, run tests, explain the central code and one concrete trace, give one bounded learning exercise, and stop. Do not solve the exercise automatically. When the owner requests exercise-only mode, provide a contract/tests and review their work rather than implementing the answer.

Keep code and explanations in English unless the owner requests otherwise. Use Rust for production; Python only for independent numerical references/export tools. Do not generate parallel production engines in multiple languages.

## Simplicity and ownership

Inspect the repository and existing instructions before editing. Preserve uncommitted changes. Do not overwrite existing `AGENTS.md` rules silently, perform destructive Git actions, commit/push, or change credentials/signing settings without authorization.

Start with one Rust package and ordinary functions over explicit buffers. No generic tensor framework, graph compiler, autograd, backend plugin system, async framework, server, UI, FFI, or empty future packages in T00. Add abstractions only after working implementations justify them.

Do not substitute calls to MLX, Candle, llama.cpp, or Core ML for our Rust forward pass. External libraries may supply tokenizer/format parsing or independent references. Keep the scalar reference available when optimizing. No `unsafe` initially; later SIMD/FFI requires local safety invariants and tests.

Introduce dependencies only for the current ticket and explain why. Prefer established parsers and tokenizers over custom formats. No heavyweight ML installation or model download for ordinary unit tests.

## Correctness and resources

Validate shapes, lengths, indices, and checked dimension arithmetic before writes. Document tensor layout and transposition conventions. Test operator outputs, intermediate activations, and teacher-forced logits; plausible generated prose is not sufficient.

Never widen numerical tolerances merely to pass tests. Preserve exact token/position/cache invariants, including emitted-but-not-yet-evaluated tokens. On partial failures, invalidate unsafe state instead of reporting a valid cache. Never silently truncate history or change precision/model/backend to fit a benchmark.

Separate logical memory estimates, physical measurements, and file sizes. Keep model weights, large artifacts, secrets, and private prompts out of Git. Require explicit opt-in before downloading large models or running sustained hardware tests. Use no private ANE API. CPU+ANE allowed does not prove ANE execution; MLX discovery does not imply ANE compatibility.

## Verification and evidence

Run applicable ordinary checks:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Use release builds for performance measurements once the benchmark exists. Keep numerical tests independent of network and Apple hardware. Real-model/reference/hardware tests are opt-in and separately documented.

When a tool or device is unavailable, report `not run` or `blocked`, the reason, and the exact follow-up command or manual check. Never invent test output, benchmark numbers, ANE placement, iPhone validation, or owner approval. Record actual runtime/device identity; a cloud runner is not the owner's Mac.

Update only the current ticket in `docs/PLAN.md` with changed files, commands/results, limitations, and evidence paths. Keep owner review pending until acknowledged. Finish with the key code explanation, one trace/exercise, and the next proposed ticket—not another implementation.
