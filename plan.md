# Async GPT-2 Forward With Hook Pause/Resume

## Goal
- Replace synchronous hook waiting (entire batch blocked by slowest job) with an async intervention flow.
- Keep forward progress for completed jobs while pausing unresolved jobs at hookpoints.
- Resume paused jobs on later passes using cached tensors at the exact paused hookpoint.

## Scope and Constraints
- GPT-2 only.
- Hookpoints supported:
  - `transformer.h.{layer}.attn`
  - `transformer.h.{layer}.mlp`
  - `transformer.h.{layer}.resid`
- Async path is default behavior in `WrapperModel`.
- Fail fast on unsupported model types or invalid hookpoints.
- Toy/scratch implementation: minimal defensive checks and no production hardening.

## Planned Architecture

### Runtime scheduler model
- Maintain a persistent set of live jobs (not one-and-done static batches).
- For each cycle:
  1. Fill from queue using available workers.
  2. Run one packed forward pass over live jobs.
  3. Jobs that complete this cycle finish and release workers.
  4. Paused jobs remain live and are retried next cycle.

### Per-job async hook state
- Track, per live job:
  - paused/pending hookpoint
  - pending hook response
  - cached module output `[seq, d_model]`
  - cached residual tensor `[seq, d_model]` for `attn`/`mlp`
- Worker request lifecycle remains one start/finish per request:
  - `start_request` once when job becomes live
  - multiple `apply_hooks`
  - `finish_request` once when done/failed

### Hook wait semantics
- At each hookpoint:
  1. Dispatch all hook calls for jobs participating at that hookpoint.
  2. Wait up to `hook_wait_ms`.
  3. If none finished, keep waiting until at least one finishes.
  4. After first completion, wait one extra `hook_wait_ms` grace window.
  5. Mark unresolved jobs as paused and drop them from this pass.

### GPT-2 forward control
- Patch GPT-2 blocks to enforce hook order in every layer:
  - `attn -> mlp -> resid`
- Hook dispatcher composes output as:
  - `attn`/`mlp`: `(write_result or cached_module_output) + cached_residual`
  - `resid`: `(write_result or cached_resid_output)`
- Rebuild packed hidden state each hook step from unpaused + resolved jobs.

## What Was Implemented

## 1) Runtime refactor (`src/batched/runtime.py`)
- Added `hook_wait_ms: int = 50` to `WrapperModel`.
- Added GPT-2-only validation via `config.model_type == "gpt2"`.
- Removed old global module `register_forward_hook` path and synchronous full-batch waiting logic.
- Added `LiveJob` state with cached tensors and pending hook metadata.
- Added persistent live-job scheduling:
  - queue claims + worker assignment
  - start request on worker once
  - repeated packed forward cycles over live jobs
  - finish request once and recycle worker
- Added strict hookpoint validation regex:
  - `^transformer\.h\.(\d+)\.(attn|mlp|resid)$`
- Implemented async hook dispatcher:
  - sends `apply_hooks`
  - polls worker responses with timeout/grace semantics
  - pauses unresolved jobs by excluding them from rebuilt packed tensor
  - resumes paused jobs when their pending hookpoint is reached again
- Added handling so jobs paused at one hookpoint continue through non-matching later hookpoints unchanged until they return to their paused hookpoint.

## 2) GPT-2 patching updates (`src/batched/gpt2.py`)
- Replaced old `Batch/Job/create_gpt2_forward` helper model with async block patch API:
  - `patch_gpt2_blocks_for_async(transformer, dispatch_hook)`
  - `create_async_gpt2_block_forward(...)`
- Kept and reused trimmed transformer forward patch:
  - `patch_gpt2_transformer_for_trimmed_sequences(...)`
- Each GPT-2 block now calls dispatcher in exact order:
  - `attn`, then `mlp`, then `resid`.

## 3) Example compatibility (`examples/test.py`)
- Updated imports and usage to the new GPT-2 patch API.
- Removed references to deleted `Batch/Job/create_gpt2_forward`.
- Added a simple passthrough dispatcher example.

## Design Rationale (How I Thought About It)
- The bottleneck was not model compute itself, but synchronization at each hookpoint. So the core change had to move from static “batch request” semantics to persistent “live job” semantics.
- To keep behavior predictable and easy to iterate on, I bound scope to GPT-2 and explicitly controlled hook order at the block level (`attn -> mlp -> resid`) instead of trying to infer generic hook ordering across arbitrary models.
- Pausing needed to be lossless for resumed computations, so I cached per-job tensors at the paused hookpoint (module output + residual when needed) and recomposed outputs deterministically on resume.
- I kept worker lifecycle simple (one worker per live request) because that minimizes protocol complexity while still enabling async progress across jobs.
- I used fail-fast assertions for unsupported shapes and states, matching the “toy/scratch” constraint.

## Config / API Change Summary
- `WrapperModel(..., hook_wait_ms=50, ...)`
- Default runtime is async GPT-2 path.
- Backward compatibility with old synchronous non-GPT-2 path was intentionally removed.

## Verification Performed
- Syntax-only verification:
  - `python3 -m compileall src examples`
- No runtime GPU/CUDA validation performed in this environment.

