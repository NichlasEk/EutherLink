# Local Agent And Resource Plan

## Goal

Build EutherLink into the local resource manager for heavy AI work on the RTX
4090 machine, so EutherOxide/EutherBooks and CLI clients can safely use local
TTS, local LLM, file operations, and scaffolding without GPU jobs fighting each
other.

The immediate problem is coordination between:

- EutherLink TTS workers: VoxCPM2 and dots.tts.
- Ollama local LLM: `qwen3-coder:30b`.
- EutherBooks/EutherOxide UI state for long running audiobook jobs.
- Remote CLI access from other machines.

## Current State

Local machine:

- EutherLink runs on `0.0.0.0:8765`.
- EutherLink health endpoint is `GET /health`.
- TTS job API already exists:
  - `POST /v1/tts/jobs`
  - `GET /v1/tts/jobs/{id}`
  - `DELETE /v1/tts/jobs/{id}`
  - `GET /v1/tts/jobs/{id}/audio`
  - `GET /v1/tts/jobs/{id}/partials/{filename}`
- TTS job status is already persisted under `data/jobs/<job_id>/status.json`.
- Dots worker runs separately on `127.0.0.1:18765`.
- Ollama runs locally on `127.0.0.1:11434`.
- `qwen3-coder:30b` is downloaded in Ollama.

Server:

- EutherOxide/EutherBooks live on `192.168.32.186`.
- EutherBooks is expected behind `/eutherbooks`.
- EutherBooks previously used EutherLink through `model_backend`.
- SSH host config exists locally as `euther-server`, using `~/.ssh/euther_server`.
- SSH currently needs an unlocked key via agent.

Implemented checkpoints:

- Commit `2539999 Add EutherLink resource controls` adds first EutherLink
  resource/job API slice.
- It is pushed to `origin/main`.
- The active EutherLink process needs a restart before these endpoints are live
  on port `8765`.
- The global GPU scheduler slice adds persistent GPU jobs under
  `data/gpu-jobs/` and exposes `/v1/gpu/jobs` for cross-service scheduling.

## Resource API

EutherLink should be the only service that decides when local heavy resources
can be started, stopped, or unloaded.

Implemented in the first slice:

- `GET /v1/resources`
  - Returns TTS, Ollama, and GPU status.
- `GET /v1/tts/jobs`
  - Returns recent in-memory and persisted TTS jobs.
- `POST /v1/resources/dots.tts/stop`
  - Stops tracked Dots worker only when no Dots TTS job is active.
- `POST /v1/resources/voxcpm2/unload`
  - Unloads VoxCPM2 from memory only when no VoxCPM2 TTS job is active.
- `POST /v1/resources/ollama/{model}/stop`
  - Calls `ollama stop <model>`.

Short term follow-up:

- Restart EutherLink so these endpoints are active on `8765`.
- Add test coverage for:
  - `GET /v1/resources`
  - `GET /v1/tts/jobs`
  - stop refusal while matching job is active
  - stop success while idle
- Add `POST /v1/resources/llm/default/stop` as a simpler alias for
  `qwen3-coder:30b`.

## GPU Policy

Initial conservative policy:

- Never kill or unload a model that has an active job.
- Allow stopping idle Dots worker.
- Allow unloading idle VoxCPM2.
- Allow stopping idle Ollama model.
- UI/CLI must show a clear conflict if a user asks to start one heavy model
  while another active job is running.

Suggested resource priority:

1. Active TTS render keeps the GPU.
2. Active LLM request keeps the GPU.
3. Idle models can be stopped/unloaded automatically after confirmation.
4. Prewarm is optional and should not block manual model changes.

Implemented global scheduler slice:

- `POST /v1/gpu/jobs`
  - Creates a persistent queued GPU job and immediately promotes it to
    `running` when no active lease exists.
- `GET /v1/gpu/jobs`
  - Lists recent persistent GPU jobs.
- `GET /v1/gpu/jobs/{id}`
  - Returns current job state.
- `POST /v1/gpu/jobs/{id}/heartbeat`
  - Extends the active lease TTL and updates progress/message.
- `POST /v1/gpu/jobs/{id}/release`
  - Marks the running job done, clears the active lease, and promotes the next
    queued job by priority/age.
- `POST /v1/gpu/jobs/{id}/cancel`
  - Cancels queued/running jobs and promotes the next waiting job when needed.
- The older `/v1/resources/gpu/lease` endpoint remains as a compatibility path
  for clients that need a blocking lease while they migrate to persistent jobs.

Next scheduler hardening:

- Add auth before exposing write endpoints beyond trusted LAN/tunnel contexts.
- Add automatic stale-job retry policy per owner, not just TTL fail.
- Add adapter hooks for ACE, ComfyUI, Dots, VoxCPM2, and Ollama so scheduler can
  prepare/release resources centrally instead of each client doing it manually.
- Add a compact admin view in EutherOxide showing active owner, queue, TTL, and
  last heartbeat.

## Job Model

TTS already has persisted job status. Extend the idea into a generic local job
index.

Job fields:

- `job_id`
- `kind`: `tts`, `llm`, `file_transfer`, `scaffold`, `shell_task`
- `status`: `queued`, `running`, `paused`, `done`, `failed`, `cancelled`
- `progress`
- `message`
- `created_at`
- `updated_at`
- `resource_owner`
- `input_summary`
- `output_refs`
- `resume_refs`
- `error`

Short term:

- Keep `/v1/tts/jobs` as-is for audiobook work.
- Add `/v1/jobs` later as a combined job index.
- Keep TTS job files backward compatible.

## EutherBooks And EutherOxide UI

EutherBooks is the likely place for audiobook job UI. EutherOxide mostly proxies
or hosts the webview.

Needed behavior:

- Show "ongoing jobs" for the selected book.
- Preserve job bookmarks across page refresh.
- Reconnect to EutherLink/EutherBooks job state after app restart.
- Surface partial audio URLs when Dots emits partial chunks.
- Show resource conflict state when the local LLM or TTS worker owns the GPU.

Data to persist:

- In backend:
  - book id
  - selected voice/model backend
  - active EutherBooks job id
  - active EutherLink job id
  - last status snapshot
- In browser/local UI:
  - last selected book
  - last active job id
  - visible "resume job" bookmark

API shape in EutherBooks:

- `GET /api/jobs/active`
- `GET /api/books/{book_id}/jobs`
- `POST /api/books/{book_id}/jobs/{job_id}/bookmark`
- `DELETE /api/books/{book_id}/jobs/{job_id}/bookmark`

Frontend behavior:

- Poll active job status while visible.
- On load, ask backend for active/bookmarked jobs.
- If a job exists, show progress and a resume/download action instead of
  starting from scratch.

## CLI Access

Do not expose raw Ollama or shell execution directly on the LAN.

First safe path:

- Keep Ollama bound to `127.0.0.1:11434`.
- Use SSH tunnel for direct LLM access:
  - `ssh -L 11434:127.0.0.1:11434 euther-server`
- Prefer CLI wrappers that call EutherLink instead of direct Ollama.

Desired CLI commands:

- `euther resources`
- `euther jobs`
- `euther job cancel <id>`
- `euther model stop qwen`
- `euther ask "..."`
- `euther send <file>`
- `euther scaffold <template> <target>`

Future EutherLink endpoints:

- `POST /v1/llm/chat`
- `POST /v1/files/send`
- `POST /v1/scaffolds/{template}`
- `GET /v1/scaffolds`

Safety rules:

- Allowlist operations first.
- No raw destructive shell commands without confirmation.
- Never expose unauthenticated write/shell endpoints outside trusted tunnel.
- Log each operation.

## SSH And Deployment

Local SSH config:

```text
Host euther-server
    HostName 192.168.32.186
    User nichlas
    IdentityFile ~/.ssh/euther_server
    IdentitiesOnly yes
```

Next deploy steps:

1. Start/unlock an SSH agent for `~/.ssh/euther_server`.
2. Verify:
   - `ssh euther-server 'hostname'`
   - `ssh euther-server 'ls /home/nichlas'`
3. Inspect server repos:
   - `/home/nichlas/EutherBooks`
   - `/home/nichlas/EutherOxide`
4. Pull or copy the relevant changes.
5. Restart only affected services.
6. Verify:
   - EutherLink `/health`
   - EutherLink `/v1/resources`
   - EutherBooks `/health`
   - EutherBooks `/voices`
   - UI active job bookmark flow.

Avoid using sudo unless service management requires it.

## Validation Checklist

Local EutherLink:

- `python -m py_compile eutherlink.py`
- temporary instance on unused port
- `GET /health`
- `GET /v1/resources`
- `GET /v1/tts/jobs`
- Dots stop while idle
- Dots stop refusal while active
- VoxCPM unload while idle
- Ollama stop for `qwen3-coder:30b`

EutherBooks:

- run existing pytest suite
- verify `/voices` still exposes Dots backends with defaults
- verify TTS job creation still forwards `model_backend`
- verify bookmarked job is returned after reload

EutherOxide:

- run frontend build/checks
- verify webview shows ongoing job state
- verify no regression in existing EutherBooks navigation.

Live smoke:

- Start one short Dots job.
- Refresh UI and resume progress view.
- Cancel job and confirm worker stop policy behaves correctly.
- Start `qwen3-coder:30b` after idle TTS resources are released.
- Stop the Ollama model from EutherLink resource endpoint.

## Open Questions

- Should EutherLink own generic LLM chat, or should CLI talk directly to Ollama
  through SSH tunnel for now?
- Should idle Dots prewarm remain enabled by default after LLM integration?
- Should the UI auto-stop idle models, or only present explicit buttons?
- Where should combined non-TTS job history live: EutherLink only, or mirrored
  into EutherBooks for book-related jobs?
