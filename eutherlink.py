#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import binascii
import concurrent.futures
import dataclasses
import hashlib
import json
import logging
import os
import random
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from voxcpm import VoxCPM


LOGGER = logging.getLogger("eutherlink")

DEFAULT_MODEL_PATH = (
    "/home/nichlas/.cache/huggingface/hub/models--openbmb--VoxCPM2/"
    "snapshots/e8b928065859f2869644c1e2881cbd21f888c659"
)
DEFAULT_DATA_DIR = "/home/nichlas/EutherLink/data"
DOTS_TTS_PYTHON = "/home/nichlas/ai/dots_tts/dots.tts/.venv/bin/python"
DOTS_TTS_RENDERER = "/home/nichlas/EutherLink/scripts/render_dots_tts.py"
DOTS_TTS_WORKER = "/home/nichlas/EutherLink/scripts/dots_tts_worker.py"
DOTS_TTS_SOAR_PATH = "/home/nichlas/ai/dots_tts/models/dots.tts-soar"
DOTS_TTS_WORKER_URL = "http://127.0.0.1:18765"
DOTS_TTS_MAX_WORDS = 120
DOTS_TTS_MIN_WORDS = 40
DOTS_TTS_MODEL_MAX_WORDS = 180
DOTS_TTS_DEFAULT_GENERATE_LENGTH = 500
DOTS_TTS_DEFAULT_STEPS = 4
DOTS_TTS_SAMPLE_RATE = 48_000
PREWARM_DOTS_DEFAULT = "1"

OutputFormat = Literal["wav", "mp3", "opus"]
ModelBackend = Literal["voxcpm2", "dots.tts-soar"]
DotsTemplateName = Literal["tts", "instruction_tts", "text_to_audio", "tts_interleave"]
DotsOdeMethod = Literal["euler", "midpoint"]


class TtsJobRequest(BaseModel):
    text: str = Field(min_length=1, max_length=250_000)
    voice_instruction: str = Field(
        default="A calm, warm Swedish audiobook narrator with clear pronunciation and natural pacing.",
        max_length=500,
    )
    language: str = Field(default="sv", max_length=16)
    output_format: OutputFormat = "opus"
    model_backend: ModelBackend = "voxcpm2"
    cfg_value: float = Field(default=2.0, ge=1.0, le=3.0)
    inference_timesteps: int = Field(default=10, ge=1, le=50)
    normalize: bool = False
    max_chunk_chars: int = Field(default=700, ge=120, le=1500)
    dots_template_name: DotsTemplateName = "tts"
    dots_ode_method: DotsOdeMethod = "euler"
    dots_num_steps: int = Field(default=DOTS_TTS_DEFAULT_STEPS, ge=1, le=50)
    dots_guidance_scale: float = Field(default=1.2, ge=0.0, le=5.0)
    dots_speaker_scale: float = Field(default=1.5, ge=0.0, le=5.0)
    dots_max_generate_length: int = Field(default=DOTS_TTS_DEFAULT_GENERATE_LENGTH, ge=128, le=4096)
    prompt_wav_base64: str | None = Field(default=None, max_length=32_000_000)
    reference_wav_base64: str | None = Field(default=None, max_length=32_000_000)
    prompt_text: str | None = Field(default=None, max_length=500)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)


class TtsJobAccepted(BaseModel):
    id: str
    status: str
    status_url: str
    audio_url: str


class TtsJobStatus(BaseModel):
    id: str
    status: str
    progress: float
    message: str
    output_format: OutputFormat
    created_at: float
    updated_at: float
    error: str | None = None
    audio_url: str | None = None
    partial_audio_urls: list[str] = Field(default_factory=list)


@dataclasses.dataclass
class RuntimeConfig:
    model_path: str
    data_dir: Path
    host: str
    port: int


@dataclasses.dataclass
class JobState:
    id: str
    request: TtsJobRequest
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    error: str | None = None
    partial_audio_files: list[str] = dataclasses.field(default_factory=list)
    created_at: float = dataclasses.field(default_factory=time.time)
    updated_at: float = dataclasses.field(default_factory=time.time)


class EutherLinkTts:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.jobs: dict[str, JobState] = {}
        self.jobs_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.model: VoxCPM | None = None
        self.dots_worker_lock = threading.Lock()
        self.dots_worker_process: subprocess.Popen[bytes] | None = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self.jobs_dir = config.data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self) -> VoxCPM:
        if self.model is None:
            with self.model_lock:
                if self.model is None:
                    self.model = VoxCPM.from_pretrained(
                        self.config.model_path,
                        load_denoiser=False,
                        local_files_only=True,
                        optimize=True,
                        device="cuda",
                    )
        return self.model

    def ensure_dots_worker(self) -> None:
        with self.dots_worker_lock:
            if self._dots_worker_healthy(timeout=0.5):
                return
            if self.dots_worker_process is not None and self.dots_worker_process.poll() is None:
                self.dots_worker_process.terminate()
                try:
                    self.dots_worker_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.dots_worker_process.kill()
                    self.dots_worker_process.wait(timeout=5)

            worker_env = os.environ.copy()
            worker_env.setdefault("NUMBA_CACHE_DIR", str(self.config.data_dir / "numba-cache"))
            dots_generate_length = int(os.environ.get("EUTHERLINK_DOTS_TTS_MAX_GENERATE_LENGTH", DOTS_TTS_DEFAULT_GENERATE_LENGTH))
            self.dots_worker_process = subprocess.Popen(
                [
                    os.environ.get("EUTHERLINK_DOTS_TTS_PYTHON", DOTS_TTS_PYTHON),
                    os.environ.get("EUTHERLINK_DOTS_TTS_WORKER", DOTS_TTS_WORKER),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "18765",
                    "--output-dir",
                    str(self.config.data_dir / "dots-worker-artifacts"),
                    "--max-generate-length",
                    str(dots_generate_length),
                ],
                cwd=str(Path(DOTS_TTS_WORKER).resolve().parent),
                env=worker_env,
            )

            deadline = time.time() + 90
            while time.time() < deadline:
                if self._dots_worker_healthy(timeout=2.0):
                    return
                if self.dots_worker_process.poll() is not None:
                    raise RuntimeError(f"dots.tts worker exited with code {self.dots_worker_process.returncode}")
                time.sleep(1)
            raise TimeoutError("dots.tts worker did not become healthy")

    def _dots_worker_healthy(self, timeout: float) -> bool:
        try:
            with urllib.request.urlopen(f"{DOTS_TTS_WORKER_URL}/health", timeout=timeout) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def dots_worker_status(self) -> dict[str, object]:
        try:
            with urllib.request.urlopen(f"{DOTS_TTS_WORKER_URL}/health", timeout=0.5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            process_running = self.dots_worker_process is not None and self.dots_worker_process.poll() is None
            return {
                "ok": False,
                "status": "starting" if process_running else "offline",
                "model_loaded": False,
            }

        model_loaded = bool(payload.get("model_loaded"))
        return {
            "ok": True,
            "status": "ready" if model_loaded else "warming",
            "model_loaded": model_loaded,
            "loaded_model": payload.get("loaded_model"),
            "precision": payload.get("precision"),
            "max_generate_length": payload.get("max_generate_length"),
        }

    def prewarm_dots_worker(self) -> None:
        try:
            self.ensure_dots_worker()
            request = urllib.request.Request(
                f"{DOTS_TTS_WORKER_URL}/preload",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=None) as response:
                if response.status != 200:
                    raise RuntimeError(f"dots.tts preload returned HTTP {response.status}")
            LOGGER.warning("TTS_TRACE dots_prewarm_done")
        except Exception:
            LOGGER.exception("TTS_TRACE dots_prewarm_failed")

    def render_dots_with_worker(self, job_id: str, request_path: Path, dots_dir: Path) -> None:
        self.ensure_dots_worker()
        progress_path = dots_dir / "progress.json"
        payload = json.dumps(
            {
                "request_json": str(request_path),
                "output_dir": str(dots_dir),
                "progress_json": str(progress_path),
            }
        ).encode("utf-8")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._post_dots_render, payload)
            last_progress_signature: tuple[int, int, str, int, int, int] | None = None
            while True:
                try:
                    response_status = future.result(timeout=0.5)
                    if response_status != 200:
                        raise RuntimeError(f"dots.tts worker returned HTTP {response_status}")
                    self._sync_dots_progress(job_id, progress_path, force=True)
                    return
                except concurrent.futures.TimeoutError:
                    signature = self._sync_dots_progress(job_id, progress_path, last_progress_signature)
                    if signature is not None:
                        last_progress_signature = signature

    def _post_dots_render(self, payload: bytes) -> int:
        request = urllib.request.Request(
            f"{DOTS_TTS_WORKER_URL}/render",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=None) as response:
            return int(response.status)

    def _sync_dots_progress(
        self,
        job_id: str,
        progress_path: Path,
        last_signature: tuple[int, int, str, int, int, int] | None = None,
        *,
        force: bool = False,
    ) -> tuple[int, int, str, int, int, int] | None:
        if not progress_path.exists():
            return None
        try:
            data = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        index = int(data.get("chunk_index") or 0)
        total = max(1, int(data.get("chunk_total") or 1))
        status = str(data.get("status") or "")
        patch_count = max(0, int(data.get("patch_count") or 0))
        patch_total = max(0, int(data.get("patch_total") or 0))
        partials = [
            name
            for name in (data.get("partials") or [])
            if isinstance(name, str) and "/" not in name and "\\" not in name and name.endswith(".wav")
        ]
        signature = (index, total, status, patch_count, patch_total, len(partials))
        if not force and signature == last_signature:
            return signature
        chunk_progress = 0.0
        if status == "done":
            chunk_progress = 1.0
        elif status == "running":
            chunk_progress = min(0.98, max(0.0, patch_count / patch_total)) if patch_total > 0 else 0.05
        elif status == "queued":
            chunk_progress = 0.05
        completed = max(0, index - 1) + chunk_progress
        progress = 0.05 + 0.9 * min(1.0, completed / total)
        detail = ""
        if status == "running" and patch_total > 0:
            detail = f" {patch_count}/{patch_total} patches"
        self._set_state(
            job_id,
            status="running",
            progress=progress,
            message=f"dots.tts-soar chunk {min(index, total)}/{total} {status}{detail}".strip(),
            partial_audio_files=partials,
        )
        return signature

    def submit(self, request: TtsJobRequest) -> JobState:
        job_id = uuid.uuid4().hex
        state = JobState(id=job_id, request=request)
        with self.jobs_lock:
            self.jobs[job_id] = state
        LOGGER.warning(
            "TTS_TRACE submit job=%s backend=%s lang=%s fmt=%s text_len=%s text_sha=%s seed_request=%s cfg=%.3f steps=%s dots_guidance=%.3f dots_speaker=%.3f dots_steps=%s dots_max_len=%s max_chunk_chars=%s has_prompt=%s has_reference=%s voice_instruction_sha=%s",
            job_id,
            request.model_backend,
            request.language,
            request.output_format,
            len(request.text),
            short_sha256(request.text.encode("utf-8")),
            request.seed,
            request.cfg_value,
            request.inference_timesteps,
            request.dots_guidance_scale,
            request.dots_speaker_scale,
            request.dots_num_steps,
            request.dots_max_generate_length,
            request.max_chunk_chars,
            bool(request.prompt_wav_base64),
            bool(request.reference_wav_base64),
            short_sha256(request.voice_instruction.encode("utf-8")),
        )
        self.executor.submit(self._run_job, job_id)
        return state

    def get(self, job_id: str) -> JobState:
        with self.jobs_lock:
            state = self.jobs.get(job_id)
        if state is None:
            state_path = self.jobs_dir / job_id / "status.json"
            if state_path.exists():
                data = json.loads(state_path.read_text(encoding="utf-8"))
                request = TtsJobRequest(**data["request"])
                state = JobState(id=job_id, request=request)
                state.status = data["status"]
                state.progress = data["progress"]
                state.message = data["message"]
                state.error = data.get("error")
                state.partial_audio_files = list(data.get("partial_audio_files", []))
                state.created_at = data["created_at"]
                state.updated_at = data["updated_at"]
            else:
                raise KeyError(job_id)
        return state

    def audio_path(self, job_id: str) -> Path:
        state = self.get(job_id)
        return self.jobs_dir / job_id / f"audio.{state.request.output_format}"

    def partial_audio_path(self, job_id: str, filename: str) -> Path:
        if "/" in filename or "\\" in filename or not filename.endswith(".wav"):
            raise ValueError("invalid partial filename")
        path = self.jobs_dir / job_id / "dots.tts-soar" / "partials" / filename
        if not path.exists():
            raise FileNotFoundError(filename)
        return path

    def _set_state(self, job_id: str, **updates: object) -> JobState:
        with self.jobs_lock:
            state = self.jobs[job_id]
            for key, value in updates.items():
                setattr(state, key, value)
            state.updated_at = time.time()
            self._write_status(state)
            return state

    def _write_status(self, state: JobState) -> None:
        job_dir = self.jobs_dir / state.id
        job_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": state.id,
            "request": state.request.model_dump(),
            "status": state.status,
            "progress": state.progress,
            "message": state.message,
            "error": state.error,
            "partial_audio_files": state.partial_audio_files,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
        }
        (job_dir / "status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _run_job(self, job_id: str) -> None:
        try:
            state = self.get(job_id)
            req = state.request
            self._set_state(job_id, status="loading", progress=0.01, message="Preparing TTS job")

            chunks = (
                split_dots_text(req.text, req.max_chunk_chars, DOTS_TTS_MAX_WORDS)
                if req.model_backend == "dots.tts-soar"
                else split_text(req.text, req.max_chunk_chars)
            )
            if not chunks:
                raise ValueError("No text to synthesize")

            voice_sample_path = write_voice_sample(job_id, self.jobs_dir, req)
            if req.model_backend == "dots.tts-soar":
                self._run_dots_tts_job(job_id, req, chunks, voice_sample_path)
                return

            self._set_state(job_id, status="loading", progress=0.01, message="Loading VoxCPM2")
            model = self.load_model()

            wav_parts: list[np.ndarray] = []
            sample_rate = int(model.tts_model.sample_rate)
            silence = np.zeros(int(sample_rate * 0.35), dtype=np.float32)
            cloned_cfg_value = min(req.cfg_value, float(os.environ.get("EUTHERLINK_CLONED_VOICE_MAX_CFG", "2.0")))
            job_seed = stable_voice_seed(voice_sample_path, req) if voice_sample_path is not None or req.seed is not None else None
            sample_size = voice_sample_path.stat().st_size if voice_sample_path is not None else 0
            sample_sha = file_short_sha256(voice_sample_path) if voice_sample_path is not None else ""
            LOGGER.warning(
                "TTS_TRACE run job=%s chunks=%s sample=%s sample_size=%s sample_sha=%s seed_request=%s seed_effective=%s seed_mode=%s cloned_cfg=%.3f prompt_text_len=%s prompt_text_sha=%s",
                job_id,
                len(chunks),
                voice_sample_path is not None,
                sample_size,
                sample_sha,
                req.seed,
                job_seed,
                "job_once" if job_seed is not None else "none",
                cloned_cfg_value,
                len(req.prompt_text or ""),
                short_sha256((req.prompt_text or "").encode("utf-8")),
            )
            prompt_cache: dict[str, Any] | None = None
            initial_prompt_cache: dict[str, Any] | None = None
            if job_seed is not None:
                set_generation_seed(job_seed)
            if voice_sample_path is not None:
                with self.model_lock:
                    prompt_cache = model.tts_model.build_prompt_cache(
                        prompt_text=req.prompt_text if req.prompt_text else None,
                        prompt_wav_path=str(voice_sample_path) if req.prompt_text else None,
                        reference_wav_path=str(voice_sample_path),
                    )
                initial_prompt_cache = prompt_cache

            for index, chunk in enumerate(chunks, start=1):
                LOGGER.warning(
                    "TTS_TRACE chunk_start job=%s chunk=%s/%s text_len=%s text_sha=%s seed_effective=%s seed_mode=%s sample_sha=%s prompt_cache=%s",
                    job_id,
                    index,
                    len(chunks),
                    len(chunk),
                    short_sha256(chunk.encode("utf-8")),
                    job_seed,
                    "job_once" if job_seed is not None else "none",
                    sample_sha,
                    prompt_cache is not None,
                )
                progress = 0.05 + 0.85 * ((index - 1) / max(1, len(chunks)))
                self._set_state(
                    job_id,
                    status="running",
                    progress=progress,
                    message=f"Synthesizing chunk {index}/{len(chunks)}",
                )
                if voice_sample_path is not None:
                    with self.model_lock:
                        wav_tensor, _text_tokens, new_audio_feat = model.tts_model.generate_with_prompt_cache(
                            target_text=chunk,
                            prompt_cache=prompt_cache,
                            max_len=4096,
                            cfg_value=cloned_cfg_value,
                            inference_timesteps=req.inference_timesteps,
                            retry_badcase=True,
                        )
                        if initial_prompt_cache is not None:
                            prompt_cache = model.tts_model.merge_prompt_cache(
                                initial_prompt_cache,
                                f" {chunk}",
                                new_audio_feat,
                            )
                    wav = wav_tensor.squeeze(0).cpu().numpy()
                else:
                    final_text = f"({req.voice_instruction}){chunk}" if req.voice_instruction.strip() else chunk
                    generate_kwargs: dict[str, object] = {
                        "text": final_text,
                        "cfg_value": req.cfg_value,
                        "inference_timesteps": req.inference_timesteps,
                        "normalize": req.normalize,
                        "denoise": False,
                    }
                    with self.model_lock:
                        wav = model.generate(**generate_kwargs)
                LOGGER.warning(
                    "TTS_TRACE chunk_done job=%s chunk=%s/%s samples=%s peak=%.5f",
                    job_id,
                    index,
                    len(chunks),
                    len(wav),
                    float(np.max(np.abs(wav))) if len(wav) else 0.0,
                )
                wav_parts.append(np.asarray(wav, dtype=np.float32))
                if index != len(chunks):
                    wav_parts.append(silence)

            audio = np.concatenate(wav_parts) if wav_parts else np.zeros(0, dtype=np.float32)
            audio = normalize_peak(audio)

            job_dir = self.jobs_dir / job_id
            wav_path = job_dir / "audio.wav"
            sf.write(wav_path, audio, sample_rate, subtype="PCM_16")

            output_path = job_dir / f"audio.{req.output_format}"
            if req.output_format == "wav":
                output_path = wav_path
            else:
                encode_audio(wav_path, output_path, req.output_format)

            self._set_state(
                job_id,
                status="done",
                progress=1.0,
                message=f"Done: {output_path.name}",
            )
            LOGGER.warning("TTS_TRACE done job=%s output=%s", job_id, output_path)
        except Exception as exc:
            LOGGER.exception("TTS_TRACE failed job=%s", job_id)
            self._set_state(
                job_id,
                status="failed",
                progress=1.0,
                message="Failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _run_dots_tts_job(
        self,
        job_id: str,
        req: TtsJobRequest,
        chunks: list[str],
        voice_sample_path: Path | None,
    ) -> None:
        if voice_sample_path is None:
            raise ValueError("dots.tts-soar requires prompt_wav_base64 or reference_wav_base64")
        if not (req.prompt_text or "").strip():
            raise ValueError("dots.tts-soar requires prompt_text matching the prompt audio")

        job_dir = self.jobs_dir / job_id
        dots_dir = job_dir / "dots.tts-soar"
        request_path = dots_dir / "request.json"
        dots_dir.mkdir(parents=True, exist_ok=True)
        seed = stable_voice_seed(voice_sample_path, req)
        model_path = os.environ.get("EUTHERLINK_DOTS_TTS_SOAR_PATH", DOTS_TTS_SOAR_PATH)
        language = dots_language(req.language)
        max_generate_length = req.dots_max_generate_length
        prompt_audio_path = prepare_dots_prompt_audio(voice_sample_path, job_dir)
        payload = {
            "model_path": model_path,
            "chunks": chunks,
            "prompt_audio_path": str(prompt_audio_path),
            "prompt_text": req.prompt_text,
            "language": language,
            "seed": seed,
            "template_name": req.dots_template_name,
            "ode_method": req.dots_ode_method,
            "num_steps": req.dots_num_steps,
            "guidance_scale": req.dots_guidance_scale,
            "speaker_scale": req.dots_speaker_scale,
            "max_generate_length": max_generate_length,
            "execution_mode": "generate_stream",
            "normalize_text": req.normalize,
        }
        request_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.warning(
            "TTS_TRACE dots_start job=%s chunks=%s max_words=%s seed_effective=%s model=%s language=%s template=%s ode=%s steps=%s guidance=%.3f speaker=%.3f max_len=%s prompt_sha=%s",
            job_id,
            len(chunks),
            max((word_count(chunk) for chunk in chunks), default=0),
            seed,
            model_path,
            language,
            req.dots_template_name,
            req.dots_ode_method,
            req.dots_num_steps,
            req.dots_guidance_scale,
            req.dots_speaker_scale,
            max_generate_length,
            file_short_sha256(prompt_audio_path),
        )
        self._set_state(
            job_id,
            status="running",
            progress=0.05,
            message=f"Synthesizing {len(chunks)} chunk(s) with dots.tts-soar",
        )
        self.render_dots_with_worker(job_id, request_path, dots_dir)

        manifest_path = dots_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        wav_parts: list[np.ndarray] = []
        sample_rate: int | None = None
        silence = np.zeros(0, dtype=np.float32)
        for index, rendered in enumerate(manifest["chunks"], start=1):
            wav, rate = sf.read(rendered["audio_path"], dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sample_rate is None:
                sample_rate = int(rate)
                silence = np.zeros(int(sample_rate * 0.35), dtype=np.float32)
            elif sample_rate != int(rate):
                raise ValueError(f"dots.tts chunk sample-rate mismatch: {sample_rate} != {rate}")
            wav_parts.append(np.asarray(wav, dtype=np.float32))
            if index != len(manifest["chunks"]):
                wav_parts.append(silence)

        if sample_rate is None:
            raise ValueError("dots.tts did not render any audio")
        audio = normalize_peak(np.concatenate(wav_parts))
        wav_path = job_dir / "audio.wav"
        sf.write(wav_path, audio, sample_rate, subtype="PCM_16")
        output_path = job_dir / f"audio.{req.output_format}"
        if req.output_format == "wav":
            output_path = wav_path
        else:
            encode_audio(wav_path, output_path, req.output_format)
        self._set_state(
            job_id,
            status="done",
            progress=1.0,
            message=f"Done: {output_path.name}",
        )
        LOGGER.warning("TTS_TRACE dots_done job=%s output=%s", job_id, output_path)


def write_voice_sample(job_id: str, jobs_dir: Path, req: TtsJobRequest) -> Path | None:
    sample_base64 = req.prompt_wav_base64 or req.reference_wav_base64
    if not sample_base64:
        return None
    try:
        sample = base64.b64decode(sample_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid voice sample audio") from exc
    if not sample or len(sample) > 24 * 1024 * 1024:
        raise ValueError("Voice sample audio is empty or too large")
    if sample[:4] != b"RIFF" or sample[8:12] != b"WAVE":
        raise ValueError("Voice sample must be WAV audio")
    sample_path = jobs_dir / job_id / "voice-sample.wav"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_bytes(sample)
    return sample_path


def prepare_dots_prompt_audio(sample_path: Path, job_dir: Path) -> Path:
    dots_sample_path = job_dir / "dots-prompt-48k.wav"
    encode_args = [
        "ffmpeg",
        "-y",
        "-i",
        str(sample_path),
        "-ac",
        "1",
        "-ar",
        str(DOTS_TTS_SAMPLE_RATE),
        "-sample_fmt",
        "s16",
        str(dots_sample_path),
    ]
    subprocess.run(encode_args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return dots_sample_path


def short_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def file_short_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def stable_voice_seed(voice_sample_path: Path | None, req: TtsJobRequest) -> int:
    if req.seed is not None:
        return req.seed
    configured = os.environ.get("EUTHERLINK_CLONED_VOICE_SEED", "").strip()
    if configured:
        try:
            return int(configured) & 0x7FFF_FFFF
        except ValueError:
            pass
    if voice_sample_path is None:
        return 0
    digest = hashlib.blake2s(voice_sample_path.read_bytes(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFF_FFFF


def set_generation_seed(seed: int) -> None:
    normalized = seed & 0x7FFF_FFFF
    random.seed(normalized)
    np.random.seed(normalized)
    torch.manual_seed(normalized)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized)


def split_text(text: str, max_chars: int) -> list[str]:
    clean = re.sub(r"\s+", " ", text.strip())
    if not clean:
        return []

    sentences = re.split(r"(?<=[.!?。！？])\s+", clean)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            for piece in split_long_sentence(sentence, max_chars):
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.append(piece.strip())
            continue

        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()

    if current:
        chunks.append(current.strip())
    return chunks


def split_dots_text(text: str, max_chars: int, max_words: int) -> list[str]:
    chunks: list[str] = []
    for chunk in split_text(text, max_chars):
        if word_count(chunk) <= max_words:
            chunks.append(chunk)
            continue
        chunks.extend(split_text_by_words(chunk, max_words, DOTS_TTS_MIN_WORDS))
    return chunks


def split_text_by_words(text: str, max_words: int, min_words: int = 0) -> list[str]:
    words = re.findall(r"\S+", text.strip())
    if not words:
        return []
    chunks = [words[index : index + max_words] for index in range(0, len(words), max_words)]
    if len(chunks) > 1 and min_words > 0 and len(chunks[-1]) < min_words:
        tail = chunks.pop()
        if len(chunks[-1]) + len(tail) <= min(DOTS_TTS_MODEL_MAX_WORDS, max_words + min_words):
            chunks[-1].extend(tail)
        else:
            split_at = (len(chunks[-1]) + len(tail)) // 2
            combined = chunks[-1] + tail
            chunks[-1] = combined[:split_at]
            chunks.append(combined[split_at:])
    return [" ".join(chunk) for chunk in chunks]


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))


def dots_language(language: str) -> str | None:
    normalized = language.strip()
    if not normalized:
        return None
    if normalized.startswith("口音:"):
        return normalized
    return normalized.upper()


def split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    parts = re.split(r"(?<=[,;:])\s+", sentence)
    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(part) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(part[i : i + max_chars] for i in range(0, len(part), max_chars))
            continue
        if current and len(current) + 1 + len(part) > max_chars:
            chunks.append(current)
            current = part
        else:
            current = f"{current} {part}".strip()
    if current:
        chunks.append(current)
    return chunks


def normalize_peak(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak > 0.98:
        return audio * (0.98 / peak)
    return audio


def encode_audio(wav_path: Path, output_path: Path, output_format: OutputFormat) -> None:
    if output_format == "mp3":
        args = ["ffmpeg", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", "128k", str(output_path)]
    elif output_format == "opus":
        args = ["ffmpeg", "-y", "-i", str(wav_path), "-codec:a", "libopus", "-b:a", "64k", str(output_path)]
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def build_app(service: EutherLinkTts) -> FastAPI:
    app = FastAPI(title="EutherLink", version="0.1.0")

    @app.on_event("startup")
    def prewarm_models() -> None:
        prewarm_dots = os.environ.get("EUTHERLINK_PREWARM_DOTS", PREWARM_DOTS_DEFAULT).strip().lower()
        if prewarm_dots in {"1", "true", "yes", "on"}:
            threading.Thread(target=service.prewarm_dots_worker, name="dots-tts-prewarm", daemon=True).start()

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "ok": True,
            "service": "EutherLink",
            "model": service.config.model_path,
            "queued_or_running": sum(1 for job in service.jobs.values() if job.status in {"queued", "loading", "running"}),
            "dots_tts": service.dots_worker_status(),
        }

    @app.post("/v1/tts/jobs", response_model=TtsJobAccepted)
    def create_tts_job(request: TtsJobRequest) -> TtsJobAccepted:
        state = service.submit(request)
        return TtsJobAccepted(
            id=state.id,
            status=state.status,
            status_url=f"/v1/tts/jobs/{state.id}",
            audio_url=f"/v1/tts/jobs/{state.id}/audio",
        )

    @app.get("/v1/tts/jobs/{job_id}", response_model=TtsJobStatus)
    def get_tts_job(job_id: str) -> TtsJobStatus:
        try:
            state = service.get(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="job not found") from None
        return status_response(state)

    @app.get("/v1/tts/jobs/{job_id}/audio")
    def get_tts_audio(job_id: str) -> FileResponse:
        try:
            state = service.get(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="job not found") from None
        if state.status != "done":
            raise HTTPException(status_code=409, detail=f"job is {state.status}")
        path = service.audio_path(job_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="audio file not found")
        return FileResponse(path, filename=path.name)

    @app.get("/v1/tts/jobs/{job_id}/partials/{filename}")
    def get_tts_partial_audio(job_id: str, filename: str) -> FileResponse:
        try:
            service.get(job_id)
            path = service.partial_audio_path(job_id, filename)
        except KeyError:
            raise HTTPException(status_code=404, detail="job not found") from None
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid partial filename") from None
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="partial audio file not found") from None
        return FileResponse(path, filename=path.name)

    return app


def status_response(state: JobState) -> TtsJobStatus:
    audio_url = f"/v1/tts/jobs/{state.id}/audio" if state.status == "done" else None
    partial_audio_urls = [
        f"/v1/tts/jobs/{state.id}/partials/{filename}"
        for filename in state.partial_audio_files
        if "/" not in filename and "\\" not in filename
    ]
    return TtsJobStatus(
        id=state.id,
        status=state.status,
        progress=state.progress,
        message=state.message,
        output_format=state.request.output_format,
        created_at=state.created_at,
        updated_at=state.updated_at,
        error=state.error,
        audio_url=audio_url,
        partial_audio_urls=partial_audio_urls,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="EutherLink local service host")
    parser.add_argument("--host", default=os.environ.get("EUTHERLINK_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("EUTHERLINK_PORT", "8765")))
    parser.add_argument("--model-path", default=os.environ.get("EUTHERLINK_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--data-dir", default=os.environ.get("EUTHERLINK_DATA_DIR", DEFAULT_DATA_DIR))
    args = parser.parse_args()

    config = RuntimeConfig(
        model_path=args.model_path,
        data_dir=Path(args.data_dir),
        host=args.host,
        port=args.port,
    )
    service = EutherLinkTts(config)

    import uvicorn

    uvicorn.run(build_app(service), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
