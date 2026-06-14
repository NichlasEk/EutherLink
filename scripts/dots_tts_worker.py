#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import soundfile as sf


DEFAULT_DOTS_ROOT = Path("/home/nichlas/ai/dots_tts/dots.tts")
DEFAULT_OUTPUT_DIR = Path("/home/nichlas/EutherLink/data/dots-worker-artifacts")
DEFAULT_MODEL_PATH = Path("/home/nichlas/ai/dots_tts/models/dots.tts-soar")
DEFAULT_MAX_GENERATE_LENGTH = 500


class RenderRequest(BaseModel):
    request_json: str
    output_dir: str
    progress_json: str | None = None


class DotsWorker:
    def __init__(self, dots_root: Path, output_dir: Path, max_generate_length: int) -> None:
        self.dots_root = dots_root.resolve()
        self.output_dir = output_dir
        self.max_generate_length = max_generate_length
        self.lock = threading.Lock()
        self.service = self._build_service()

    def _build_service(self) -> Any:
        for import_root in (self.dots_root, self.dots_root / "src"):
            import_root_str = str(import_root)
            if import_root_str not in sys.path:
                sys.path.insert(0, import_root_str)

        from apps.gradio.service import GradioAppService, build_gradio_app_config

        config = build_gradio_app_config(
            model_name_or_path=str(DEFAULT_MODEL_PATH),
            output_dir=self.output_dir,
            execution_mode="generate_stream",
            precision="bfloat16",
            optimize=False,
            max_generate_length=self.max_generate_length,
            repo_root=self.dots_root,
        )
        return GradioAppService(config)

    def preload(self) -> dict[str, Any]:
        with self.lock:
            metrics = self.service.warmup()
            resolved_model_path = str(metrics.get("resolved_model_name_or_path") or DEFAULT_MODEL_PATH)
        return {
            "ok": True,
            "model_loaded": True,
            "loaded_model": resolved_model_path,
            "warmup": metrics,
        }

    def render(self, request_json: Path, output_dir: Path, progress_json: Path | None = None) -> dict[str, Any]:
        from apps.gradio.service import SynthesisRequest

        payload = json.loads(request_json.read_text(encoding="utf-8"))
        output_dir.mkdir(parents=True, exist_ok=True)
        if progress_json is not None:
            progress_json.parent.mkdir(parents=True, exist_ok=True)

        model_path = str(payload["model_path"])
        chunks = [str(chunk) for chunk in payload["chunks"]]
        prompt_audio_path = str(payload["prompt_audio_path"])
        prompt_text = str(payload["prompt_text"])
        seed = int(payload.get("seed", 42))

        rendered_chunks: list[dict[str, Any]] = []
        with self.lock:
            for index, chunk in enumerate(chunks, start=1):
                self._write_progress(progress_json, index, len(chunks), "running")
                progress_state: dict[str, int] = {}
                request = SynthesisRequest(
                    model_name_or_path=model_path,
                    text=chunk,
                    prompt_audio_path=prompt_audio_path,
                    prompt_text=prompt_text,
                    execution_mode=str(payload.get("execution_mode", "generate_stream")),
                    template_name=str(payload.get("template_name", "tts")),
                    language=payload.get("language") or None,
                    ode_method=str(payload.get("ode_method", "euler")),
                    num_steps=int(payload.get("num_steps", 8)),
                    guidance_scale=float(payload.get("guidance_scale", 1.2)),
                    speaker_scale=float(payload.get("speaker_scale", 1.5)),
                    normalize_text=bool(payload.get("normalize_text", False)),
                    seed=seed,
                )
                sink_id = self._add_progress_log_sink(progress_json, index, len(chunks), progress_state)
                try:
                    result = self._generate_streaming_partials(
                        request,
                        output_dir,
                        progress_json,
                        index,
                        len(chunks),
                        progress_state,
                    )
                finally:
                    if sink_id is not None:
                        from loguru import logger as loguru_logger

                        loguru_logger.remove(sink_id)
                self._write_progress(progress_json, index, len(chunks), "done", progress_state.get("patch_total"), progress_state.get("patch_total"))
                rendered_chunks.append(
                    {
                        "index": index,
                        "audio_path": result.audio_path,
                        "metrics": result.metrics,
                        "status": result.status,
                    }
                )

        manifest = {
            "model_path": model_path,
            "chunk_count": len(rendered_chunks),
            "seed": seed,
            "chunks": rendered_chunks,
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {"ok": True, "manifest_path": str(manifest_path)}

    def _generate_streaming_partials(
        self,
        request: Any,
        output_dir: Path,
        progress_json: Path | None,
        chunk_index: int,
        chunk_total: int,
        progress_state: dict[str, int],
    ) -> Any:
        from apps.gradio.service import SynthesisResult
        from lightning import seed_everything

        normalized_request = self.service._normalize_request(request)  # noqa: SLF001
        seed_everything(normalized_request.seed)
        runtime, resolved_model = self.service._get_runtime(normalized_request.model_name_or_path)  # noqa: SLF001

        partial_dir = output_dir / "partials"
        partial_dir.mkdir(parents=True, exist_ok=True)
        stream_chunks: list[torch.Tensor] = []
        full_chunks: list[torch.Tensor] = []
        partial_paths: list[str] = []
        sample_rate = runtime.sample_rate
        stream_group_size = 4

        for stream_index, tensor in enumerate(
            runtime.generate_stream(**self.service._build_runtime_generate_kwargs(normalized_request)),  # noqa: SLF001
            start=1,
        ):
            cpu_tensor = tensor.detach().float().cpu()
            stream_chunks.append(cpu_tensor)
            full_chunks.append(cpu_tensor)
            if len(stream_chunks) >= stream_group_size:
                partial_paths.append(
                    self._write_partial_audio(partial_dir, chunk_index, len(partial_paths) + 1, stream_chunks, sample_rate)
                )
                progress_state["partials"] = partial_paths
                stream_chunks = []
                self._write_progress(
                    progress_json,
                    chunk_index,
                    chunk_total,
                    "running",
                    progress_state.get("patch_count"),
                    progress_state.get("patch_total"),
                    partial_paths,
                )

        if stream_chunks:
            partial_paths.append(
                self._write_partial_audio(partial_dir, chunk_index, len(partial_paths) + 1, stream_chunks, sample_rate)
            )
            progress_state["partials"] = partial_paths
            self._write_progress(
                progress_json,
                chunk_index,
                chunk_total,
                "running",
                progress_state.get("patch_count"),
                progress_state.get("patch_total"),
                partial_paths,
            )
        if not full_chunks:
            raise ValueError("dots.tts stream did not return audio")

        audio = torch.cat(full_chunks, dim=-1)
        artifact_paths = self.service._write_generation_artifacts(  # noqa: SLF001
            request_id=self.service._build_stream_request_id(runtime, normalized_request),  # noqa: SLF001
            audio=audio,
            sample_rate=sample_rate,
            request=normalized_request,
            resolved_model_name_or_path=str(resolved_model),
        )
        return SynthesisResult(
            audio_path=str(artifact_paths["output_path"]),
            metrics={
                "sample_rate": sample_rate,
                "stream_partial_count": len(partial_paths),
            },
            status="ok",
        )

    @staticmethod
    def _write_partial_audio(partial_dir: Path, chunk_index: int, partial_index: int, chunks: list[torch.Tensor], sample_rate: int) -> str:
        audio = torch.cat(chunks, dim=-1).squeeze().numpy()
        path = partial_dir / f"chunk-{chunk_index:03d}-part-{partial_index:03d}.wav"
        temp_path = path.with_name(f".{path.name}.tmp")
        sf.write(temp_path, audio, sample_rate, subtype="PCM_16")
        temp_path.replace(path)
        return path.name

    def _add_progress_log_sink(
        self,
        progress_json: Path | None,
        chunk_index: int,
        chunk_total: int,
        progress_state: dict[str, int],
    ) -> int | None:
        if progress_json is None:
            return None

        from loguru import logger as loguru_logger

        def capture(message: object) -> None:
            text = str(message)
            prepared = re.search(r"prompt_audio_patch_count=(\d+) max_audio_patch_count=(\d+)", text)
            if prepared:
                prompt_patches = int(prepared.group(1))
                max_patches = int(prepared.group(2))
                progress_state["patch_total"] = max(1, max_patches - prompt_patches)
                self._write_progress(
                    progress_json,
                    chunk_index,
                    chunk_total,
                    "running",
                    0,
                    progress_state["patch_total"],
                )
                return

            progress = re.search(r"payload_audio_patches=(\d+)", text)
            if progress:
                patch_count = int(progress.group(1))
                progress_state["patch_count"] = patch_count
                patch_total = progress_state.get("patch_total") or max(1, self.max_generate_length)
                self._write_progress(
                    progress_json,
                    chunk_index,
                    chunk_total,
                    "running",
                    patch_count,
                    patch_total,
                    progress_state.get("partials"),
                )

        return loguru_logger.add(capture, level="INFO")

    @staticmethod
    def _write_progress(
        progress_json: Path | None,
        chunk_index: int,
        chunk_total: int,
        status: str,
        patch_count: int | None = None,
        patch_total: int | None = None,
        partials: list[str] | None = None,
    ) -> None:
        if progress_json is None:
            return
        payload = {
            "chunk_index": chunk_index,
            "chunk_total": chunk_total,
            "status": status,
        }
        if patch_count is not None:
            payload["patch_count"] = patch_count
        if patch_total is not None:
            payload["patch_total"] = patch_total
        if partials is not None:
            payload["partials"] = partials
        temp_path = progress_json.with_name(f".{progress_json.name}.{threading.get_ident()}.tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        temp_path.replace(progress_json)


def create_app(worker: DotsWorker) -> FastAPI:
    app = FastAPI(title="EutherLink Dots TTS Worker")

    @app.get("/health")
    def health() -> dict[str, object]:
        metadata = worker.service.metadata()
        return {
            "ok": True,
            "model_loaded": metadata.get("model_loaded"),
            "loaded_model": metadata.get("loaded_model_name_or_path"),
            "precision": metadata.get("configured_precision"),
            "max_generate_length": metadata.get("configured_max_generate_length"),
        }

    @app.post("/preload")
    def preload() -> dict[str, Any]:
        return worker.preload()

    @app.post("/render")
    def render(request: RenderRequest) -> dict[str, Any]:
        progress_json = Path(request.progress_json) if request.progress_json else None
        return worker.render(Path(request.request_json), Path(request.output_dir), progress_json)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent dots.tts worker for EutherLink.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--dots-root", type=Path, default=DEFAULT_DOTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-generate-length", type=int, default=DEFAULT_MAX_GENERATE_LENGTH)
    args = parser.parse_args()

    worker = DotsWorker(args.dots_root, args.output_dir, args.max_generate_length)
    uvicorn.run(create_app(worker), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
