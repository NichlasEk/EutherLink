#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


DEFAULT_DOTS_ROOT = Path("/home/nichlas/ai/dots_tts/dots.tts")
DEFAULT_OUTPUT_DIR = Path("/home/nichlas/EutherLink/data/dots-worker-artifacts")
DEFAULT_MODEL_PATH = Path("/home/nichlas/ai/dots_tts/models/dots.tts-soar")
DEFAULT_MAX_GENERATE_LENGTH = 128


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
            execution_mode="generate",
            precision="bfloat16",
            optimize=False,
            max_generate_length=self.max_generate_length,
            repo_root=self.dots_root,
        )
        return GradioAppService(config)

    def preload(self) -> dict[str, Any]:
        with self.lock:
            _, resolved_model_path = self.service._get_runtime(str(DEFAULT_MODEL_PATH))
        return {
            "ok": True,
            "model_loaded": True,
            "loaded_model": resolved_model_path,
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
                    execution_mode="generate",
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
                    result = self.service.generate(request)
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
                patch_total = progress_state.get("patch_total") or max(1, self.max_generate_length)
                self._write_progress(
                    progress_json,
                    chunk_index,
                    chunk_total,
                    "running",
                    patch_count,
                    patch_total,
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
