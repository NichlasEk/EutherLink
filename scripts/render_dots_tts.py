#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_DOTS_ROOT = Path("/home/nichlas/ai/dots_tts/dots.tts")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render dots.tts chunks for EutherLink.")
    parser.add_argument("request_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--dots-root",
        type=Path,
        default=DEFAULT_DOTS_ROOT,
        help="Path to the dots.tts source checkout.",
    )
    args = parser.parse_args()

    dots_root = args.dots_root.resolve()
    for import_root in (dots_root, dots_root / "src"):
        import_root_str = str(import_root)
        if import_root_str not in sys.path:
            sys.path.insert(0, import_root_str)

    from apps.gradio.service import (  # noqa: PLC0415
        GradioAppService,
        SynthesisRequest,
        build_gradio_app_config,
    )

    payload = json.loads(args.request_json.read_text(encoding="utf-8"))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = str(payload["model_path"])
    chunks = [str(chunk) for chunk in payload["chunks"]]
    prompt_audio_path = str(payload["prompt_audio_path"])
    prompt_text = str(payload["prompt_text"])
    seed = int(payload.get("seed", 42))

    config = build_gradio_app_config(
        model_name_or_path=model_path,
        output_dir=output_dir / "artifacts",
        execution_mode=str(payload.get("execution_mode", "generate")),
        precision=str(payload.get("precision", "bfloat16")),
        optimize=bool(payload.get("optimize", False)),
        max_generate_length=int(payload.get("max_generate_length", 500)),
        repo_root=dots_root,
    )
    service = GradioAppService(config)

    rendered_chunks: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        request = SynthesisRequest(
            model_name_or_path=model_path,
            text=chunk,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            execution_mode=config.execution_mode,
            template_name=str(payload.get("template_name", "tts")),
            language=payload.get("language") or None,
            ode_method=str(payload.get("ode_method", "euler")),
            num_steps=int(payload.get("num_steps", 10)),
            guidance_scale=float(payload.get("guidance_scale", 1.2)),
            speaker_scale=float(payload.get("speaker_scale", 1.5)),
            normalize_text=bool(payload.get("normalize_text", False)),
            seed=seed,
        )
        result = service.generate(request)
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
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
