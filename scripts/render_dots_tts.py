#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import soundfile as sf


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
    prompt_audio_value = payload.get("prompt_audio_path")
    prompt_audio_path = str(prompt_audio_value).strip() if prompt_audio_value else None
    prompt_text_value = payload.get("prompt_text")
    prompt_text = str(prompt_text_value).strip() if prompt_text_value else None
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
    if prompt_audio_path is None:
        prompt_audio_path, prompt_text = generate_preset_prompt_audio(
            service=service,
            model_path=model_path,
            output_dir=output_dir,
            language=payload.get("language") or None,
            template_name=str(payload.get("template_name", "tts")),
            ode_method=str(payload.get("ode_method", "euler")),
            num_steps=int(payload.get("num_steps", 10)),
            guidance_scale=float(payload.get("guidance_scale", 1.2)),
            speaker_scale=float(payload.get("speaker_scale", 1.5)),
            normalize_text=bool(payload.get("normalize_text", False)),
            seed=seed,
        )

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


def generate_preset_prompt_audio(
    service: Any,
    model_path: str,
    output_dir: Path,
    language: str | None,
    template_name: str,
    ode_method: str,
    num_steps: int,
    guidance_scale: float,
    speaker_scale: float,
    normalize_text: bool,
    seed: int,
) -> tuple[str, str]:
    from apps.gradio.service import SynthesisRequest  # noqa: PLC0415
    from dots_tts.utils.util import seed_everything  # noqa: PLC0415

    prompt_text = preset_prompt_text(language)
    prompt_path = output_dir / "preset-prompt.wav"
    seed_everything(seed)
    runtime, _resolved_model = service._get_runtime(model_path)  # noqa: SLF001
    request = SynthesisRequest(
        model_name_or_path=model_path,
        text=prompt_text,
        prompt_audio_path=None,
        prompt_text=None,
        execution_mode="generate",
        template_name=template_name,
        language=language,
        ode_method=ode_method,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        speaker_scale=speaker_scale,
        normalize_text=normalize_text,
        seed=seed,
    )
    generation = runtime.generate(**service._build_runtime_generate_kwargs(request))  # noqa: SLF001
    sf.write(
        prompt_path,
        generation["audio"].float().cpu().squeeze().numpy(),
        int(generation["sample_rate"]),
        subtype="PCM_16",
    )
    return str(prompt_path), prompt_text


def preset_prompt_text(language: str | None) -> str:
    normalized = (language or "").strip().lower()
    if normalized.startswith("en"):
        return (
            "The morning light moves slowly across the room. "
            "This is a calm audiobook narrator voice with clear pronunciation, steady pacing, and natural pauses."
        )
    return (
        "Solen går långsamt upp över skogen. "
        "Det här är en lugn berättarröst för ljudböcker, med tydligt uttal, jämnt tempo och naturliga pauser."
    )


if __name__ == "__main__":
    main()
