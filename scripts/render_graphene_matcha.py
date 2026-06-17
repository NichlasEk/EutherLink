#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import onnxruntime as ort
import soundfile as sf
from misaki.en import G2P


SAMPLE_RATE = 22_050
DECODER_WINDOW = 64
HOP_LENGTH = 256
N_TIMESTEPS = 5
TEMPERATURE = 0.667

PUNCTUATION = ";:,.!?¡¿—…\"«»“” "
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
LETTERS_IPA = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθ"
    "œɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)
LETTERS_IPA_NONSTANDARD = "ᵊ"
SYMBOLS = "_" + PUNCTUATION + LETTERS + LETTERS_IPA + LETTERS_IPA_NONSTANDARD
SYMBOL_TO_ID = {char: index for index, char in enumerate(SYMBOLS)}
PAD_ID = SYMBOL_TO_ID["_"]


def split_text(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    index = 0
    while index < len(text):
        score = 0
        while index < len(text):
            char = text[index]
            index += 1
            score += 4 if char.isdigit() else 11 if char == "-" else 1
            segment = text[start:index]
            if index == len(text) or should_break(segment, char, score):
                break
        chunk = text[start:index].strip()
        if chunk:
            chunks.append(chunk)
        start = index
    return chunks


def should_break(segment: str, char: str, score: int) -> bool:
    if score >= 500:
        return True
    if char.isalpha():
        return False
    if segment.endswith((". ", "? ", "! ", "\n")):
        return True
    if score >= 350 and segment.endswith(", "):
        return True
    if score >= 450 and char.isspace():
        return True
    if score >= 250 and segment.endswith(
        (": ", " ;", "—", " ...", "... ", " …", "… ", ' "', '" ', " “", "” ", " (", ") ")
    ):
        return True
    return False


def encode_symbols(phoneme_text: str) -> np.ndarray:
    ids: list[int] = []
    for char in phoneme_text:
        symbol_id = SYMBOL_TO_ID.get(char)
        if symbol_id is None:
            continue
        ids.extend((PAD_ID, symbol_id))
    ids.append(PAD_ID)
    return np.asarray([ids], dtype=np.int64)


def normalize_peak(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32)
    max_abs = float(np.max(np.abs(audio)))
    if max_abs <= 0:
        return audio.astype(np.float32)
    return (audio.astype(np.float32) * min(1.0, peak / max_abs)).astype(np.float32)


class MatchaRenderer:
    def __init__(self, assets_dir: Path) -> None:
        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(str(assets_dir / "encoder.onnx"), providers=providers)
        self.decoder = ort.InferenceSession(str(assets_dir / "decoder.onnx"), providers=providers)
        self.g2p = G2P(trf=False, british=False, fallback=None)

    def render(self, text: str, *, length_scale: float = 1.0) -> tuple[np.ndarray, dict[str, Any]]:
        chunks = split_text(text)
        audio_parts: list[np.ndarray] = []
        phoneme_chunks: list[str] = []
        y_lengths: list[int] = []
        timings: list[dict[str, float]] = []
        silence = np.zeros(int(SAMPLE_RATE * 0.18), dtype=np.float32)

        for index, chunk in enumerate(chunks):
            chunk_started = time.perf_counter()
            phonemes = self.g2p(chunk)[0]
            phonemes = re.sub(r"\s+", " ", phonemes).strip()
            phoneme_chunks.append(phonemes)
            ids = encode_symbols(phonemes)
            if ids.shape[1] <= 1:
                continue

            encode_started = time.perf_counter()
            y_length, mu_y, y_mask = self.encode(ids, length_scale)
            decode_started = time.perf_counter()
            pcm = self.decode(y_length, mu_y, y_mask)
            y_lengths.append(y_length)
            audio_parts.append(pcm)
            if index + 1 < len(chunks):
                audio_parts.append(silence)
            timings.append(
                {
                    "phoneme_sec": encode_started - chunk_started,
                    "encode_sec": decode_started - encode_started,
                    "decode_sec": time.perf_counter() - decode_started,
                }
            )

        audio = normalize_peak(np.concatenate(audio_parts) if audio_parts else np.zeros(0, dtype=np.float32))
        perf = {
            "chunk_count": len(chunks),
            "phoneme_chunks": phoneme_chunks,
            "y_lengths": y_lengths,
            "timings": timings,
        }
        return audio, perf

    def encode(self, ids: np.ndarray, length_scale: float) -> tuple[int, np.ndarray, np.ndarray]:
        result = self.encoder.run(
            None,
            {
                "x": ids,
                "x_lengths": np.asarray([ids.shape[1]], dtype=np.int64),
                "length_scale": np.asarray([length_scale], dtype=np.float32),
            },
        )
        y_length = int(result[0][0])
        return y_length, np.asarray(result[1][0], dtype=np.float32), np.asarray(result[2][0], dtype=np.float32)

    def decode(self, y_length: int, mu_y: np.ndarray, y_mask: np.ndarray) -> np.ndarray:
        parts: list[np.ndarray] = []
        batches = int(math.ceil(y_length / DECODER_WINDOW))
        for batch in range(batches):
            start = batch * DECODER_WINDOW
            end = start + DECODER_WINDOW
            mu_window = np.zeros((1, mu_y.shape[0], DECODER_WINDOW), dtype=np.float32)
            mask_window = np.ones((1, y_mask.shape[0], DECODER_WINDOW), dtype=np.float32)
            copy_mu = max(0, min(DECODER_WINDOW, mu_y.shape[1] - start))
            copy_mask = max(0, min(DECODER_WINDOW, y_mask.shape[1] - start))
            if copy_mu:
                mu_window[0, :, :copy_mu] = mu_y[:, start : start + copy_mu]
            if copy_mask:
                mask_window[0, :, :copy_mask] = y_mask[:, start : start + copy_mask]
            pcm = self.decoder.run(
                None,
                {
                    "mu_y": mu_window,
                    "y_mask": mask_window,
                    "n_timesteps": np.asarray([N_TIMESTEPS], dtype=np.int64),
                    "temperature": np.asarray([TEMPERATURE], dtype=np.float32),
                },
            )[0][0].astype(np.float32)
            unpadded = max(0, min(end, y_length) - start) * HOP_LENGTH
            parts.append(pcm[:unpadded])
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render GrapheneOS Matcha EN TTS to WAV")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--output-wav", required=True)
    parser.add_argument("--manifest-json", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
    root = Path(request.get("assets_dir") or "/home/nichlas/SpeechServices/app/src/main/res/raw")
    text = str(request["text"])
    length_scale = float(request.get("length_scale") or 1.0)

    started = time.perf_counter()
    renderer = MatchaRenderer(root)
    audio, perf = renderer.render(text, length_scale=length_scale)
    output_path = Path(args.output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, SAMPLE_RATE, subtype="PCM_16")
    manifest = {
        "audio_path": str(output_path),
        "sample_rate": SAMPLE_RATE,
        "duration_sec": float(len(audio) / SAMPLE_RATE) if SAMPLE_RATE else 0.0,
        "perf": {**perf, "total_sec": time.perf_counter() - started},
    }
    Path(args.manifest_json).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
