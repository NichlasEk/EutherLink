# EutherLink

EutherLink is a small LAN service for running heavy local AI jobs on the RTX 4090
machine and letting smaller Euther services call it over HTTP.

The first worker is VoxCPM2 text-to-speech for audiobook-style rendering.
`dots.tts-soar` and `dots.tts-mf` are also available as optional backends through the separate
library under `/home/nichlas/ai/dots_tts`.

## Start

```sh
/home/nichlas/EutherLink/start.sh
```

Default bind:

```text
0.0.0.0:8765
```

Default TTS scheduling:

```text
EUTHERLINK_TTS_PARALLELISM=2
```

Two EutherLink TTS jobs can be active at once. Dots jobs still use a single
protected render slot around the persistent Dots worker, so a second Dots job is
accepted and reports `Waiting for Dots render slot` instead of racing the model.

## Health

```sh
curl http://127.0.0.1:8765/health
```

## Create A TTS Job

```sh
curl -s http://127.0.0.1:8765/v1/tts/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Det här är ett kort test av svensk text till tal.",
    "voice_instruction": "A calm, warm Swedish audiobook narrator with clear pronunciation and natural pacing.",
    "model_backend": "voxcpm2",
    "output_format": "opus"
  }'
```

Poll the returned `status_url`, then download `audio_url` when status is `done`.

```sh
curl -L http://127.0.0.1:8765/v1/tts/jobs/JOB_ID/audio -o test.opus
```

## Optional dots.tts Backends

The Dots backend is isolated from VoxCPM2:

```text
/home/nichlas/ai/dots_tts/dots.tts              # Space/source checkout
/home/nichlas/ai/dots_tts/dots.tts/.venv        # Python 3.12 runtime
/home/nichlas/ai/dots_tts/models/dots.tts-soar  # downloaded model weights
/home/nichlas/ai/dots_tts/models/dots.tts-mf    # faster MeanFlow weights
/home/nichlas/ai/dots_tts/render_eutherlink.py  # subprocess renderer
```

Use SOAR by setting `"model_backend": "dots.tts-soar"` or the faster MeanFlow
student by setting `"model_backend": "dots.tts-mf"`. Dots requires a reference
WAV and matching `prompt_text`, so send either `prompt_wav_base64` or
`reference_wav_base64`.
