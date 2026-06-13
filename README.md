# EutherLink

EutherLink is a small LAN service for running heavy local AI jobs on the RTX 4090
machine and letting smaller Euther services call it over HTTP.

The first worker is VoxCPM2 text-to-speech for audiobook-style rendering.

## Start

```sh
/home/nichlas/EutherLink/start.sh
```

Default bind:

```text
0.0.0.0:8765
```

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
    "output_format": "opus"
  }'
```

Poll the returned `status_url`, then download `audio_url` when status is `done`.

```sh
curl -L http://127.0.0.1:8765/v1/tts/jobs/JOB_ID/audio -o test.opus
```
