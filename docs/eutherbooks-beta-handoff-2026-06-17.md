# EutherBooks beta handoff 2026-06-17

## Kort nulage

EutherBooks beta ar i ett anvandbart men fortfarande nattkansligt lage. Senaste stora stegen:

- EutherBooks Player APK 0.1.28 ar byggd pa servern och frontad via LAN.
- Dots MF ar huvud-TTS for beta.
- GrapheneOS Matcha EN ar integrerad som faktisk fallbackrenderer via `/home/nichlas/SpeechServices`.
- Nytt `auto-fallback`-lage finns genom EutherLink, EutherBooks API och EutherBooks Player.
- EutherBooks API visar nu `auto-*` roster som kan valjas av appen.
- Kapitel/prefetch/fortsattning fungerar battre an tidigare, men appen kan fortfarande fastna tills den far "attention". Nasta steg bor fokusera pa bakgrundsuppspelning och playback watchdog.

## Viktiga repo och ansvar

- Lokal EutherLink checkout: `/home/nichlas/EutherLink`
- Server EutherBooks checkout: `nichlas@192.168.32.186:/home/nichlas/EutherBooks`
- Server EutherOxide checkout: `nichlas@192.168.32.186:/home/nichlas/EutherOxide`
- SpeechServices fallbackmodell: `nichlas@192.168.32.186:/home/nichlas/SpeechServices`

For EutherOxide/EutherBooks appflodet ar servern pa `192.168.32.186` kallan for deployment och APK-bygge.

SSH-kommandon anvander normalt:

```bash
SSH_AUTH_SOCK=/home/nichlas/.ssh/agent/euther_codex.sock ssh -F /dev/null -i /home/nichlas/.ssh/euther_server nichlas@192.168.32.186
```

## Senaste commits

EutherLink:

- `8c4e942 Add auto fallback TTS routing`

EutherBooks:

- `197c820 Add auto fallback TTS model routing`

EutherOxide:

- `b7f9e17 Add auto fallback model selection`

## Senaste APK

EutherBooks Player:

- Version: `0.1.28`
- versionCode: `1028`
- LAN URL: `http://192.168.32.186:8080/downloads/EutherBooksPlayer-release-signed.apk`
- SHA-256: `0bdf6ffd37cc5c7394edb3dacb186df34d5d2850a812a5374c12160d6e970f78`

Verifierad med:

```bash
apksigner verify --verbose /home/nichlas/EutherBooksPlayer-release-signed.apk
aapt dump badging /home/nichlas/EutherBooksPlayer-release-signed.apk | head -1
sha256sum /home/nichlas/EutherBooksPlayer-release-signed.apk
curl -fsS -o /tmp/eutherbooks-apk-check.apk http://192.168.32.186:8080/downloads/EutherBooksPlayer-release-signed.apk
sha256sum /tmp/eutherbooks-apk-check.apk
```

Viktigt: `HEAD` mot downloadrouten kan ge 403 eftersom apphosten bara hanterar GET for download. Testa med GET.

## Aktuellt funktionslage

### EutherLink

`auto-fallback` ar tillagt som model backend.

Regel just nu:

- `auto-fallback` valjer normalt `dots.tts-mf`.
- Om spraket borjar med `en`, GrapheneOS Matcha ar redo, och antal aktiva Dots-jobb ar minst `EUTHERLINK_DOTS_MAIN_QUEUE_LIMIT`, valjs `grapheneos-matcha-en`.
- Default for `EUTHERLINK_DOTS_MAIN_QUEUE_LIMIT` ar `2`.

Health ska visa:

- Dots status `ready`
- `grapheneos_matcha.ready=true`
- `tts_parallelism=2`

Snabb health:

```bash
curl -fsS http://192.168.32.88:8765/health
```

### EutherBooks

API:t exponerar auto fallback-roster. Exempel som ska finnas:

- `auto-sv-female-warm`
- `auto-en-female-warm`
- `grapheneos-matcha-en`

Verifiera pa server:

```bash
curl -fsSL http://127.0.0.1:8088/voices | python3 -c 'import json,sys; voices=json.load(sys.stdin); print([v for v in voices if v.get("id") in {"auto-en-female-warm","auto-sv-female-warm","grapheneos-matcha-en"}])'
```

### EutherOxide / EutherBooks Player

Appen har modellvalet `Auto fallback`.

Statiska fallbackroster finns for:

- `auto-sv-female-warm`
- `auto-en-female-warm`

APK-bygge pa server:

```bash
cd /home/nichlas/EutherOxide
npm run build
scripts/eutherbooks-player-release-apk.sh
```

Efter nytt bygge, verifiera APK och starta om host:

```bash
systemctl is-active eutherbooks.service eutherhost.service
```

Om servicebehov:

```bash
sudo systemctl restart eutherbooks.service eutherhost.service
```

## Tester som redan korts

EutherLink:

```bash
python -m py_compile eutherlink.py scripts/render_graphene_matcha.py
```

Manuell auto fallback-smoke passerade:

- tom ko + engelska -> `dots.tts-mf`
- tva aktiva Dots-jobb + engelska -> `grapheneos-matcha-en`
- tva aktiva Dots-jobb + svenska -> `dots.tts-mf`

EutherBooks:

```bash
cd /home/nichlas/EutherBooks
pytest
```

Senaste resultat:

- `57 passed`

EutherOxide:

```bash
cd /home/nichlas/EutherOxide
npm run build
scripts/eutherbooks-player-release-apk.sh
```

Senaste APK-byggning gick klart och signerade APK:n korrekt.

## Kanda smasaker i arbetskopior

Lokal EutherLink:

- `docs/` ar untracked sedan tidigare. Denna handoff ligger ocksa dar.

Server EutherOxide:

- `webview/build-info.ts` var ostagad efter APK-bygge.
- Det ar bara genererad build-id/tidsstampelbrus och committades inte i senaste slice.

## Nasta rekommenderade arbetssteg

Prioritet 1: stabil nattlyssning.

Problemet fran beta:

- Ibland fastnar uppspelningen tills appen far attention.
- Appen ar inte helt redo for lyssning i sangen med slackt skarm.

Plan:

1. Lagg in playback watchdog i EutherBooks Player.
   - Mata faktisk audio-progress.
   - Upptack att positionen star still trots att state sager playing/buffering.
   - Logga stopporsak, current chapter, queued chapter, audio src och network status.
   - Forsok mjuk recovery: reload current source, reattach audio, eller hoppa till samma position.

2. Gor bakgrundslaget robust.
   - Kontrollera Android media session/notification.
   - Kontrollera audio focus.
   - Kontrollera om Tauri WebView/JS timers stryps nar skarmen slacks.
   - Vid behov flytta kritisk playback/progress-watchdog narmare native/foreground service-lagret.

3. Forbattra kapitelko och prefetch-insyn.
   - Visa nuvarande kapitel, nasta kapitel och prefetch-status.
   - Visa om nasta kapitel redan har audio, vantar pa TTS, eller har felat.
   - Gor foregaende/nasta kapitel robust aven om TTS-jobb ligger efter.

4. Bygg en liten beta-telemetrisida.
   - Senaste playback-stopp.
   - Senaste TTS-fel.
   - Modellval per kapitel.
   - Om `auto-fallback` slog over till GrapheneOS Matcha.
   - Jobbtider och ko-langd.

5. Efter watchdog/telemetri: testa faktisk nattprofil.
   - Starta lang bok.
   - Slack skarm.
   - Lat spela flera kapitel.
   - Kontrollera att nasta kapitel startar utan app-attention.
   - Kontrollera att fastnad playback antingen recoverar eller loggar tydlig orsak.

## Snabba restart- och kontrollkommandon

Server services:

```bash
SSH_AUTH_SOCK=/home/nichlas/.ssh/agent/euther_codex.sock ssh -F /dev/null -i /home/nichlas/.ssh/euther_server nichlas@192.168.32.186 'systemctl is-active eutherbooks.service eutherhost.service'
```

EutherBooks voices:

```bash
SSH_AUTH_SOCK=/home/nichlas/.ssh/agent/euther_codex.sock ssh -F /dev/null -i /home/nichlas/.ssh/euther_server nichlas@192.168.32.186 'curl -fsSL http://127.0.0.1:8088/voices | python3 -c "import json,sys; voices=json.load(sys.stdin); print(any(v.get(\"id\")==\"auto-en-female-warm\" and v.get(\"model_backend\")==\"auto-fallback\" for v in voices))"'
```

APK download:

```bash
SSH_AUTH_SOCK=/home/nichlas/.ssh/agent/euther_codex.sock ssh -F /dev/null -i /home/nichlas/.ssh/euther_server nichlas@192.168.32.186 'curl -fsS -o /tmp/eutherbooks-apk-check.apk http://192.168.32.186:8080/downloads/EutherBooksPlayer-release-signed.apk && sha256sum /tmp/eutherbooks-apk-check.apk'
```

EutherLink local health:

```bash
curl -fsS http://192.168.32.88:8765/health
```

## Om man maste ateruppta efter krasch

1. Kontrollera lokal status:

```bash
cd /home/nichlas/EutherLink
git status --short
curl -fsS http://192.168.32.88:8765/health
```

2. Kontrollera serverstatus:

```bash
SSH_AUTH_SOCK=/home/nichlas/.ssh/agent/euther_codex.sock ssh -F /dev/null -i /home/nichlas/.ssh/euther_server nichlas@192.168.32.186 'cd /home/nichlas/EutherBooks && git status --short && cd /home/nichlas/EutherOxide && git status --short && systemctl is-active eutherbooks.service eutherhost.service'
```

3. Om allt ar uppe, fortsatta med "stabil nattlyssning" ovan.

4. Om appen saknar Auto fallback:

- Installera om APK 0.1.28 fran LAN-URL.
- Kontrollera `/voices` att `auto-en-female-warm` finns.
- Starta om `eutherbooks.service` och `eutherhost.service`.

5. Om ljud inte spelar:

- Borja med EutherLink health.
- Kontrollera att Dots MF ar ready.
- Kontrollera att GrapheneOS Matcha ar ready.
- Testa en kort TTS-jobb manuellt innan appfelsokning.
