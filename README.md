# Road Watch

[![ci](https://github.com/kprsnt2/road-watch/actions/workflows/ci.yml/badge.svg)](https://github.com/kprsnt2/road-watch/actions/workflows/ci.yml)

Dashcam traffic-violation **evidence** for Hyderabad / Telangana.

Road Watch does not issue challans and is not a reporting channel. It turns
dashcam footage into evidence that the official traffic-police channel can act
on, and turns violations into an anonymous public hotspot map.

- Detects **triple riding** and **riding without a helmet** in video, confirmed across several frames.
- Checks **number plates** against Indian formats (TS/TG, Delhi, BH series) and flags non-standard ones.
- **Seals the original video** with a hash and a device signature so edits are detectable.
- Exports **anonymous hotspots** (≈270 m grid cell, violation type, hour) — no plates, no images.

> **Status: phase 0 (research).** The pipeline works end to end; accuracy on real
> dashcam footage has not been measured yet. Every detection is `pending_review` —
> a person must check it before reporting anything.

## Quick start (Docker)

```bash
docker pull ghcr.io/kprsnt2/road-watch:latest
cd folder-with-your-video

docker run --rm -v "$PWD:/data" ghcr.io/kprsnt2/road-watch keygen device.key
docker run --rm -v "$PWD:/data" ghcr.io/kprsnt2/road-watch scan ride.mp4 \
    --gpx ride.gpx --start 2026-11-14T10:00:00+05:30 --key device.key --out out
docker run --rm -v "$PWD:/data" ghcr.io/kprsnt2/road-watch verify ride.mp4
```

PowerShell: use `-v "${PWD}:/data"`. More in [docs/DEPLOY.md](docs/DEPLOY.md).

Example output:

```
1 violation(s) -> out/events.jsonl
      0.4s  triple_riding  riders=3 conf=1.0 at 17.40654,78.46917
```

## Install from source

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/kprsnt2/road-watch && cd road-watch
uv sync --extra detect
uv run python -m pytest -q
```

Get the detector (one time, needs PyTorch — use a separate environment):

```bash
uv run --with ultralytics --with onnx --with onnxslim python -c \
  "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', imgsz=640, opset=17, simplify=True)"
mkdir -p models && mv yolo11n.onnx models/
```

Then:

```bash
uv run python -m road_watch plate "TG09EA1234" "T609EA12B4" "BOSS"
uv run python -m road_watch scan ride.mp4 --gpx ride.gpx --start 2026-11-14T10:00:00+05:30 --out out
```

> Windows with Application Control: run everything as `python -m …`. The `.exe`
> launchers uv/pip generate (e.g. `pytest.exe`) may be blocked.

## Recording footage

1. **Mount the phone** (dashboard or handlebar mount). Never film handheld while driving.
2. Start a **GPS logger app** that exports **GPX** (any app with GPX export works).
3. Record with the phone's camera or a dashcam app, landscape, 1080p is plenty.
4. Note the **start time** of the video (or rely on the file's modified time).
5. After the ride: `scan ride.mp4 --gpx ride.gpx --start <time>`.

## Commands

| Command | What it does |
|---|---|
| `scan VIDEO` | Detect violations. Options: `--gpx`, `--start`, `--fps` (default 5), `--window`/`--min-hits` (default 3 of 5 frames), `--model`, `--helmet-model`, `--key`, `--out` |
| `plate TEXT…` | Parse and validate plate text |
| `keygen PATH` | Create a device signing key; prints the public key |
| `verify VIDEO [--public-key B64]` | Check a sealed video hasn't been edited or re-signed |

### Outputs of `scan`

| File | Contains | Share? |
|---|---|---|
| `out/events.jsonl` | Each violation: time offset, type, riders, box, GPS | **Private** |
| `out/keyframes/*.jpg` | The confirming frame with the vehicle boxed | **Private** — attach to an official report |
| `out/hotspots.jsonl` | Grid cell, violation type, hour of week | Public map only |
| `VIDEO.evidence.json` | Hash + signature of the original video (with `--key`) | With the report |

## How it works

| Module | Role |
|---|---|
| `detect.py` | ONNX Runtime wrapper for YOLOv8/YOLO11 models; the same `.onnx` runs on Android |
| `rules.py` | Assigns people to two-wheelers, applies violation rules, tracks vehicles across frames, fires once per vehicle |
| `plates.py` | Plate parsing with format-aware OCR correction |
| `evidence.py` | SHA-256 + Ed25519 sealing and verification of the untouched original video |
| `geo.py` | GPX loading, position interpolation, anonymous grid cells, minimum-count aggregation |
| `scan.py` | Ties it together |

Design choices that favour **precision over recall**: a violation must appear in
3 of the last 5 analysed frames of the same tracked vehicle; low-confidence
detections are dropped even if that misses some violations; nothing is
reported automatically.

### Smoke test (pretrained COCO `yolo11n`, Wikimedia Commons photos from Hyderabad)

| Photo | Expected | Got |
|---|---|---|
| ThreePersonsMotorBikeHyderabad.jpg | triple riding | ✅ triple riding |
| Three persons riding on a 2 wheeler and talking on mobile in Hyderabad | triple riding | ✅ triple riding (2 standing pedestrians correctly excluded) |
| Three persons riding on a 2 wheeler in Hyderabad | triple riding | ❌ missed — 3rd rider at 0.29 confidence |
| Two Wheeler Hyderabad.jpg | nothing | ✅ nothing |

Four photos show the pipeline runs; they are not an accuracy measure.

## Roadmap

1. **Phase 0 (now):** collect dashcam rides, hand-label ~300 frames, measure precision/recall per violation. Go/no-go: ≥95% precision.
2. Train helmet and plate detectors (India Driving Dataset from IIIT Hyderabad, public helmet and Indian-plate datasets), add plate OCR.
3. Android app: dashcam mode, on-device detection, review queue, evidence export to the official channel.
4. Public hotspot map.

## Privacy & responsible use

- Plates and exact locations never leave your device unless you send a report yourself.
- The public map only shows grid cells with at least 3 reports.
- Do not use this to harass individuals. Report through official channels only.
- Plate numbers linked to locations are personal data under India's DPDP Act — treat outputs accordingly.

## Licence

Code: MIT — see [LICENSE](LICENSE).
The default detector (Ultralytics YOLO11) is **AGPL-3.0**; it is downloaded at
build time, not included in this repo. A closed-source product needs an
Ultralytics licence or an Apache-2.0 detector (e.g. YOLOX, RTMDet).
