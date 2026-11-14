# Road Watch

Dashcam-based traffic-violation **evidence** for Hyderabad / Telangana.

Road Watch does not issue challans and is not a reporting channel. It turns
dashcam footage into evidence the official channel can act on, and turns
violations into an anonymous public hotspot map.

Design doc: `../brainstorms/2026-11-14-traffic-reporting-design.html`

## What works today (phase 0)

| Module | What it does | Tests |
|---|---|---|
| `plates.py` | Parses noisy OCR text into Indian registration numbers (standard, Delhi category letter, BH series, TS/TG). Fixes common OCR confusions only where the format demands it; flags non-standard plates | 21 |
| `evidence.py` | Seals the **original** video: SHA-256 + Ed25519 signature from a device key. Detects edited video, edited manifest, and re-signing with another key | 7 |
| `rules.py` | Detector-agnostic rules: which persons ride which two-wheeler; triple riding; no helmet; IoU tracking and multi-frame confirmation; fires once per vehicle | 11 |
| `geo.py` | GPX loading, position at a video timestamp, ~270 m grid cells for the public map, drops cells with fewer than 3 reports | 8 |
| `detect.py` | ONNX Runtime wrapper for YOLOv8/YOLO11 models (same file runs on Android) | smoke-tested |
| `scan.py` | Video/image -> confirmed violations, key frames, hotspot records, optional evidence seal | smoke-tested |

### Smoke test, pretrained COCO `yolo11n`, Wikimedia photos from Hyderabad

| Photo | Expected | Got |
|---|---|---|
| ThreePersonsMotorBikeHyderabad.jpg | triple riding | triple riding ✅ |
| Three persons riding … talking on mobile | triple riding | triple riding ✅ (2 standing pedestrians correctly excluded) |
| Three persons riding on a 2 wheeler | triple riding | missed — 3rd rider at 0.29 confidence, below 0.35 threshold |
| Two Wheeler Hyderabad.jpg | nothing | nothing ✅ |

Four photos prove the pipeline runs end to end. They are not an accuracy measure.

## Usage

```bash
uv sync --extra detect
uv run python -m pytest -q                      # 47 tests

uv run python -m road_watch plate "TG09EA1234" "T609EA12B4" "BOSS"
uv run python -m road_watch keygen device.key
uv run --extra detect python -m road_watch scan ride.mp4 --gpx ride.gpx \
    --start 2026-11-14T10:00:00+05:30 --key device.key --out out/
uv run python -m road_watch verify ride.mp4
```

Commands run as `python -m …` because Windows Application Control on this
machine blocks the `.exe` launchers that pip/uv generate (and the `h3` DLL,
which is why the map uses a pure-Python grid).

### Getting the model

`models/` is git-ignored. Export once with Ultralytics (needs PyTorch, in a
separate environment):

```python
from ultralytics import YOLO
YOLO("yolo11n.pt").export(format="onnx", imgsz=640, opset=17, simplify=True)
```

> **Licence:** Ultralytics YOLO weights are AGPL-3.0. Fine for research and an
> open-source repo; a closed-source app needs their commercial licence or an
> Apache-2.0 detector (e.g. YOLOX, RTMDet).

## Guardrails built into the code

- Violations need **3 of the last 5** analysed frames on the same tracked vehicle, and fire once.
- Everything is `pending_review`; a human confirms before anything is sent.
- `events.jsonl` (plates, exact GPS) stays on the device. Only `hotspots.jsonl`
  (grid cell, violation type, hour of week) is meant for the public map.
- The original video is never trimmed or re-encoded; events point into it by offset.

## Next

1. Record 2–3 hours of your own dashcam rides (phone mount + GPS logger app exporting GPX).
2. Hand-label ~300 frames → first real precision / recall numbers per violation.
3. Train a helmet / no-helmet model and a plate detector (Colab or Kaggle GPU;
   IDD from IIIT Hyderabad + public helmet and Indian-plate datasets).
4. Plate OCR on the plate crops -> `plates.parse_plate`.
5. Go / no-go: ≥95% precision on your own footage.
