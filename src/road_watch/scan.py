"""Scan a dashcam video (or image) for violations.

    python -m road_watch scan ride.mp4 --model models/yolo11n.onnx --out out/
        [--gpx ride.gpx] [--start 2026-11-14T10:00:00+05:30]
        [--helmet-model models/helmet.onnx] [--plate-model models/plate.onnx]
        [--key device.key]   # also seal the ORIGINAL video as evidence

Writes:
  out/events.jsonl       one confirmed violation per line (private: stays on device)
  out/keyframes/*.jpg    the frame where each violation was confirmed, box drawn
  out/hotspots.jsonl     anonymous records for the public map (no plate, no image)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

from .detect import YoloOnnx
from .evidence import Event, load_key, seal
from .geo import hotspot_event, load_gpx, position_at
from .rules import Box, ViolationTracker

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _frames(path: Path, sample_fps: float):
    """Yield (frame_index, time_s, bgr). Images yield one frame repeated so the
    tracker's persistence check can still be exercised in a smoke test."""
    if path.suffix.lower() in IMAGE_EXT:
        img = cv2.imread(str(path))
        if img is None:
            raise SystemExit(f"cannot read image {path}")
        for i in range(5):
            yield i, i / sample_fps, img
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / sample_fps))
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                yield i, i / fps, frame
        i += 1
    cap.release()


def _draw(img, boxes: list[Box], highlight: Box, label: str):
    out = img.copy()
    for b in boxes:
        cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (180, 180, 180), 1)
    cv2.rectangle(out, (int(highlight.x1), int(highlight.y1)), (int(highlight.x2), int(highlight.y2)), (0, 0, 255), 3)
    cv2.putText(out, label, (int(highlight.x1), max(20, int(highlight.y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out


def scan(
    video: Path,
    model: Path,
    out_dir: Path,
    *,
    gpx: Path | None = None,
    start: datetime | None = None,
    helmet_model: Path | None = None,
    sample_fps: float = 5.0,
    window: int = 5,
    min_hits: int = 3,
    key: Path | None = None,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "keyframes").mkdir(exist_ok=True)

    detectors = [YoloOnnx(model, keep={"person", "motorcycle"})]
    if helmet_model:
        # Public helmet datasets name the class differently; map the common spellings.
        detectors.append(YoloOnnx(helmet_model, keep={"no_helmet"}, rename={
            "Without Helmet": "no_helmet", "without_helmet": "no_helmet", "head": "no_helmet",
            "no-helmet": "no_helmet", "NoHelmet": "no_helmet",
        }))
    tracker = ViolationTracker(window=window, min_hits=min_hits)
    tracker.NO_HELMET_CLASSES = ("no_helmet",)

    if start is None:
        start = datetime.fromtimestamp(video.stat().st_mtime, tz=timezone.utc)
    fixes = load_gpx(gpx) if gpx else []

    records, evidence_events = [], []
    for frame_idx, t, img in _frames(video, sample_fps):
        boxes = [b for d in detectors for b in d(img)]
        for v in tracker.update(frame_idx, t, boxes):
            when = start + timedelta(seconds=t)
            pos = position_at(fixes, when) if fixes else None
            kf = out_dir / "keyframes" / f"{video.stem}_{frame_idx:06d}_{v.violation}_t{v.track_id}.jpg"
            cv2.imwrite(str(kf), _draw(img, boxes, v.box, f"{v.violation} ({v.riders} riders)"))
            rec = {
                "video": video.name,
                "offset_s": round(t, 2),
                "time": when.isoformat(),
                "violation": v.violation,
                "confidence": round(v.confidence, 2),
                "riders": v.riders,
                "track_id": v.track_id,
                "box": [round(x, 1) for x in (v.box.x1, v.box.y1, v.box.x2, v.box.y2)],
                "lat": pos[0] if pos else None,
                "lon": pos[1] if pos else None,
                "keyframe": kf.name,
                "status": "pending_review",
            }
            records.append(rec)
            evidence_events.append(Event(rec["offset_s"], v.violation, rec["confidence"],
                                         rec["lat"], rec["lon"]))

    with open(out_dir / "events.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "hotspots.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            if r["lat"] is not None:
                f.write(json.dumps(hotspot_event(r["violation"], r["lat"], r["lon"],
                                                 datetime.fromisoformat(r["time"]))) + "\n")
    if key and records and video.suffix.lower() not in IMAGE_EXT:
        seal(video, evidence_events, load_key(key), recorded_at=start, app_version="0.1.0")
    return records
