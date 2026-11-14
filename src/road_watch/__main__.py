"""Command line: python -m road_watch <command> ...

Run as a module (not a console script) because Windows Application Control
blocks the .exe launchers that pip/uv generate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def _start_time(text: str) -> datetime:
    try:
        t = datetime.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an ISO 8601 time: {text!r}") from None
    if t.tzinfo is None:
        # A bare time would be compared against the GPS track's UTC times and
        # silently land hours away (or crash); make the user say which zone.
        raise argparse.ArgumentTypeError(f"{text!r} has no UTC offset; write e.g. {text}+05:30")
    return t


def _positive(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        v = 0.0
    if not v > 0:
        raise argparse.ArgumentTypeError(f"must be a positive number, got {text!r}")
    return v


def main(argv: list[str] | None = None) -> int:
    from . import __version__

    p = argparse.ArgumentParser(prog="road_watch")
    p.add_argument("--version", action="version", version=f"road_watch {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="detect violations in a video or image")
    s.add_argument("video", type=Path)
    s.add_argument("--model", type=Path,
                   default=Path(os.environ.get("ROAD_WATCH_MODEL", "models/yolo11n.onnx")),
                   help="ONNX detector (default: $ROAD_WATCH_MODEL or models/yolo11n.onnx)")
    s.add_argument("--helmet-model", type=Path)
    s.add_argument("--out", type=Path, default=Path("out"))
    s.add_argument("--gpx", type=Path)
    s.add_argument("--start", type=_start_time,
                   help="video start time, ISO 8601 with offset (default: file modified time minus duration)")
    s.add_argument("--fps", type=_positive, default=5.0, help="frames per second to analyse")
    s.add_argument("--window", type=int, default=5)
    s.add_argument("--min-hits", type=int, default=3)
    s.add_argument("--key", type=Path, help="device key: seal the original video as evidence")

    pl = sub.add_parser("plate", help="check plate text, e.g. 'TG09EA1234'")
    pl.add_argument("text", nargs="+")

    k = sub.add_parser("keygen", help="create a device signing key")
    k.add_argument("path", type=Path)

    v = sub.add_parser("verify", help="verify a sealed video")
    v.add_argument("video", type=Path)
    v.add_argument("--public-key", help="trusted device public key (base64)")

    a = p.parse_args(argv)
    if a.cmd == "scan" and not 1 <= a.min_hits <= a.window:
        s.error("need 1 <= --min-hits <= --window")

    if a.cmd == "plate":
        from .plates import parse_plate
        for t in a.text:
            r = parse_plate(t)
            print(json.dumps({"input": t, "kind": r.kind, "normalized": r.normalized,
                              "corrections": r.corrections, "telangana": r.is_telangana,
                              "reasons": list(r.reasons)}))
        return 0

    if a.cmd == "keygen":
        from .evidence import generate_key, public_key_b64
        try:
            pub = generate_key(a.path)
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print("public key:", public_key_b64(pub))
        return 0

    if a.cmd == "verify":
        from .evidence import verify
        r = verify(a.video, a.public_key)
        print("OK" if r.ok else "FAILED")
        for prob in r.problems:
            print(" -", prob)
        if r.ok and a.public_key is None:
            print(f"note: signed with key {r.signer}")
            print("      Without --public-key this only shows the video matches its manifest,")
            print("      not which device sealed it.")
        return 0 if r.ok else 1

    if a.cmd == "scan":
        from .scan import guess_start, scan
        try:
            if a.start is None and a.video.is_file():
                a.start = guess_start(a.video)
                print(f"note: no --start given; assuming recording began {a.start.isoformat()} "
                      "(file modified time minus duration)", file=sys.stderr)
            recs = scan(a.video, a.model, a.out, gpx=a.gpx, start=a.start, helmet_model=a.helmet_model,
                        sample_fps=a.fps, window=a.window, min_hits=a.min_hits, key=a.key)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"{len(recs)} violation(s) -> {a.out / 'events.jsonl'}")
        for r in recs:
            loc = f" at {r['lat']:.5f},{r['lon']:.5f}" if r["lat"] is not None else ""
            print(f"  {r['offset_s']:>7.1f}s  {r['violation']:<14} riders={r['riders']} conf={r['confidence']}{loc}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
