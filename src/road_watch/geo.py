"""GPS track handling and anonymous hotspot aggregation.

* ``load_gpx`` reads a GPX track (what most dashcam / GPS-logger apps export).
* ``position_at`` interpolates the position at a video timestamp.
* ``hotspot_event`` turns a violation into the ONLY thing that leaves the phone
  for the public map: a grid cell (~270 m square), the violation type, and the
  hour of the week. No plate, no image, no exact point.

The grid is plain lat/lon squares rather than H3 hexagons: no native library
(H3's DLL is blocked on some Windows machines), identical on phone and server.
"""

from __future__ import annotations

import bisect
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

IST = timezone(timedelta(hours=5, minutes=30))
CELL_DEG = 0.0025  # ~278 m north-south, ~265 m east-west at Hyderabad's latitude


def cell_id(lat: float, lon: float, size: float = CELL_DEG) -> str:
    """Stable id of the grid square containing the point, e.g. 'g2500:6976:31339'."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"bad coordinates {lat}, {lon}")
    return f"g{round(size * 1e6)}:{math.floor(lat / size)}:{math.floor(lon / size)}"


def cell_bounds(cell: str) -> tuple[float, float, float, float]:
    """(south, west, north, east) of a cell, for drawing it on a map."""
    prefix, i, j = cell.split(":")
    size = int(prefix[1:]) / 1e6
    s, w = int(i) * size, int(j) * size
    return s, w, s + size, w + size


@dataclass(frozen=True)
class Fix:
    t: datetime
    lat: float
    lon: float


def load_gpx(path: Path) -> list[Fix]:
    """Read track points (trkpt) with timestamps, sorted by time."""
    root = ET.parse(Path(path)).getroot()
    fixes = []
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] != "trkpt":
            continue
        t_el = next((c for c in el if c.tag.rsplit("}", 1)[-1] == "time"), None)
        if t_el is None or not t_el.text:
            continue
        t = datetime.fromisoformat(t_el.text.strip().replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        fixes.append(Fix(t, float(el.attrib["lat"]), float(el.attrib["lon"])))
    fixes.sort(key=lambda f: f.t)
    return fixes


def position_at(fixes: list[Fix], when: datetime, max_gap_s: float = 30.0) -> tuple[float, float] | None:
    """Linearly interpolate lat/lon at ``when``. None if outside the track or the gap is too big."""
    if not fixes:
        return None
    times = [f.t for f in fixes]
    i = bisect.bisect_left(times, when)
    if i < len(fixes) and fixes[i].t == when:
        return fixes[i].lat, fixes[i].lon
    if i == 0 or i == len(fixes):
        return None
    a, b = fixes[i - 1], fixes[i]
    span = (b.t - a.t).total_seconds()
    if span > max_gap_s:
        return None
    k = (when - a.t).total_seconds() / span
    return a.lat + k * (b.lat - a.lat), a.lon + k * (b.lon - a.lon)


def hotspot_event(violation: str, lat: float, lon: float, when: datetime, size: float = CELL_DEG) -> dict:
    """The anonymous record sent to the public map."""
    local = when.astimezone(IST)
    return {
        "cell": cell_id(lat, lon, size),
        "violation": violation,
        "hour_of_week": local.weekday() * 24 + local.hour,  # 0 = Monday 00:00 IST
        "day": local.date().isoformat(),
    }


def aggregate(events: list[dict], min_count: int = 3) -> dict[str, dict[str, int]]:
    """cell -> {violation: count}. Cells with fewer than ``min_count`` reports are
    dropped so a single report can't be traced back to one ride."""
    counts: dict[str, dict[str, int]] = {}
    for e in events:
        counts.setdefault(e["cell"], {}).setdefault(e["violation"], 0)
        counts[e["cell"]][e["violation"]] += 1
    return {c: v for c, v in counts.items() if sum(v.values()) >= min_count}
