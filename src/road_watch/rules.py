"""Turn per-frame detections into confirmed violations.

Detector-agnostic: feed it boxes from any model (YOLO COCO for person and
motorcycle, a helmet model for heads, a plate model for plates).

Precision matters more than recall here — a false accusation is worse than a
missed violation — so every rule requires the condition to persist across
several frames of the same tracked vehicle before it fires, and fires only once
per vehicle.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: str
    conf: float = 1.0

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _x_overlap_frac(person: Box, bike: Box) -> float:
    ix = max(0.0, min(person.x2, bike.x2) - max(person.x1, bike.x1))
    return ix / person.w if person.w else 0.0


def rider_score(person: Box, bike: Box) -> float:
    """0 if this person is clearly not on this two-wheeler, else a match score (higher = better).

    A rider's box: mostly within the bike horizontally, top above the bike,
    bottom (legs) inside the lower part of the bike box, and not wildly taller
    than the bike (that would be a pedestrian close to the camera).
    """
    xo = _x_overlap_frac(person, bike)
    if xo < 0.5:
        return 0.0
    if not (person.y1 < bike.y1):
        return 0.0
    if not (bike.y1 + 0.3 * bike.h <= person.y2 <= bike.y2 + 0.15 * bike.h):
        return 0.0
    if person.h > 3.0 * bike.h:
        return 0.0
    return xo


def assign_riders(bikes: list[Box], persons: list[Box]) -> dict[int, list[Box]]:
    """Map bike index -> riders. Each person rides at most one bike (best score wins)."""
    out: dict[int, list[Box]] = {i: [] for i in range(len(bikes))}
    for p in persons:
        scores = [(rider_score(p, b), i) for i, b in enumerate(bikes)]
        best, idx = max(scores, default=(0.0, -1))
        if best > 0:
            out[idx].append(p)
    return out


def head_without_helmet(rider: Box, heads: list[Box]) -> bool:
    """True if a 'no helmet' head box sits in the top 35% of this rider's box."""
    top_limit = rider.y1 + 0.35 * rider.h
    for hd in heads:
        cx, cy = hd.center
        if rider.x1 <= cx <= rider.x2 and rider.y1 - 0.1 * rider.h <= cy <= top_limit:
            return True
    return False


# ---- temporal confirmation -------------------------------------------------

@dataclass
class _Track:
    id: int
    box: Box
    last_frame: int
    history: dict[str, deque] = field(default_factory=dict)
    fired: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class Violation:
    track_id: int
    violation: str       # "triple_riding" | "no_helmet"
    frame: int
    time_s: float
    confidence: float    # fraction of recent frames that showed it
    box: Box
    riders: int


class ViolationTracker:
    """Feed one frame at a time; get confirmed violations back.

    A violation fires when it is seen in at least ``min_hits`` of the last
    ``window`` frames in which that vehicle was visible.
    """

    BIKE_CLASSES = ("motorcycle", "motorbike", "scooter", "two_wheeler")
    PERSON_CLASSES = ("person", "rider")
    NO_HELMET_CLASSES = ("head", "no_helmet", "without_helmet")

    def __init__(self, window: int = 5, min_hits: int = 3, match_iou: float = 0.3, max_gap: int = 10):
        if not 1 <= min_hits <= window:
            raise ValueError("need 1 <= min_hits <= window")
        self.window, self.min_hits, self.match_iou, self.max_gap = window, min_hits, match_iou, max_gap
        self._tracks: list[_Track] = []
        self._next_id = 1

    def _match(self, bike: Box, frame: int) -> _Track:
        best, best_iou = None, self.match_iou
        for t in self._tracks:
            if frame - t.last_frame > self.max_gap:
                continue
            v = iou(t.box, bike)
            if v >= best_iou:
                best, best_iou = t, v
        if best is None:
            best = _Track(self._next_id, bike, frame)
            self._next_id += 1
            self._tracks.append(best)
        best.box, best.last_frame = bike, frame
        return best

    def update(self, frame: int, time_s: float, boxes: list[Box]) -> list[Violation]:
        bikes = [b for b in boxes if b.cls in self.BIKE_CLASSES]
        persons = [b for b in boxes if b.cls in self.PERSON_CLASSES]
        heads = [b for b in boxes if b.cls in self.NO_HELMET_CLASSES]
        riders_by_bike = assign_riders(bikes, persons)

        fired: list[Violation] = []
        used: set[int] = set()
        for i, bike in enumerate(bikes):
            track = self._match(bike, frame)
            if track.id in used:  # two boxes matched one track; treat second as new next frame
                continue
            used.add(track.id)
            riders = riders_by_bike[i]
            checks = {
                "triple_riding": len(riders) >= 3,
                "no_helmet": any(head_without_helmet(r, heads) for r in riders),
            }
            for name, seen in checks.items():
                hist = track.history.setdefault(name, deque(maxlen=self.window))
                hist.append(seen)
                hits = sum(hist)
                if name not in track.fired and hits >= self.min_hits:
                    track.fired.add(name)
                    fired.append(
                        Violation(track.id, name, frame, time_s, hits / len(hist), bike, len(riders))
                    )

        self._tracks = [t for t in self._tracks if frame - t.last_frame <= self.max_gap]
        return fired
