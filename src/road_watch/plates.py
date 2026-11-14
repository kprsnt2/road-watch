"""Indian vehicle registration plate parsing and validation.

Takes raw OCR text (noisy: spaces, dashes, O/0 and I/1 confusions) and decides
whether it is a well-formed Indian registration number.

Formats handled:
  * Standard:  SS DD [L]{0,3} NNNN     e.g. TG 09 EA 1234, TS 07 FK 0420
               Delhi allows a category letter after the district: DL 3C AB 1234
  * BH series: YY BH NNNN L{1,2}       e.g. 22 BH 1234 AA

A plate that fails to parse is reported as ``nonstandard`` — that covers both
genuinely illegal plates (fancy fonts, names, stickers) and plain OCR failures,
so it always needs a human look before it is used as evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product

# Current and legacy state / UT codes.
STATE_CODES = frozenset(
    {
        "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
        "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
        "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TG", "TN", "TR", "TS",
        "UK", "UA", "UP", "WB",
    }
)
TELANGANA_CODES = frozenset({"TS", "TG"})  # TS before 2024, TG for new registrations

# OCR confusions, applied only where the format expects the other kind of char.
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B", "4": "A", "7": "T"}
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5",
             "G": "6", "B": "8", "A": "4", "T": "7"}

# Series letters never issued because they look like digits.
_SERIES_FORBIDDEN = frozenset("IO")


@dataclass(frozen=True)
class PlateResult:
    raw: str
    kind: str  # "standard" | "bh" | "nonstandard" | "empty"
    normalized: str = ""  # canonical spaced form, e.g. "TG 09 EA 1234"
    state: str = ""
    district: str = ""
    series: str = ""
    number: str = ""
    corrections: int = 0  # how many characters were OCR-corrected
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def valid(self) -> bool:
        return self.kind in ("standard", "bh")

    @property
    def is_telangana(self) -> bool:
        return self.state in TELANGANA_CODES

    @property
    def compact(self) -> str:
        return self.normalized.replace(" ", "")


def clean(raw: str) -> str:
    """Uppercase and strip everything that is not A-Z or 0-9 (IND strip, dots, dashes...)."""
    text = raw.upper()
    # The HSRP "IND" marker is often read as a prefix by OCR.
    text = re.sub(r"^\W*IND\W+", "", text)
    return re.sub(r"[^A-Z0-9]", "", text)


def _coerce(chars: str, want: str) -> tuple[str, int] | None:
    """Coerce chars to all-letters ('L') or all-digits ('D'). Returns (text, corrections)."""
    out, fixes = [], 0
    for c in chars:
        if want == "L":
            if c.isalpha():
                out.append(c)
            elif c in _TO_LETTER:
                out.append(_TO_LETTER[c])
                fixes += 1
            else:
                return None
        else:
            if c.isdigit():
                out.append(c)
            elif c in _TO_DIGIT:
                out.append(_TO_DIGIT[c])
                fixes += 1
            else:
                return None
    return "".join(out), fixes


def _standard_candidates(text: str):
    """Yield (score, PlateResult-kwargs) for every way text can split as a standard plate."""
    n = len(text)
    for d_len, delhi_cat, s_len, num_len in product((1, 2), (0, 1), (0, 1, 2, 3), (1, 2, 3, 4)):
        if 2 + d_len + delhi_cat + s_len + num_len != n:
            continue
        i = 0
        parts = {}
        state = _coerce(text[i:i + 2], "L"); i += 2
        district = _coerce(text[i:i + d_len], "D"); i += d_len
        cat = _coerce(text[i:i + delhi_cat], "L") if delhi_cat else ("", 0); i += delhi_cat
        series = _coerce(text[i:i + s_len], "L") if s_len else ("", 0); i += s_len
        number = _coerce(text[i:i + num_len], "D")
        if None in (state, district, cat, series, number):
            continue
        if state[0] not in STATE_CODES:
            continue
        if delhi_cat and state[0] != "DL":
            continue
        if set(series[0]) & _SERIES_FORBIDDEN:
            continue
        if int(number[0]) == 0 or int(district[0]) == 0:
            continue
        parts = dict(
            state=state[0],
            district=district[0].zfill(2) + cat[0],
            series=series[0],
            number=number[0].zfill(4),
        )
        fixes = state[1] + district[1] + cat[1] + series[1] + number[1]
        # Prefer fewer corrections, then the most common shapes:
        # 2-digit district, 4-digit number, 2-letter series. 3-letter series are
        # rare, so "DL 3C AB" beats "DL 3 CAB".
        shape_penalty = (d_len != 2) + (num_len != 4) + {0: 0.5, 1: 0.5, 2: 0, 3: 1}[s_len]
        yield (fixes, shape_penalty), parts, fixes


def _bh_candidate(text: str):
    if len(text) not in (9, 10):
        return None
    year = _coerce(text[0:2], "D")
    bh = _coerce(text[2:4], "L")
    number = _coerce(text[4:8], "D")
    series = _coerce(text[8:], "L")
    if None in (year, bh, number, series) or bh[0] != "BH":
        return None
    if set(series[0]) & _SERIES_FORBIDDEN or int(number[0]) == 0:
        return None
    fixes = year[1] + bh[1] + number[1] + series[1]
    return dict(state="BH", district=year[0], series=series[0], number=number[0]), fixes


def parse_plate(raw: str, max_corrections: int = 2) -> PlateResult:
    """Parse OCR text into a PlateResult. Never raises."""
    text = clean(raw or "")
    if not text:
        return PlateResult(raw=raw or "", kind="empty", reasons=("no characters read",))

    bh = _bh_candidate(text)
    if bh and bh[1] <= max_corrections:
        p, fixes = bh
        return PlateResult(
            raw=raw, kind="bh", corrections=fixes,
            normalized=f"{p['district']} BH {p['number']} {p['series']}", **p,
        )

    best = min(_standard_candidates(text), key=lambda c: c[0], default=None)
    if best and best[2] <= max_corrections:
        _, p, fixes = best
        normalized = " ".join(x for x in (p["state"], p["district"], p["series"], p["number"]) if x)
        return PlateResult(raw=raw, kind="standard", corrections=fixes, normalized=normalized, **p)

    reasons = []
    if len(text) < 6:
        reasons.append(f"too short ({len(text)} chars)")
    elif len(text) > 11:
        reasons.append(f"too long ({len(text)} chars)")
    if text[:2] not in STATE_CODES and not text[:2].isdigit():
        reasons.append(f"unknown state code '{text[:2]}'")
    if best and best[2] > max_corrections:
        reasons.append(f"needs {best[2]} OCR corrections (max {max_corrections})")
    if not reasons:
        reasons.append("does not match any registration format")
    return PlateResult(raw=raw, kind="nonstandard", reasons=tuple(reasons))
