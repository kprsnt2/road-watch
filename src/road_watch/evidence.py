"""Tamper-evident evidence packages.

A package is a JSON manifest stored next to the ORIGINAL, unedited video file:

  * SHA-256 of the video file,
  * the event(s) inside it (time offset, violation type, plate, GPS),
  * an Ed25519 signature over the canonical manifest, made with a per-device key.

Why a signature and not just a hash: anyone who edits the clip can recompute a
hash. A signature made at capture time with a key that never leaves the device
(Android Keystore in the app; a key file here) proves the manifest came from
that device and that neither manifest nor video changed since.

What it does NOT prove: that the device clock or GPS were honest. For that the
app should also send the manifest hash to a server (or an RFC 3161 timestamp
service) at capture time — see ``manifest_digest``.

The video is never re-encoded or trimmed: events point into the original file
by time offset, because any cut changes the bytes and breaks the hash.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

MANIFEST_VERSION = 1
MANIFEST_SUFFIX = ".evidence.json"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _canonical(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


# ---- device keys -----------------------------------------------------------

def generate_key(path: Path) -> Ed25519PublicKey:
    """Create a device key file. Refuses to overwrite an existing key."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; refusing to overwrite a device key")
    key = Ed25519PrivateKey.generate()
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return key.public_key()


def load_key(path: Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("device key must be Ed25519")
    return key


def public_key_b64(key: Ed25519PrivateKey | Ed25519PublicKey) -> str:
    pub = key.public_key() if isinstance(key, Ed25519PrivateKey) else key
    raw = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


# ---- sealing ---------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    offset_s: float            # seconds from start of the video
    violation: str             # e.g. "triple_riding", "no_helmet", "nonstandard_plate"
    confidence: float
    lat: float | None = None
    lon: float | None = None
    plate: str | None = None   # normalized plate text if read
    note: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != ""}


def seal(
    video: Path,
    events: list[Event],
    key: Ed25519PrivateKey,
    *,
    recorded_at: datetime | None = None,
    device: str = "",
    app_version: str = "",
) -> Path:
    """Write ``<video>.evidence.json`` and return its path."""
    video = Path(video)
    body = {
        "version": MANIFEST_VERSION,
        "video": {
            "name": video.name,
            "sha256": sha256_file(video),
            "bytes": video.stat().st_size,
        },
        "recorded_at": (recorded_at or datetime.now(timezone.utc)).isoformat(),
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "app_version": app_version,
        "public_key": public_key_b64(key),
        "events": [e.to_dict() for e in events],
    }
    signature = key.sign(_canonical(body))
    manifest = {"body": body, "signature": base64.b64encode(signature).decode()}
    out = video.with_name(video.name + MANIFEST_SUFFIX)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def manifest_digest(manifest_path: Path) -> str:
    """Short fingerprint to register with a server / timestamp service at capture time."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return hashlib.sha256(_canonical(manifest["body"])).hexdigest()


# ---- verification ----------------------------------------------------------

@dataclass(frozen=True)
class Verification:
    ok: bool
    problems: tuple[str, ...]


def verify(video: Path, trusted_public_key_b64: str | None = None) -> Verification:
    """Check the video against its manifest.

    Pass ``trusted_public_key_b64`` (from the device's registration) to make sure
    the manifest was signed by that device and not re-signed with another key.
    """
    video = Path(video)
    manifest_path = video.with_name(video.name + MANIFEST_SUFFIX)
    problems: list[str] = []
    if not manifest_path.exists():
        return Verification(False, ("no evidence manifest found",))

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        body, sig = manifest["body"], base64.b64decode(manifest["signature"])
        pub_b64 = body["public_key"]
    except (ValueError, KeyError, TypeError) as exc:
        return Verification(False, (f"manifest unreadable: {exc}",))

    if trusted_public_key_b64 is not None and pub_b64 != trusted_public_key_b64:
        problems.append("signed by a different device key than the trusted one")

    try:
        Ed25519PublicKey.from_public_bytes(base64.b64decode(pub_b64)).verify(sig, _canonical(body))
    except (InvalidSignature, ValueError):
        problems.append("signature does not match manifest (manifest was edited)")

    if not video.exists():
        problems.append("video file missing")
    else:
        if video.stat().st_size != body["video"]["bytes"]:
            problems.append("video size changed")
        if sha256_file(video) != body["video"]["sha256"]:
            problems.append("video content changed since sealing")

    return Verification(not problems, tuple(problems))
