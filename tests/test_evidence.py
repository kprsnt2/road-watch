import json
import os

import pytest

from road_watch.evidence import (
    Event,
    generate_key,
    load_key,
    manifest_digest,
    public_key_b64,
    seal,
    verify,
)


@pytest.fixture
def setup(tmp_path):
    video = tmp_path / "ride.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 100)
    key_path = tmp_path / "device.key"
    generate_key(key_path)
    key = load_key(key_path)
    events = [Event(12.4, "triple_riding", 0.91, 17.4065, 78.4772, "TG 09 EA 1234")]
    manifest = seal(video, events, key, device="test-phone", app_version="0.1.0")
    return video, manifest, key, key_path


def test_untouched_video_verifies(setup):
    video, _, key, _ = setup
    result = verify(video, trusted_public_key_b64=public_key_b64(key))
    assert result.ok, result.problems


def test_edited_video_fails(setup):
    video, *_ = setup
    data = bytearray(video.read_bytes())
    data[500] ^= 0xFF  # flip one byte, same size
    video.write_bytes(bytes(data))
    result = verify(video)
    assert not result.ok
    assert "video content changed since sealing" in result.problems


def test_edited_manifest_fails(setup):
    video, manifest, *_ = setup
    m = json.loads(manifest.read_text())
    m["body"]["events"][0]["plate"] = "TG 09 ZZ 9999"  # frame someone else
    manifest.write_text(json.dumps(m))
    result = verify(video)
    assert not result.ok
    assert any("signature" in p for p in result.problems)


def test_resigned_with_another_key_is_caught(setup, tmp_path):
    video, _, original_key, _ = setup
    other_path = tmp_path / "attacker.key"
    generate_key(other_path)
    # Attacker edits the video then re-seals it with their own key.
    video.write_bytes(video.read_bytes() + b"extra")
    seal(video, [Event(1.0, "no_helmet", 0.8)], load_key(other_path))
    assert verify(video).ok  # self-consistent...
    result = verify(video, trusted_public_key_b64=public_key_b64(original_key))
    assert not result.ok   # ...but not from the registered device
    assert any("different device key" in p for p in result.problems)


def test_missing_manifest(tmp_path):
    v = tmp_path / "x.mp4"
    v.write_bytes(b"abc")
    assert not verify(v).ok


def test_key_is_never_overwritten(setup):
    *_, key_path = setup
    with pytest.raises(FileExistsError):
        generate_key(key_path)


def test_digest_is_stable(setup):
    _, manifest, *_ = setup
    assert manifest_digest(manifest) == manifest_digest(manifest)
    assert len(manifest_digest(manifest)) == 64


def test_trusted_key_comparison_ignores_whitespace(setup):
    video, _, key, _ = setup
    assert verify(video, trusted_public_key_b64=public_key_b64(key) + "\n").ok


def test_verification_reports_signer(setup):
    video, _, key, _ = setup
    assert verify(video).signer == public_key_b64(key)


@pytest.mark.parametrize("drop", ["video", "public_key"])
def test_incomplete_manifest_fails_cleanly(setup, drop):
    video, manifest, *_ = setup
    m = json.loads(manifest.read_text())
    del m["body"][drop]
    manifest.write_text(json.dumps(m))
    result = verify(video)
    assert not result.ok
    assert "manifest unreadable" in result.problems[0]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_key_file_is_owner_only(setup):
    *_, key_path = setup
    assert key_path.stat().st_mode & 0o077 == 0
