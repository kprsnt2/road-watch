"""End-to-end scan over a synthetic video, with the detector replaced by a stub."""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
pytest.importorskip("onnxruntime")

from road_watch import __version__, scan as scan_mod  # noqa: E402
from road_watch.__main__ import main  # noqa: E402
from road_watch.evidence import generate_key, verify  # noqa: E402
from road_watch.rules import Box  # noqa: E402

START = datetime(2026, 11, 14, 10, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))


class TripleRidingEverywhere:
    """Stands in for YoloOnnx: every frame shows one motorcycle with three riders."""

    def __init__(self, *_, **__):
        pass

    def __call__(self, img):
        bike = Box(100, 200, 200, 280, "motorcycle", 0.9)
        riders = [Box(105 + 25 * i, 130, 135 + 25 * i, 256, "person", 0.9) for i in range(3)]
        return [bike, *riders]


def make_video(path, fps, seconds=2):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 320))
    assert w.isOpened()
    for _ in range(int(fps * seconds)):
        w.write(np.zeros((320, 320, 3), np.uint8))
    w.release()
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(scan_mod, "YoloOnnx", TripleRidingEverywhere)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"stub")
    return tmp_path, model


@pytest.mark.parametrize("fps", [30, 60])
def test_violation_found_at_any_frame_rate(env, fps):
    tmp, model = env
    video = make_video(tmp / "ride.avi", fps)
    recs = scan_mod.scan(video, model, tmp / "out", start=START)
    assert [r["violation"] for r in recs] == ["triple_riding"]
    assert recs[0]["offset_s"] == pytest.approx(0.4, abs=0.05)   # 3rd sample at 5 fps
    assert (tmp / "out" / "keyframes" / recs[0]["keyframe"]).is_file()
    lines = (tmp / "out" / "events.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == recs


def test_sealed_with_package_version(env):
    tmp, model = env
    video = make_video(tmp / "ride.avi", 30)
    generate_key(tmp / "device.key")
    scan_mod.scan(video, model, tmp / "out", start=START, key=tmp / "device.key")
    assert verify(video).ok
    body = json.loads((tmp / "ride.avi.evidence.json").read_text())["body"]
    assert body["app_version"] == __version__
    assert body["recorded_at"] == START.isoformat()


def test_default_start_is_modified_time_minus_duration(env):
    tmp, _ = env
    video = make_video(tmp / "ride.avi", 30, seconds=2)
    end = datetime(2026, 11, 14, 5, 0, tzinfo=timezone.utc)
    os.utime(video, (end.timestamp(), end.timestamp()))
    assert scan_mod.guess_start(video) == end - timedelta(seconds=2)


def test_bad_inputs_are_rejected_before_any_output(env):
    tmp, model = env
    video = make_video(tmp / "ride.avi", 30)
    with pytest.raises(ValueError, match="UTC offset"):
        scan_mod.scan(video, model, tmp / "out", start=datetime(2026, 11, 14, 10, 0))
    with pytest.raises(FileNotFoundError, match="model"):
        scan_mod.scan(video, tmp / "missing.onnx", tmp / "out", start=START)
    with pytest.raises(FileNotFoundError, match="device key"):
        scan_mod.scan(video, model, tmp / "out", start=START, key=tmp / "nope.key")
    assert not (tmp / "out").exists()


def test_cli_rejects_start_without_offset(env, capsys):
    tmp, model = env
    with pytest.raises(SystemExit) as exc:
        main(["scan", str(tmp / "ride.avi"), "--model", str(model), "--start", "2026-11-14T10:00:00"])
    assert exc.value.code == 2
    assert "no UTC offset" in capsys.readouterr().err


def test_cli_reports_missing_video_without_traceback(env, capsys):
    tmp, model = env
    assert main(["scan", str(tmp / "nope.mp4"), "--model", str(model), "--out", str(tmp / "out")]) == 2
    assert "video not found" in capsys.readouterr().err


def test_cli_keygen_refuses_overwrite(tmp_path, capsys):
    key = tmp_path / "device.key"
    assert main(["keygen", str(key)]) == 0
    before = key.read_bytes()
    assert main(["keygen", str(key)]) == 2
    assert key.read_bytes() == before
