from datetime import datetime, timedelta, timezone

import pytest

from road_watch.geo import (
    Fix,
    aggregate,
    cell_bounds,
    cell_id,
    hotspot_event,
    load_gpx,
    position_at,
)

T0 = datetime(2026, 11, 14, 4, 30, tzinfo=timezone.utc)  # 10:00 IST, Saturday

GPX = """<?xml version="1.0"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
 <trk><trkseg>
  <trkpt lat="17.4400" lon="78.3489"><time>2026-11-14T04:30:10Z</time></trkpt>
  <trkpt lat="17.4300" lon="78.3489"><time>2026-11-14T04:30:00Z</time></trkpt>
  <trkpt lat="17.4500" lon="78.3489"><time>2026-11-14T04:30:20Z</time></trkpt>
  <trkpt lat="17.9" lon="78.9"></trkpt>
 </trkseg></trk>
</gpx>"""


@pytest.fixture
def fixes(tmp_path):
    p = tmp_path / "ride.gpx"
    p.write_text(GPX)
    return load_gpx(p)


def test_gpx_sorted_and_untimed_points_skipped(fixes):
    assert [f.lat for f in fixes] == [17.43, 17.44, 17.45]


def test_interpolation(fixes):
    lat, lon = position_at(fixes, T0 + timedelta(seconds=5))
    assert lat == pytest.approx(17.435)
    assert lon == pytest.approx(78.3489)


def test_exact_and_out_of_range(fixes):
    assert position_at(fixes, T0) == (17.43, 78.3489)
    assert position_at(fixes, T0 - timedelta(seconds=1)) is None
    assert position_at(fixes, T0 + timedelta(minutes=5)) is None


def test_large_gap_refused():
    track = [Fix(T0, 17.0, 78.0), Fix(T0 + timedelta(minutes=5), 17.1, 78.1)]
    assert position_at(track, T0 + timedelta(minutes=2)) is None


def test_hotspot_event_has_no_exact_location():
    e = hotspot_event("no_helmet", 17.4401, 78.3489, T0)
    assert set(e) == {"cell", "violation", "hour_of_week", "day"}
    assert e["hour_of_week"] == 5 * 24 + 10   # Saturday 10:00 IST
    # Two points ~10 m apart fall in the same cell.
    assert hotspot_event("no_helmet", 17.44015, 78.34895, T0)["cell"] == e["cell"]


def test_cell_contains_point_and_is_street_sized():
    lat, lon = 17.4065, 78.4772
    s, w, n, east = cell_bounds(cell_id(lat, lon))
    assert s <= lat < n and w <= lon < east
    assert 250 < (n - s) * 111_000 < 300   # metres north-south


def test_bad_coordinates_rejected():
    with pytest.raises(ValueError):
        cell_id(117.4, 78.4)


def test_aggregate_drops_sparse_cells():
    a = hotspot_event("no_helmet", 17.44, 78.35, T0)
    b = hotspot_event("triple_riding", 17.36, 78.47, T0)
    agg = aggregate([a, a, dict(a, violation="triple_riding"), b], min_count=3)
    assert agg == {a["cell"]: {"no_helmet": 2, "triple_riding": 1}}


def test_malformed_points_are_skipped(tmp_path):
    p = tmp_path / "bad.gpx"
    p.write_text("""<gpx><trk><trkseg>
      <trkpt lat="17.43" lon="78.34"><time>not a time</time></trkpt>
      <trkpt lon="78.34"><time>2026-11-14T04:30:00Z</time></trkpt>
      <trkpt lat="917.43" lon="78.34"><time>2026-11-14T04:30:05Z</time></trkpt>
      <trkpt lat="17.44" lon="78.34"><time>2026-11-14T04:30:10Z</time></trkpt>
    </trkseg></trk></gpx>""")
    assert [f.lat for f in load_gpx(p)] == [17.44]
