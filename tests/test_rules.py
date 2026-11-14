from road_watch.rules import Box, ViolationTracker, assign_riders, rider_score


def bike(x=100, y=200, w=100, h=80):
    return Box(x, y, x + w, y + h, "motorcycle", 0.9)


def rider(x, bike_box: Box, cls="person"):
    # A seated rider: top well above the bike, legs ending in the bike's lower half.
    return Box(x, bike_box.y1 - 70, x + 30, bike_box.y1 + 0.7 * bike_box.h, cls, 0.9)


def no_helmet_head(r: Box):
    return Box(r.x1 + 5, r.y1, r.x2 - 5, r.y1 + 20, "head", 0.8)


def frame_with(n_riders=1, helmetless=0, b=None):
    b = b or bike()
    riders = [rider(b.x1 + 5 + 25 * i, b) for i in range(n_riders)]
    heads = [no_helmet_head(r) for r in riders[:helmetless]]
    return [b, *riders, *heads]


def test_rider_on_bike_is_matched():
    b = bike()
    assert rider_score(rider(110, b), b) > 0


def test_pedestrian_beside_bike_is_not_a_rider():
    b = bike()
    beside = Box(b.x2 + 10, b.y1 - 70, b.x2 + 40, b.y2, "person")
    assert rider_score(beside, b) == 0


def test_tall_pedestrian_in_front_is_not_a_rider():
    b = bike()
    close = Box(b.x1 + 10, b.y1 - 300, b.x1 + 60, b.y2, "person")  # very tall = near the camera
    assert rider_score(close, b) == 0


def test_person_assigned_to_only_one_bike():
    b1, b2 = bike(100), bike(180)
    p = rider(185, b2)
    assigned = assign_riders([b1, b2], [p])
    assert sum(len(v) for v in assigned.values()) == 1
    assert assigned[1] == [p]


def run(frames, tracker=None):
    t = tracker or ViolationTracker(window=5, min_hits=3)
    out = []
    for i, boxes in enumerate(frames):
        out += t.update(i, i / 10, boxes)
    return out


def test_triple_riding_confirmed_after_persistence():
    events = run([frame_with(3)] * 5)
    triple = [e for e in events if e.violation == "triple_riding"]
    assert len(triple) == 1                 # fires once, not every frame
    assert triple[0].frame == 2             # on the 3rd consistent frame
    assert triple[0].riders == 3


def test_single_glitch_frame_does_not_fire():
    frames = [frame_with(2), frame_with(3), frame_with(2), frame_with(2), frame_with(2)]
    assert not [e for e in run(frames) if e.violation == "triple_riding"]


def test_two_riders_never_triple():
    assert not [e for e in run([frame_with(2)] * 10) if e.violation == "triple_riding"]


def test_no_helmet_detected():
    events = run([frame_with(2, helmetless=1)] * 4)
    assert [e.violation for e in events] == ["no_helmet"]


def test_helmeted_riders_ok():
    assert run([frame_with(2, helmetless=0)] * 10) == []


def test_moving_bike_keeps_its_track():
    frames = [frame_with(3, b=bike(x=100 + 8 * i)) for i in range(6)]
    events = run(frames)
    assert len({e.track_id for e in events}) == 1


def test_two_bikes_tracked_separately():
    frames = [frame_with(3, b=bike(50)) + frame_with(1, helmetless=1, b=bike(400)) for _ in range(4)]
    events = run(frames)
    assert {e.violation for e in events} == {"triple_riding", "no_helmet"}
    assert len({e.track_id for e in events}) == 2
