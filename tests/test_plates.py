import pytest

from road_watch.plates import parse_plate


@pytest.mark.parametrize(
    "raw, normalized",
    [
        ("TG 09 EA 1234", "TG 09 EA 1234"),
        ("ts07fk0420", "TS 07 FK 0420"),
        ("TS-09-E-4567", "TS 09 E 4567"),
        ("IND TG09EA1234", "TG 09 EA 1234"),       # HSRP marker read as prefix
        ("KA 01 AB 12", "KA 01 AB 0012"),           # short numbers are zero-padded
        ("DL 3C AB 1234", "DL 03C AB 1234"),        # Delhi category letter
        ("MH 12 1234", "MH 12 1234"),               # old plate, no series
        ("22 BH 1234 AA", "22 BH 1234 AA"),         # Bharat series
    ],
)
def test_valid_plates(raw, normalized):
    r = parse_plate(raw)
    assert r.valid, r.reasons
    assert r.normalized == normalized
    assert r.corrections == 0


@pytest.mark.parametrize(
    "raw, normalized",
    [
        ("T609EA1234", "TG 09 EA 1234"),   # 6 read for G in the state code
        ("TG0SEA1234", "TG 05 EA 1234"),   # S read for 5 in district
        ("TG09EA12B4", "TG 09 EA 1284"),   # B read for 8 in number
    ],
)
def test_ocr_confusions_are_corrected(raw, normalized):
    r = parse_plate(raw)
    assert r.valid
    assert r.normalized == normalized
    assert r.corrections == 1


def test_forbidden_series_letters_are_not_invented():
    # "E0" in the series must not become "EO" (O is never issued as a series letter).
    r = parse_plate("TS09E01234")
    assert not (r.valid and "O" in r.series)


@pytest.mark.parametrize(
    "raw",
    ["BOSS", "JAI TELANGANA", "XX 09 EA 1234", "TG 09 EA 0000", "", "   ", "TG09EA123456789"],
)
def test_nonstandard_plates_are_flagged(raw):
    r = parse_plate(raw)
    assert not r.valid
    assert r.kind in ("nonstandard", "empty")
    assert r.reasons


def test_too_many_corrections_rejected():
    # Every character would need fixing — that's guessing, not reading.
    assert not parse_plate("7609EA12B4", max_corrections=1).valid


def test_telangana_detection():
    assert parse_plate("TS 09 EA 1234").is_telangana
    assert parse_plate("TG 09 EA 1234").is_telangana
    assert not parse_plate("AP 09 EA 1234").is_telangana
