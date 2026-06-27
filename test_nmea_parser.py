from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np

from nmea_parser import NMEAFrame, NMEAReader, frames_to_profile, parse_line
from sim_generator import format_gpgga


VALID_LINE = format_gpgga(12 * 3600 + 35 * 60 + 19.111, 545.4).strip()
INVALID_LINE = VALID_LINE[:-2] + "00"


def test_parse_line_valid_checksum() -> None:
    frame = parse_line(VALID_LINE)

    assert frame is not None
    assert frame.valid is True
    assert frame.timestamp_utc == "123519.111"
    assert frame.radar_alt_m == 545.4


def test_parse_line_invalid_checksum() -> None:
    frame = parse_line(INVALID_LINE)

    assert frame is not None
    assert frame.valid is False
    assert frame.radar_alt_m == 545.4


def test_frames_to_profile_interpolates_missing_values() -> None:
    frames = [
        NMEAFrame(timestamp_utc="1", radar_alt_m=100.0, raw="", valid=True),
        NMEAFrame(timestamp_utc="2", radar_alt_m=float("nan"), raw="", valid=True),
        NMEAFrame(timestamp_utc="3", radar_alt_m=120.0, raw="", valid=True),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        profile = frames_to_profile(frames, speed_mps=50.0, freq_hz=5.0)

    assert profile.shape == (3,)
    assert np.allclose(profile, np.array([100.0, 110.0, 120.0]))


def test_reader_read_window_reads_valid_frames(tmp_path: Path) -> None:
    nmea_path = tmp_path / "sample.nmea"
    nmea_path.write_text(
        "\n".join([INVALID_LINE, VALID_LINE, VALID_LINE]) + "\n",
        encoding="ascii",
    )

    reader = NMEAReader.from_file(nmea_path)
    try:
        frames = reader.read_window(2)
    finally:
        reader.close()

    assert len(frames) == 2
    assert all(frame.valid for frame in frames)
