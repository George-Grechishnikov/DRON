"""NMEA-0183 parser for TERRAIN NAVIGATOR.

Example:
    reader = NMEAReader.from_file("output/traj1.nmea")
    frames = reader.read_window(50)
    profile = frames_to_profile(frames, speed_mps=50.0, freq_hz=5.0)
    # profile.shape == (50,)
"""

from __future__ import annotations

import logging
import socket
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, TextIO

import numpy as np


LOGGER = logging.getLogger(__name__)
VALID_SENTENCE_TYPES = {"GPGGA", "GNGGA"}
ROLLING_INVALID_WINDOW = 100
INVALID_WARNING_THRESHOLD = 0.05


@dataclass(frozen=True)
class NMEAFrame:
    """Parsed radar-altimeter frame."""

    timestamp_utc: str
    radar_alt_m: float
    raw: str
    valid: bool


def nmea_checksum(sentence: str) -> str:
    """Return the XOR checksum for an NMEA payload."""

    checksum = 0
    for char in sentence:
        checksum ^= ord(char)
    return f"{checksum:02X}"


def parse_line(line: str) -> Optional[NMEAFrame]:
    """Parse one NMEA line into an NMEAFrame.

    Returns None for unsupported sentence types. For checksum mismatch,
    returns a frame with valid=False instead of raising.
    """

    raw = line.strip()
    if not raw or not raw.startswith("$") or "*" not in raw:
        return None

    payload, checksum = raw[1:].split("*", 1)
    fields = payload.split(",")
    if not fields or fields[0] not in VALID_SENTENCE_TYPES:
        return None

    timestamp = fields[1] if len(fields) > 1 else ""
    radar_alt_token = fields[9] if len(fields) > 9 else ""
    unit_token = fields[10] if len(fields) > 10 else ""
    is_valid = nmea_checksum(payload) == checksum.upper()

    if unit_token and unit_token != "M":
        is_valid = False

    radar_alt_m = float(radar_alt_token) if radar_alt_token else float("nan")
    return NMEAFrame(
        timestamp_utc=timestamp,
        radar_alt_m=radar_alt_m,
        raw=raw,
        valid=is_valid,
    )


class NMEAReader:
    """NMEA source reader for file and UDP modes."""

    def __init__(
        self,
        *,
        file_handle: TextIO | None = None,
        sock: socket.socket | None = None,
        source_name: str,
    ) -> None:
        self._file_handle = file_handle
        self._socket = sock
        self._source_name = source_name
        self._udp_buffer = ""
        self._recent_validity: deque[bool] = deque(maxlen=ROLLING_INVALID_WINDOW)

    @classmethod
    def from_file(cls, path: str | Path) -> "NMEAReader":
        """Create a reader from a .nmea file."""

        file_handle = Path(path).open("r", encoding="ascii", newline="")
        return cls(file_handle=file_handle, source_name=str(path))

    @classmethod
    def from_udp(cls, host: str, port: int) -> "NMEAReader":
        """Create a reader from a non-blocking UDP socket."""

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((host, port))
        sock.setblocking(False)
        return cls(sock=sock, source_name=f"udp://{host}:{port}")

    def __iter__(self) -> Iterator[NMEAFrame]:
        """Yield parsed frames one by one."""

        if self._file_handle is not None:
            yield from self._iter_file()
            return

        if self._socket is not None:
            yield from self._iter_udp()
            return

        raise RuntimeError("Reader has no source configured")

    def close(self) -> None:
        """Close the underlying source."""

        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def read_window(self, n: int) -> list[NMEAFrame]:
        """Read up to n valid frames from the source."""

        if n <= 0:
            return []

        frames: list[NMEAFrame] = []
        iterator = iter(self)
        while len(frames) < n:
            try:
                frame = next(iterator)
            except StopIteration:
                break
            if frame.valid:
                frames.append(frame)
        return frames

    def _iter_file(self) -> Iterator[NMEAFrame]:
        assert self._file_handle is not None
        for line in self._file_handle:
            frame = parse_line(line)
            if frame is None:
                continue
            self._track_validity(frame.valid)
            yield frame

    def _iter_udp(self) -> Iterator[NMEAFrame]:
        assert self._socket is not None
        while True:
            try:
                packet, _ = self._socket.recvfrom(4096)
            except BlockingIOError:
                return

            self._udp_buffer += packet.decode("ascii", errors="ignore")
            lines = self._udp_buffer.splitlines(keepends=True)
            if lines and not lines[-1].endswith(("\n", "\r")):
                self._udp_buffer = lines.pop()
            else:
                self._udp_buffer = ""

            for line in lines:
                frame = parse_line(line)
                if frame is None:
                    continue
                self._track_validity(frame.valid)
                yield frame

    def _track_validity(self, is_valid: bool) -> None:
        self._recent_validity.append(is_valid)
        if len(self._recent_validity) < self._recent_validity.maxlen:
            return

        invalid_ratio = 1.0 - (sum(self._recent_validity) / len(self._recent_validity))
        if invalid_ratio > INVALID_WARNING_THRESHOLD:
            LOGGER.warning(
                "Invalid NMEA ratio over last %d frames from %s is %.1f%%",
                len(self._recent_validity),
                self._source_name,
                invalid_ratio * 100.0,
            )


def _interpolate_missing(values: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values in a 1D array."""

    if values.ndim != 1:
        raise ValueError("Interpolation expects a 1D array")
    if values.size == 0:
        return values

    result = values.astype(float, copy=True)
    mask = np.isnan(result)
    if not mask.any():
        return result

    valid_idx = np.flatnonzero(~mask)
    if valid_idx.size == 0:
        raise ValueError("Cannot interpolate profile with no valid altitude values")

    missing_idx = np.flatnonzero(mask)
    result[missing_idx] = np.interp(missing_idx, valid_idx, result[valid_idx])
    return result


def frames_to_profile(
    frames: list[NMEAFrame], speed_mps: float, freq_hz: float
) -> np.ndarray:
    """Convert frames into a 1D terrain profile array."""

    if speed_mps <= 0:
        raise ValueError("speed_mps must be positive")
    if freq_hz <= 0:
        raise ValueError("freq_hz must be positive")

    if not frames:
        return np.empty((0,), dtype=float)

    altitudes = np.array([frame.radar_alt_m for frame in frames], dtype=float)
    profile = _interpolate_missing(altitudes)
    return profile

