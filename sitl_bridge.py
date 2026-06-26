"""ArduPilot SITL bridge for TERRAIN NAVIGATOR.

This module is intentionally scoped to "person 1" responsibilities:

- connect to ArduPilot SITL over MAVLink;
- read telemetry;
- derive a unified sample stream;
- emulate GNSS on/off transitions;
- expose helpers compatible with the current NMEA-based pipeline.
"""

from __future__ import annotations

import argparse
import logging
import math
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Protocol

from constants import FIXED_BARO_ALTITUDE_M
from dem_loader import DEMLoader
from nmea_parser import NMEAFrame, parse_line
from sim_generator import format_gpgga


LOGGER = logging.getLogger(__name__)
UINT16_MAX = 65535


class MAVLinkConnection(Protocol):
    """Small protocol surface needed from a MAVLink connection."""

    def wait_heartbeat(self, timeout: float | None = None) -> Any:
        """Wait for an autopilot heartbeat."""

    def recv_match(self, type: str | list[str] | None = None, blocking: bool = False, timeout: float | None = None) -> Any:
        """Read one MAVLink message."""

    def close(self) -> None:
        """Close the connection."""


@dataclass(frozen=True)
class SITLSample:
    """Unified sample produced from SITL telemetry."""

    timestamp: float
    lat: float
    lon: float
    alt_msl: float
    heading_deg: float
    ground_speed_mps: float
    radar_alt_m: float
    gnss_available: bool

    def as_dict(self) -> dict[str, float | bool]:
        """Return the sample in the agreed unified dictionary format."""

        return dict(asdict(self))


class SITLBridge:
    """Read ArduPilot SITL telemetry and expose unified terrain-nav samples."""

    def __init__(
        self,
        connection_string: str,
        dem_loader: DEMLoader,
        *,
        gnss_drop_after_s: float | None = None,
        gnss_recover_after_s: float | None = None,
        poll_timeout_s: float = 1.0,
        mock_messages: Iterable[Any] | None = None,
        time_fn: Callable[[], float] | None = None,
        mavlink_factory: Callable[[str], MAVLinkConnection] | None = None,
    ) -> None:
        self.connection_string = connection_string
        self.dem_loader = dem_loader
        self.gnss_drop_after_s = gnss_drop_after_s
        self.gnss_recover_after_s = gnss_recover_after_s
        self.poll_timeout_s = poll_timeout_s
        self._mock_iter = iter(mock_messages) if mock_messages is not None else None
        self._time_fn = time_fn or time.monotonic
        self._mavlink_factory = mavlink_factory or _default_mavlink_factory
        self._connection: MAVLinkConnection | None = None
        self._heartbeat_seen = self._mock_iter is not None
        self._start_time_s: float | None = None
        self._state: dict[str, float] = {}
        self._gnss_override: bool | None = None

    def connect(self) -> None:
        """Open the MAVLink connection and wait for the first heartbeat."""

        if self._mock_iter is not None:
            LOGGER.info("SITL bridge running in mock mode")
            return

        self._connection = self._mavlink_factory(self.connection_string)
        self._connection.wait_heartbeat(timeout=self.poll_timeout_s)
        self._heartbeat_seen = True
        LOGGER.info("Connected to SITL via %s", self.connection_string)

    def close(self) -> None:
        """Close the bridge connection."""

        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def samples(self) -> Iterator[SITLSample]:
        """Yield samples continuously until the source is exhausted."""

        while True:
            sample = self.read_sample()
            if sample is None:
                return
            yield sample

    def read_sample(self) -> SITLSample | None:
        """Read one telemetry update and convert it into a unified sample."""

        if not self._heartbeat_seen:
            self.connect()

        while True:
            message = self._recv_message()
            if message is None:
                return None if self._mock_iter is not None else None

            self._update_state(message)
            sample = self._build_sample(message)
            if sample is not None:
                return sample

    def sample_to_nmea_sentence(self, sample: SITLSample) -> str:
        """Convert a sample into the existing radar-altimeter NMEA format."""

        return format_gpgga(sample.timestamp, sample.radar_alt_m)

    def sample_to_nmea_frame(self, sample: SITLSample) -> NMEAFrame:
        """Convert a sample into an NMEAFrame compatible with the current pipeline."""

        frame = parse_line(self.sample_to_nmea_sentence(sample))
        if frame is None:
            raise ValueError("Generated NMEA sentence could not be parsed")
        return frame

    def set_gnss_enabled(self, enabled: bool) -> None:
        """Force GNSS availability on or off until cleared or toggled again."""

        self._gnss_override = bool(enabled)

    def toggle_gnss(self) -> bool:
        """Toggle the current GNSS availability override and return the new state."""

        new_state = not self.current_gnss_enabled
        self._gnss_override = new_state
        return new_state

    def clear_gnss_override(self) -> None:
        """Return GNSS availability control back to timer/default logic."""

        self._gnss_override = None

    @property
    def current_gnss_enabled(self) -> bool:
        """Expose the current externally visible GNSS state."""

        if self._gnss_override is not None:
            return self._gnss_override
        if self._start_time_s is None:
            return True
        return self._is_gnss_available(self._time_fn())

    def _recv_message(self) -> Any | None:
        if self._mock_iter is not None:
            try:
                return next(self._mock_iter)
            except StopIteration:
                return None

        assert self._connection is not None
        return self._connection.recv_match(
            type=["GLOBAL_POSITION_INT", "VFR_HUD", "ATTITUDE"],
            blocking=True,
            timeout=self.poll_timeout_s,
        )

    def _update_state(self, message: Any) -> None:
        message_type = _message_type(message)
        if message_type == "GLOBAL_POSITION_INT":
            lat = getattr(message, "lat", None)
            lon = getattr(message, "lon", None)
            alt = getattr(message, "alt", None)
            vx = getattr(message, "vx", None)
            vy = getattr(message, "vy", None)
            heading = getattr(message, "hdg", None)
            if lat is not None and lon is not None:
                self._state["lat"] = float(lat) / 1e7
                self._state["lon"] = float(lon) / 1e7
            if alt is not None:
                self._state["alt_msl"] = float(alt) / 1000.0
            if vx is not None and vy is not None:
                vx_mps = float(vx) / 100.0
                vy_mps = float(vy) / 100.0
                self._state["ground_speed_mps"] = math.hypot(vx_mps, vy_mps)
            if heading is not None and int(heading) != UINT16_MAX:
                self._state["heading_deg"] = (float(heading) / 100.0) % 360.0
            if getattr(message, "time_boot_ms", None) is not None:
                self._state["timestamp"] = float(message.time_boot_ms) / 1000.0
            return

        if message_type == "VFR_HUD":
            groundspeed = getattr(message, "groundspeed", None)
            heading = getattr(message, "heading", None)
            if groundspeed is not None:
                self._state["ground_speed_mps"] = float(groundspeed)
            if heading is not None:
                self._state["heading_deg"] = float(heading) % 360.0
            return

        if message_type == "ATTITUDE":
            yaw = getattr(message, "yaw", None)
            if yaw is not None and "heading_deg" not in self._state:
                self._state["heading_deg"] = math.degrees(float(yaw)) % 360.0

    def _build_sample(self, message: Any) -> SITLSample | None:
        del message
        required = ("lat", "lon")
        if any(key not in self._state for key in required):
            return None

        current_time_s = self._time_fn()
        if self._start_time_s is None:
            self._start_time_s = current_time_s

        lat = self._state["lat"]
        lon = self._state["lon"]
        alt_msl = FIXED_BARO_ALTITUDE_M
        heading_deg = self._state.get("heading_deg", 0.0)
        speed_mps = self._state.get("ground_speed_mps", 0.0)
        timestamp = self._state.get("timestamp", current_time_s)

        try:
            terrain_elevation_m = self.dem_loader.get_elevation(lat, lon)
            radar_alt_m = max(0.0, FIXED_BARO_ALTITUDE_M - terrain_elevation_m)
        except ValueError:
            LOGGER.warning("SITL sample is outside DEM bounds: lat=%.6f lon=%.6f", lat, lon)
            radar_alt_m = float("nan")

        return SITLSample(
            timestamp=timestamp,
            lat=lat,
            lon=lon,
            alt_msl=alt_msl,
            heading_deg=heading_deg,
            ground_speed_mps=speed_mps,
            radar_alt_m=radar_alt_m,
            gnss_available=self._is_gnss_available(current_time_s),
        )

    def _is_gnss_available(self, current_time_s: float) -> bool:
        if self._gnss_override is not None:
            return self._gnss_override
        if self._start_time_s is None:
            return True

        elapsed_s = current_time_s - self._start_time_s
        if self.gnss_drop_after_s is None:
            return True
        if elapsed_s < self.gnss_drop_after_s:
            return True
        if self.gnss_recover_after_s is None:
            return False
        return elapsed_s >= self.gnss_recover_after_s


def _message_type(message: Any) -> str:
    get_type = getattr(message, "get_type", None)
    if callable(get_type):
        return str(get_type())
    return type(message).__name__


def _default_mavlink_factory(connection_string: str) -> MAVLinkConnection:
    try:
        from pymavlink import mavutil
    except ImportError as exc:
        raise ImportError(
            "pymavlink is required for real SITL connections. Install it with: "
            "python -m pip install pymavlink"
        ) from exc
    return mavutil.mavlink_connection(connection_string)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build a small smoke-test CLI for the SITL bridge."""

    parser = argparse.ArgumentParser(description="Smoke-test the TERRAIN NAVIGATOR SITL bridge")
    parser.add_argument("--connect", default="udp:127.0.0.1:14550", help="MAVLink connection string")
    parser.add_argument("--dem", required=True, type=Path, help="Path to DEM GeoTIFF")
    parser.add_argument("--count", type=int, default=5, help="Number of samples to print")
    parser.add_argument("--stream-nmea-udp", action="store_true", help="Stream synthesized radar-altimeter GPGGA messages over UDP")
    parser.add_argument("--udp-host", default="127.0.0.1", help="UDP host for --stream-nmea-udp")
    parser.add_argument("--udp-port", type=int, default=10110, help="UDP port for --stream-nmea-udp")
    parser.add_argument("--stream-rate-hz", type=float, help="Optional maximum NMEA output rate in Hz")
    parser.add_argument("--gnss-off", action="store_true", help="Force GNSS unavailable for this smoke test")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def _stream_nmea_udp(
    bridge: SITLBridge,
    *,
    host: str,
    port: int,
    max_rate_hz: float | None,
    count: int,
) -> None:
    """Stream synthesized GPGGA sentences over UDP."""

    min_period_s = 0.0 if max_rate_hz is None else 1.0 / max(max_rate_hz, 1e-6)
    last_sent_at = 0.0
    sent_count = 0

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for sample in bridge.samples():
            now = time.monotonic()
            if min_period_s > 0.0 and (now - last_sent_at) < min_period_s:
                continue
            sentence = bridge.sample_to_nmea_sentence(sample)
            sock.sendto(sentence.encode("ascii"), (host, port))
            LOGGER.info("Sent NMEA to udp://%s:%d | %s", host, port, sentence.strip())
            last_sent_at = now
            sent_count += 1
            if count > 0 and sent_count >= count:
                break


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for quick SITL bridge verification."""

    args = build_argument_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    with DEMLoader(args.dem) as dem:
        bridge = SITLBridge(args.connect, dem)
        if args.gnss_off:
            bridge.set_gnss_enabled(False)
        bridge.connect()
        try:
            if args.stream_nmea_udp:
                _stream_nmea_udp(
                    bridge,
                    host=args.udp_host,
                    port=args.udp_port,
                    max_rate_hz=args.stream_rate_hz,
                    count=args.count,
                )
            else:
                for index, sample in enumerate(bridge.samples(), start=1):
                    LOGGER.info("SITL sample %d: %s", index, sample.as_dict())
                    if index >= args.count:
                        break
        finally:
            bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
