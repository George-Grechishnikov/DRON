"""Quick correlation benchmark for desktop, Jetson, or Raspberry Pi runs."""

from __future__ import annotations

import argparse
import time

import numpy as np

from correlator import Correlator


def _make_reference_matrix(azimuth_count: int, length: int, offset_steps: int, seed: int) -> np.ndarray:
    total_length = length + offset_steps
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size=(azimuth_count, total_length))
    kernel = np.array([0.2, 0.6, 0.2], dtype=float)
    smoothed = np.array([np.convolve(row, kernel, mode="same") for row in base], dtype=float)
    trend = np.linspace(-0.5, 0.5, total_length)
    for azimuth in range(azimuth_count):
        smoothed[azimuth] += trend * (azimuth / max(azimuth_count - 1, 1))
    return smoothed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark TERRAIN NAVIGATOR correlation performance")
    parser.add_argument("--azimuths", type=int, default=360, help="Number of azimuth hypotheses")
    parser.add_argument("--length", type=int, default=128, help="Measured profile length")
    parser.add_argument("--offset-steps", type=int, default=67, help="Max offset in samples")
    parser.add_argument("--iterations", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--feature-refine-top-k", type=int, default=0, help="Enable top-k terrain feature refinement")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    ref_matrix = _make_reference_matrix(args.azimuths, args.length, args.offset_steps, args.seed)
    h_meas = ref_matrix[args.azimuths // 3, 25 : 25 + args.length]
    azimuth_axis = np.arange(args.azimuths, dtype=float)
    correlator = Correlator(
        profile_length_m=float(args.length * 30.0),
        step_m=30.0,
        max_offset_m=float(args.offset_steps * 30.0),
        feature_refine_top_k=args.feature_refine_top_k,
    )

    timings: list[float] = []
    best_result = None
    for _ in range(args.iterations):
        started_at = time.perf_counter()
        best_result = correlator.compute(h_meas, ref_matrix, azimuths_deg=azimuth_axis)
        timings.append(time.perf_counter() - started_at)

    assert best_result is not None
    timings_array = np.asarray(timings, dtype=float)
    print(f"iterations={args.iterations}")
    print(f"mean_s={timings_array.mean():.6f}")
    print(f"p95_s={np.percentile(timings_array, 95):.6f}")
    print(f"best_azimuth_deg={best_result.best_azimuth_deg:.1f}")
    print(f"best_offset_steps={best_result.best_offset_steps}")
    print(f"pslr_db={best_result.pslr_db:.3f}")
    print(f"ambiguous={best_result.is_ambiguous}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
