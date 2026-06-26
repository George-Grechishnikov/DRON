from __future__ import annotations

from benchmark_correlator import main


def test_benchmark_correlator_main_runs(capsys) -> None:
    exit_code = main(["--iterations", "1", "--azimuths", "32", "--length", "32", "--offset-steps", "8"])

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "mean_s=" in captured
    assert "best_azimuth_deg=" in captured
