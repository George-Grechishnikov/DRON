"""User-facing runner for the terrain-navigation case workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from case_reader import CaseInputConfig, iter_case_unified_samples
from main import _path_from_config, load_yaml_config, main as run_main


def _report_paths(report_path: Path) -> dict[str, Path]:
    suffix = report_path.suffix or ".html"
    stem = report_path.name[: -len(suffix)] if report_path.name.endswith(suffix) else report_path.stem
    return {
        "html": report_path,
        "summary_txt": report_path.with_name(f"{stem}.summary.txt"),
        "summary_json": report_path.with_name(f"{stem}.summary.json"),
        "records_csv": report_path.with_name(f"{stem}.records.csv"),
    }


def _validate_case_inputs(config_path: Path) -> tuple[dict[str, Path], Path, dict]:
    payload = load_yaml_config(config_path)
    required = {
        "dem_path": _path_from_config(payload, "dem_path", config_path),
        "radar_data_path": _path_from_config(payload, "radar_data_path", config_path),
        "truth_path": _path_from_config(payload, "truth_path", config_path),
        "barometer_path": _path_from_config(payload, "barometer_path", config_path),
    }
    missing = [name for name, path in required.items() if path is None or not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required case input(s): " + ", ".join(missing)
        )
    visualization = payload.get("visualization", {}) if isinstance(payload.get("visualization"), dict) else {}
    report_path_value = visualization.get("export_report_path", "output/terrain_navigator_report.html")
    report_path = Path(str(report_path_value))
    return {key: Path(value) for key, value in required.items() if value is not None}, report_path, payload


def _fully_validate_case_payload(payload: dict, resolved_inputs: dict[str, Path]) -> int:
    frequency_hz = float(payload.get("sample_rate_hz", payload.get("frequency_hz", 0.0)))
    if frequency_hz <= 0:
        raise ValueError("config.yaml must define a positive sample_rate_hz or frequency_hz")
    gnss_drop_after = payload.get("gnss_drop_after_s")
    case_config = CaseInputConfig(
        dem_path=resolved_inputs["dem_path"],
        radar_data_path=resolved_inputs["radar_data_path"],
        truth_path=resolved_inputs["truth_path"],
        barometer_path=resolved_inputs["barometer_path"],
        sample_rate_hz=frequency_hz,
        gnss_drop_after_s=None if gnss_drop_after is None else float(gnss_drop_after),
    )
    sample_count = sum(1 for _ in iter_case_unified_samples(case_config))
    if sample_count <= 0:
        raise ValueError("case input set is empty after parsing")
    return sample_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the TERRAIN NAVIGATOR case workflow")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("input") / "incoming" / "config.yaml",
        help="Path to case config.yaml",
    )
    parser.add_argument("--no-open-report", action="store_true", help="Do not auto-open the HTML report")
    parser.add_argument("--validate-only", action="store_true", help="Validate case inputs without running the pipeline")
    args = parser.parse_args(argv)

    resolved_inputs, report_path, payload = _validate_case_inputs(args.config)
    if args.validate_only:
        sample_count = _fully_validate_case_payload(payload, resolved_inputs)
        print("Case inputs validated:")
        for name, path in resolved_inputs.items():
            print(f"  {name}: {path}")
        print(f"  parsed_samples: {sample_count}")
        print(f"  report_path: {report_path}")
        return 0

    main_argv = ["--config", str(args.config)]
    if not args.no_open_report:
        main_argv.append("--open-report")
    exit_code = run_main(main_argv)
    if exit_code != 0:
        return exit_code

    outputs = _report_paths(report_path)
    print("Case run completed.")
    print(f"HTML report: {outputs['html']}")
    print(f"Summary TXT: {outputs['summary_txt']}")
    print(f"Summary JSON: {outputs['summary_json']}")
    print(f"Records CSV: {outputs['records_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
