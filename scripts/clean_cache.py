"""Remove generated caches and build artifacts for TERRAIN NAVIGATOR."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIRECTORY_TARGETS = [
    PROJECT_ROOT / ".pytest_cache",
    PROJECT_ROOT / "output",
    PROJECT_ROOT / "terrain_nav_core" / "build",
]
FILE_TARGETS = [
    PROJECT_ROOT / "terrain_navigator.log",
]
RECURSIVE_DIR_NAMES = {"__pycache__"}
RECURSIVE_FILE_SUFFIXES = {".pyc"}


def find_recursive_targets(root: Path) -> list[Path]:
    """Collect Python cache artifacts under the project tree."""

    results: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir() and path.name in RECURSIVE_DIR_NAMES:
            results.append(path)
        elif path.is_file() and path.suffix in RECURSIVE_FILE_SUFFIXES:
            results.append(path)
    return results


def unique_existing_targets() -> list[Path]:
    """Return existing cleanup targets without duplicates."""

    candidates = [*DIRECTORY_TARGETS, *FILE_TARGETS, *find_recursive_targets(PROJECT_ROOT)]
    seen: set[Path] = set()
    results: list[Path] = []
    for candidate in candidates:
        if candidate.exists() and candidate not in seen:
            seen.add(candidate)
            results.append(candidate)

    # If a directory is already scheduled for deletion, skip its children.
    unique_targets: list[Path] = []
    for candidate in sorted(results, key=lambda path: (len(path.parts), str(path))):
        if any(parent in unique_targets for parent in candidate.parents):
            continue
        unique_targets.append(candidate)
    return unique_targets


def remove_target(path: Path) -> None:
    """Delete one file or directory target."""

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Clean TERRAIN NAVIGATOR caches and build artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting")
    args = parser.parse_args(argv)

    targets = unique_existing_targets()
    if not targets:
        print("Nothing to clean")
        return 0

    for target in targets:
        print(target.relative_to(PROJECT_ROOT))
        if not args.dry_run:
            remove_target(target)

    print("Clean complete" if not args.dry_run else "Dry run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
