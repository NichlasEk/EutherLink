#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_DIR = Path("/home/nichlas/EutherLink/data")
DEFAULT_JOBS_MAX_AGE_HOURS = 24
DEFAULT_DOTS_ARTIFACTS_MAX_AGE_HOURS = 12
DEFAULT_MIN_AGE_HOURS = 2
DEFAULT_JOBS_MAX_BYTES = 1_500_000_000
DEFAULT_DOTS_ARTIFACTS_MAX_BYTES = 500_000_000


@dataclass(frozen=True)
class CleanupTarget:
    label: str
    path: Path
    max_age_hours: float
    max_bytes: int
    min_age_hours: float


@dataclass
class CleanupStats:
    scanned: int = 0
    removed: int = 0
    bytes_removed: int = 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove stale transient EutherLink TTS data.")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("EUTHERLINK_DATA_DIR", DEFAULT_DATA_DIR)))
    parser.add_argument(
        "--jobs-max-age-hours",
        type=float,
        default=float(os.environ.get("EUTHERLINK_CLEANUP_JOBS_MAX_AGE_HOURS", DEFAULT_JOBS_MAX_AGE_HOURS)),
    )
    parser.add_argument(
        "--dots-artifacts-max-age-hours",
        type=float,
        default=float(
            os.environ.get(
                "EUTHERLINK_CLEANUP_DOTS_ARTIFACTS_MAX_AGE_HOURS",
                DEFAULT_DOTS_ARTIFACTS_MAX_AGE_HOURS,
            ),
        ),
    )
    parser.add_argument(
        "--min-age-hours",
        type=float,
        default=float(os.environ.get("EUTHERLINK_CLEANUP_MIN_AGE_HOURS", DEFAULT_MIN_AGE_HOURS)),
    )
    parser.add_argument(
        "--jobs-max-bytes",
        type=int,
        default=int(os.environ.get("EUTHERLINK_CLEANUP_JOBS_MAX_BYTES", DEFAULT_JOBS_MAX_BYTES)),
    )
    parser.add_argument(
        "--dots-artifacts-max-bytes",
        type=int,
        default=int(os.environ.get("EUTHERLINK_CLEANUP_DOTS_ARTIFACTS_MAX_BYTES", DEFAULT_DOTS_ARTIFACTS_MAX_BYTES)),
    )
    parser.add_argument("--dry-run", action="store_true", default=os.environ.get("EUTHERLINK_CLEANUP_DRY_RUN") == "1")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    targets = [
        CleanupTarget("jobs", data_dir / "jobs", args.jobs_max_age_hours, args.jobs_max_bytes, args.min_age_hours),
        CleanupTarget(
            "dots-worker-artifacts",
            data_dir / "dots-worker-artifacts",
            args.dots_artifacts_max_age_hours,
            args.dots_artifacts_max_bytes,
            args.min_age_hours,
        ),
    ]

    for target in targets:
        _validate_target(data_dir, target.path, target.max_age_hours)

    total = CleanupStats()
    for target in targets:
        stats = clean_target(target, time.time(), dry_run=args.dry_run)
        total.scanned += stats.scanned
        total.removed += stats.removed
        total.bytes_removed += stats.bytes_removed
        print(
            f"{target.label}: scanned={stats.scanned} removed={stats.removed} "
            f"bytes_removed={stats.bytes_removed} dry_run={args.dry_run}",
            flush=True,
        )

    print(
        f"total: scanned={total.scanned} removed={total.removed} "
        f"bytes_removed={total.bytes_removed} dry_run={args.dry_run}",
        flush=True,
    )
    return 0


def _validate_target(data_dir: Path, path: Path, max_age_hours: float) -> None:
    if max_age_hours < 1:
        raise SystemExit(f"Refusing cleanup with max age below 1 hour for {path}")
    resolved = path.resolve()
    if data_dir not in resolved.parents:
        raise SystemExit(f"Refusing cleanup outside data dir: {resolved}")


def clean_target(target: CleanupTarget, now: float, dry_run: bool) -> CleanupStats:
    stats = CleanupStats()
    if not target.path.exists():
        return stats
    max_age_cutoff = now - target.max_age_hours * 3600
    min_age_cutoff = now - target.min_age_hours * 3600
    entries: list[tuple[Path, float, int]] = []
    for child in target.path.iterdir():
        if not child.is_dir():
            continue
        stats.scanned += 1
        try:
            mtime = child.stat().st_mtime
        except OSError:
            continue
        size = directory_size(child)
        entries.append((child, mtime, size))

    total_bytes = sum(size for _child, _mtime, size in entries)
    stale_paths = {child for child, mtime, _size in entries if mtime <= max_age_cutoff}
    removable_for_size = [
        (child, mtime, size)
        for child, mtime, size in sorted(entries, key=lambda entry: entry[1])
        if mtime <= min_age_cutoff
    ]
    for child, _mtime, size in removable_for_size:
        if total_bytes <= target.max_bytes:
            break
        stale_paths.add(child)
        total_bytes -= size

    for child, _mtime, size in sorted(entries, key=lambda entry: entry[1]):
        if child not in stale_paths:
            continue
        if not dry_run:
            shutil.rmtree(child)
        stats.removed += 1
        stats.bytes_removed += size
        print(f"{'would remove' if dry_run else 'removed'} {target.label}: {child} bytes={size}", flush=True)
    return stats


def directory_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for file_name in files:
            try:
                total += (Path(root) / file_name).stat().st_size
            except OSError:
                continue
    return total


if __name__ == "__main__":
    sys.exit(main())
