#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "eutherlink.toml"
DEFAULT_DATA_DIR = DEFAULT_CONFIG_PATH.parent / "data"
DEFAULT_JOBS_MAX_AGE_HOURS = 24
DEFAULT_DOTS_ARTIFACTS_MAX_AGE_HOURS = 12
DEFAULT_DOTS_TEMP_OUTPUTS_MAX_AGE_HOURS = 6
DEFAULT_MIN_AGE_HOURS = 2
DEFAULT_JOBS_MAX_BYTES = 1_500_000_000
DEFAULT_DOTS_ARTIFACTS_MAX_BYTES = 500_000_000
DEFAULT_DOTS_TEMP_OUTPUTS_MAX_BYTES = 2_000_000_000


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
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=Path(os.environ.get("EUTHERLINK_CONFIG", DEFAULT_CONFIG_PATH)))
    config_args, remaining_args = config_parser.parse_known_args()
    toml_config = load_toml_config(config_args.config)

    parser = argparse.ArgumentParser(description="Remove stale transient EutherLink TTS data.")
    parser.add_argument("--config", type=Path, default=config_args.config)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(env_or_config("EUTHERLINK_DATA_DIR", toml_config, "server", "data_dir", DEFAULT_DATA_DIR)),
    )
    parser.add_argument(
        "--jobs-max-age-hours",
        type=float,
        default=float(
            env_or_config(
                "EUTHERLINK_CLEANUP_JOBS_MAX_AGE_HOURS",
                toml_config,
                "cleanup",
                "jobs_max_age_hours",
                DEFAULT_JOBS_MAX_AGE_HOURS,
            )
        ),
    )
    parser.add_argument(
        "--dots-artifacts-max-age-hours",
        type=float,
        default=float(
            env_or_config(
                "EUTHERLINK_CLEANUP_DOTS_ARTIFACTS_MAX_AGE_HOURS",
                toml_config,
                "cleanup",
                "dots_artifacts_max_age_hours",
                DEFAULT_DOTS_ARTIFACTS_MAX_AGE_HOURS,
            ),
        ),
    )
    parser.add_argument(
        "--dots-temp-output-dir",
        type=Path,
        default=optional_path(env_or_config("DOTS_TTS_TEMP_OUTPUT_DIR", toml_config, "dots_tts", "temp_output_dir")),
    )
    parser.add_argument(
        "--dots-temp-outputs-max-age-hours",
        type=float,
        default=float(
            env_or_config(
                "EUTHERLINK_CLEANUP_DOTS_TEMP_OUTPUTS_MAX_AGE_HOURS",
                toml_config,
                "cleanup",
                "dots_temp_outputs_max_age_hours",
                DEFAULT_DOTS_TEMP_OUTPUTS_MAX_AGE_HOURS,
            ),
        ),
    )
    parser.add_argument(
        "--min-age-hours",
        type=float,
        default=float(
            env_or_config("EUTHERLINK_CLEANUP_MIN_AGE_HOURS", toml_config, "cleanup", "min_age_hours", DEFAULT_MIN_AGE_HOURS)
        ),
    )
    parser.add_argument(
        "--jobs-max-bytes",
        type=int,
        default=int(
            env_or_config("EUTHERLINK_CLEANUP_JOBS_MAX_BYTES", toml_config, "cleanup", "jobs_max_bytes", DEFAULT_JOBS_MAX_BYTES)
        ),
    )
    parser.add_argument(
        "--dots-artifacts-max-bytes",
        type=int,
        default=int(
            env_or_config(
                "EUTHERLINK_CLEANUP_DOTS_ARTIFACTS_MAX_BYTES",
                toml_config,
                "cleanup",
                "dots_artifacts_max_bytes",
                DEFAULT_DOTS_ARTIFACTS_MAX_BYTES,
            )
        ),
    )
    parser.add_argument(
        "--dots-temp-outputs-max-bytes",
        type=int,
        default=int(
            env_or_config(
                "EUTHERLINK_CLEANUP_DOTS_TEMP_OUTPUTS_MAX_BYTES",
                toml_config,
                "cleanup",
                "dots_temp_outputs_max_bytes",
                DEFAULT_DOTS_TEMP_OUTPUTS_MAX_BYTES,
            ),
        ),
    )
    parser.add_argument("--dry-run", action="store_true", default=os.environ.get("EUTHERLINK_CLEANUP_DRY_RUN") == "1")
    args = parser.parse_args(remaining_args)

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
    dots_temp_output_dir = args.dots_temp_output_dir.resolve() if args.dots_temp_output_dir is not None else None
    if args.dots_temp_output_dir is not None:
        targets.append(
            CleanupTarget(
                "dots-temp-outputs",
                args.dots_temp_output_dir,
                args.dots_temp_outputs_max_age_hours,
                args.dots_temp_outputs_max_bytes,
                args.min_age_hours,
            )
        )

    for target in targets:
        _validate_target(data_dir, dots_temp_output_dir, target.path, target.max_age_hours)

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


def load_toml_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        loaded = tomllib.load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config file did not contain a TOML table: {path}")
    return loaded


def config_get(config: dict[str, object], section: str, key: str, default: object = None) -> object:
    value = config.get(section, {})
    if not isinstance(value, dict):
        return default
    return value.get(key, default)


def env_or_config(env_name: str, config: dict[str, object], section: str, key: str, default: object = None) -> object:
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return env_value
    return config_get(config, section, key, default)


def optional_path(value: object) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value))


def _validate_target(data_dir: Path, dots_temp_output_dir: Path | None, path: Path, max_age_hours: float) -> None:
    if max_age_hours < 1:
        raise SystemExit(f"Refusing cleanup with max age below 1 hour for {path}")
    resolved = path.resolve()
    if dots_temp_output_dir is not None and resolved == dots_temp_output_dir:
        return
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
