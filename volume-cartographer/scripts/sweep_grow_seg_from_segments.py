#!/usr/bin/env python3
"""
Parameter sweep runner for vc_grow_seg_from_segments.

This script:
1) Reads a baseline trace params JSON.
2) Reads sweep definitions from a text file with rows like:
      "step": 2,10,4
      "z_loc_loss_w": 0.01,0.1,0.03
   If a parameter appears multiple times, rows are treated as piecewise segments
   for that parameter and merged into one unique value list.
3) Runs one parameter at a time while all other parameters remain at defaults.
   (No Cartesian product.)
4) After each run, inspects newly created directories in traces/:
   - If a new non-empty auto_trace_* exists, deletes only auto_grown_* created
     during this run and writes a matching auto_trace_*.json param snapshot.
   - If no new auto_trace_* exists, picks the newest eligible auto_grown_*
     created during this run (excluding *_opt, *inp_hr, *_opt_inp_hr), renames
     it to auto_saved_*, writes auto_saved_*.json params, and deletes the rest
     of auto_grown_* created during this run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SweepParam:
    key_path: str
    min_val: Decimal
    max_val: Decimal
    step: Decimal
    output_scale: int = 0

    def values(self) -> list[int | float]:
        if self.step < 0:
            raise ValueError(f"Step must be >= 0 for '{self.key_path}'")

        quantizer = Decimal("1").scaleb(-self.output_scale)
        if self.step == 0:
            if self.min_val != self.max_val:
                raise ValueError(
                    f"Step 0 requires min == max for '{self.key_path}'"
                )
            if self.output_scale > 0:
                return [float(self.min_val.quantize(quantizer))]
            if self.min_val == self.min_val.to_integral_value():
                return [int(self.min_val)]
            return [float(self.min_val)]

        ascending = self.min_val <= self.max_val
        out: list[int | float] = []
        cur = self.min_val
        while (cur <= self.max_val) if ascending else (cur >= self.max_val):
            if self.output_scale > 0:
                # Keep decimal-style sweeps as floats (e.g. 1.0, 2.0),
                # instead of collapsing whole points to ints.
                out.append(float(cur.quantize(quantizer)))
            elif cur == cur.to_integral_value():
                out.append(int(cur))
            else:
                out.append(float(cur))
            cur = cur + self.step if ascending else cur - self.step
        return out


def parse_sweep_file(path: Path) -> list[SweepParam]:
    """
    Parse lines formatted as:
      "param.path": min,max,step
    Comments (# ...) and blank lines are allowed. Repeated parameter keys are
    handled later as piecewise segments for one parameter.
    """
    params: list[SweepParam] = []
    for i, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()

        if ":" not in line:
            raise ValueError(f"{path}:{i}: Missing ':' separator")

        lhs, rhs = line.split(":", 1)
        key = lhs.strip().strip('"').strip("'")
        if not key:
            raise ValueError(f"{path}:{i}: Empty parameter key")

        pieces = [p.strip() for p in rhs.split(",")]
        if len(pieces) != 3:
            raise ValueError(
                f"{path}:{i}: Expected 'min,max,step' after ':', got '{rhs.strip()}'"
            )
        scale = 0
        for p in pieces:
            if "." in p:
                scale = max(scale, len(p.split(".", 1)[1]))
        try:
            min_val, max_val, step = (Decimal(p) for p in pieces)
        except InvalidOperation as exc:
            raise ValueError(
                f"{path}:{i}: min,max,step must be numeric"
            ) from exc

        params.append(
            SweepParam(
                key_path=key,
                min_val=min_val,
                max_val=max_val,
                step=step,
                output_scale=scale,
            )
        )

    if not params:
        raise ValueError(f"No sweep parameters found in {path}")
    return params


def get_dir_mtime_ns(path: Path) -> int:
    return path.stat().st_mtime_ns


def is_nonempty_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False


def list_dirs_with_prefix(root: Path, prefix: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not root.exists():
        return out
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            out[p.name] = p
    return out


def set_key_path(obj: dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set nested dict value using dot-separated key path (e.g. "a.b.c").
    """
    parts = key_path.split(".")
    cur: Any = obj
    for part in parts[:-1]:
        if not isinstance(cur, dict):
            raise KeyError(f"Cannot descend through non-dict at '{part}' in '{key_path}'")
        if part not in cur:
            raise KeyError(f"Missing key '{part}' in '{key_path}'")
        cur = cur[part]
    leaf = parts[-1]
    if not isinstance(cur, dict):
        raise KeyError(f"Cannot set leaf in non-dict for '{key_path}'")
    if leaf not in cur:
        raise KeyError(f"Missing key '{leaf}' in '{key_path}'")
    cur[leaf] = value


def build_values_by_key(params: list[SweepParam]) -> dict[str, list[int | float]]:
    # Repeated parameter rows are treated as piecewise segments for one parameter.
    values_by_key: dict[str, list[int | float]] = {}
    scale_by_key: dict[str, int] = {}
    for p in params:
        if p.key_path in scale_by_key and scale_by_key[p.key_path] != p.output_scale:
            raise ValueError(
                f"Inconsistent decimal scale for '{p.key_path}': "
                f"{scale_by_key[p.key_path]} vs {p.output_scale}. "
                "Use one decimal precision per parameter across all segments."
            )
        scale_by_key.setdefault(p.key_path, p.output_scale)
        values_by_key.setdefault(p.key_path, [])
        for v in p.values():
            if v not in values_by_key[p.key_path]:
                values_by_key[p.key_path].append(v)
    return values_by_key


def build_one_at_a_time_runs(params: list[SweepParam]) -> list[dict[str, int | float]]:
    values_by_key = build_values_by_key(params)
    runs: list[dict[str, int | float]] = []
    for key, values in values_by_key.items():
        for value in values:
            runs.append({key: value})
    return runs


def build_combinations(params: list[SweepParam]) -> list[dict[str, int | float]]:
    """
    Backward-compatible alias for callers using the old function name.
    """
    return build_one_at_a_time_runs(params)


def write_params_snapshot(snapshot_path: Path, params_obj: dict[str, Any]) -> None:
    snapshot_path.write_text(json.dumps(params_obj, indent=2) + "\n")


def next_available_name(root: Path, desired: str) -> str:
    candidate = desired
    if not (root / candidate).exists():
        return candidate
    ts = int(time.time())
    candidate = f"{desired}_{ts}"
    if not (root / candidate).exists():
        return candidate
    i = 2
    while (root / f"{candidate}_{i}").exists():
        i += 1
    return f"{candidate}_{i}"


def should_exclude_from_failure_pick(name: str) -> bool:
    return name.endswith("inp_hr") or name.endswith("_opt_inp_hr") or name.endswith("_opt")


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def pid_is_running_linux(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def acquire_single_run_lock(lock_path: Path) -> None:
    while True:
        try:
            fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"{os.getpid()}\n")
            return
        except FileExistsError:
            existing_pid: int | None = None
            if lock_path.exists():
                content = lock_path.read_text().strip()
                if content.isdigit():
                    existing_pid = int(content)

            if existing_pid is not None and pid_is_running_linux(existing_pid):
                raise RuntimeError(
                    f"Another sweep appears active (pid={existing_pid}) with lock {lock_path}"
                )

            # Stale lock: remove and retry atomic create.
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def build_status_payload(
    *,
    state: str,
    pid: int,
    total_runs: int,
    completed_runs: int,
    failures: int,
    started_at_epoch_s: float,
    ended_at_epoch_s: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "state": state,
        "pid": pid,
        "started_at_epoch_s": started_at_epoch_s,
        "last_updated_epoch_s": time.time(),
        "total_runs": total_runs,
        "completed_runs": completed_runs,
        "failures": failures,
    }
    if ended_at_epoch_s is not None:
        payload["ended_at_epoch_s"] = ended_at_epoch_s
    return payload


def run_one_iteration(
    binary: Path,
    volume_path: Path,
    paths_dir: Path,
    traces_dir: Path,
    segment_path: Path,
    baseline_params: dict[str, Any],
    combo: dict[str, int | float],
    run_idx: int,
    total_runs: int,
    work_dir: Path,
) -> int:
    params_obj = json.loads(json.dumps(baseline_params))
    for k, v in combo.items():
        set_key_path(params_obj, k, v)

    params_path = work_dir / f"trace_params_{run_idx:04d}.json"
    write_params_snapshot(params_path, params_obj)

    pre_grown = list_dirs_with_prefix(traces_dir, "auto_grown_")
    pre_trace = list_dirs_with_prefix(traces_dir, "auto_trace_")

    cmd = [
        str(binary),
        str(volume_path),
        str(paths_dir),
        str(traces_dir),
        str(params_path),
        str(segment_path),
    ]

    print(f"\n[{run_idx}/{total_runs}] Running:")
    print("  " + " ".join(cmd))
    print(f"  Params: {combo}")

    proc = subprocess.run(cmd, check=False)
    print(f"  Exit code: {proc.returncode}")

    post_grown = list_dirs_with_prefix(traces_dir, "auto_grown_")
    post_trace = list_dirs_with_prefix(traces_dir, "auto_trace_")

    created_grown = [
        p
        for name, p in post_grown.items()
        if name not in pre_grown
    ]
    created_trace = [
        p
        for name, p in post_trace.items()
        if name not in pre_trace
    ]
    created_trace_nonempty = [p for p in created_trace if is_nonempty_dir(p)]

    if created_trace_nonempty:
        print("  New non-empty auto_trace_* detected.")
        kept_trace_dir = max(created_trace_nonempty, key=get_dir_mtime_ns)
        final_params_name = f"{kept_trace_dir.name}.json"
        final_params_path = traces_dir / next_available_name(traces_dir, final_params_name)
        shutil.move(str(params_path), str(final_params_path))
        params_path = final_params_path
        print(f"  Saved params file: {params_path}")

        for grown_dir in created_grown:
            if grown_dir.exists():
                shutil.rmtree(grown_dir)
                print(f"  Deleted run-grown folder: {grown_dir}")
    else:
        print("  No new non-empty auto_trace_* detected. Applying failure handling.")
        eligible = [p for p in created_grown if not should_exclude_from_failure_pick(p.name)]
        selected: Path | None = None
        if eligible:
            selected = max(eligible, key=get_dir_mtime_ns)
            suffix = selected.name[len("auto_grown_") :]
            desired_saved_name = f"auto_saved_{suffix}"
            final_saved_name = next_available_name(traces_dir, desired_saved_name)
            renamed = traces_dir / final_saved_name
            selected.rename(renamed)
            selected = renamed
            print(f"  Renamed retained folder to: {selected}")

            final_params_name = f"{selected.name}.json"
            final_params_path = traces_dir / next_available_name(traces_dir, final_params_name)
            shutil.move(str(params_path), str(final_params_path))
            params_path = final_params_path
            print(f"  Saved params file: {params_path}")
        else:
            print("  No eligible auto_grown_* folder found to retain.")
            if params_path.exists():
                params_path.unlink()
                print(f"  Deleted temporary params file: {params_path}")

        for grown_dir in created_grown:
            if selected is not None and grown_dir.resolve() == selected.resolve():
                continue
            if grown_dir.exists():
                shutil.rmtree(grown_dir)
                print(f"  Deleted run-grown folder: {grown_dir}")

    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep trace params for vc_grow_seg_from_segments one parameter at a time."
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("/home/djosey/Documents/villa/volume-cartographer/build/bin/vc_grow_seg_from_segments"),
        help="Path to vc_grow_seg_from_segments binary",
    )
    parser.add_argument("--volume", type=Path, required=True, help="Input volume path (zarr)")
    parser.add_argument("--paths-dir", type=Path, required=True, help="Paths directory argument")
    parser.add_argument("--traces-dir", type=Path, required=True, help="Traces directory argument")
    parser.add_argument(
        "--segment-path",
        type=Path,
        default=Path("/media/djosey/nvme3/PHerc0139_ds2.volpkg/paths/auto_grown_20260221035941927"),
        help="Fixed final command argument for vc_grow_seg_from_segments",
    )
    parser.add_argument(
        "--baseline-params",
        type=Path,
        default=Path("/media/djosey/nvme3/PHerc0139_ds2.volpkg/trace_params.json"),
        help="Baseline trace params JSON used as defaults for every run",
    )
    parser.add_argument(
        "--sweep-file",
        type=Path,
        required=True,
        help='Text file of sweep definitions, one per line, e.g. "step": 2,10,4 or "x": 0.1,1.0,0.1',
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for generated per-run params JSON files (default: <traces-dir>/sweep_params_work)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned combinations and exit without running command",
    )

    args = parser.parse_args()

    for p in [
        args.binary,
        args.volume,
        args.paths_dir,
        args.traces_dir,
        args.segment_path,
        args.baseline_params,
        args.sweep_file,
    ]:
        if not p.exists():
            print(f"Error: Path does not exist: {p}", file=sys.stderr)
            return 1

    if args.work_dir is None:
        work_dir = args.traces_dir / "sweep_params_work"
    else:
        work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    status_path = work_dir / "sweep_status.json"
    lock_path = work_dir / "sweep_active.lock"

    sweep_params = parse_sweep_file(args.sweep_file)
    baseline_params = json.loads(args.baseline_params.read_text())
    values_by_key = build_values_by_key(sweep_params)
    runs = build_one_at_a_time_runs(sweep_params)

    try:
        acquire_single_run_lock(lock_path)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    started_at_epoch_s = time.time()
    write_status(
        status_path,
        build_status_payload(
            state="running",
            pid=os.getpid(),
            started_at_epoch_s=started_at_epoch_s,
            total_runs=len(runs),
            completed_runs=0,
            failures=0,
        ),
    )

    print(f"Loaded {len(sweep_params)} sweep rows from {args.sweep_file}")
    print(f"Unique swept parameters: {len(values_by_key)}")
    print(f"Total runs (one-at-a-time): {len(runs)}")
    if args.dry_run:
        for idx, run_override in enumerate(runs, start=1):
            print(f"[{idx}/{len(runs)}] {run_override}")
        write_status(
            status_path,
            build_status_payload(
                state="dry_run_complete",
                pid=os.getpid(),
                started_at_epoch_s=started_at_epoch_s,
                ended_at_epoch_s=time.time(),
                total_runs=len(runs),
                completed_runs=0,
                failures=0,
            ),
        )
        if lock_path.exists():
            lock_path.unlink()
        return 0

    failures = 0
    completed_runs = 0
    try:
        for idx, run_override in enumerate(runs, start=1):
            try:
                rc = run_one_iteration(
                    binary=args.binary,
                    volume_path=args.volume,
                    paths_dir=args.paths_dir,
                    traces_dir=args.traces_dir,
                    segment_path=args.segment_path,
                    baseline_params=baseline_params,
                    combo=run_override,
                    run_idx=idx,
                    total_runs=len(runs),
                    work_dir=work_dir,
                )
                if rc != 0:
                    failures += 1
            except Exception as exc:
                failures += 1
                print(f"  ERROR during run {idx}: {exc}", file=sys.stderr)
            finally:
                completed_runs = idx
                write_status(
                    status_path,
                    build_status_payload(
                        state="running",
                        pid=os.getpid(),
                        started_at_epoch_s=started_at_epoch_s,
                        total_runs=len(runs),
                        completed_runs=completed_runs,
                        failures=failures,
                    ),
                )

        print("\nSweep complete.")
        print(f"Runs: {len(runs)}")
        print(f"Runs with non-zero exit or exception: {failures}")
        write_status(
            status_path,
            build_status_payload(
                state="completed",
                pid=os.getpid(),
                started_at_epoch_s=started_at_epoch_s,
                ended_at_epoch_s=time.time(),
                total_runs=len(runs),
                completed_runs=len(runs),
                failures=failures,
            ),
        )
        return 0 if failures == 0 else 2
    except KeyboardInterrupt:
        write_status(
            status_path,
            build_status_payload(
                state="interrupted",
                pid=os.getpid(),
                started_at_epoch_s=started_at_epoch_s,
                ended_at_epoch_s=time.time(),
                total_runs=len(runs),
                completed_runs=completed_runs,
                failures=failures,
            ),
        )
        raise
    finally:
        if lock_path.exists():
            lock_path.unlink()


if __name__ == "__main__":
    sys.exit(main())
