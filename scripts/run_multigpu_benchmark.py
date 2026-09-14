#!/usr/bin/env python3
"""Build, run, and report HephFlow multi-GPU scaling experiments.

The script temporarily edits the compile-time benchmark configuration, builds
one executable per GPU-count/algorithm variant, restores the original source
files, and writes a portable result bundle containing logs, CSV, JSON, and a
Markdown summary.

Examples:
  python scripts/run_multigpu_benchmark.py --preset k80
  python3 scripts/run_multigpu_benchmark.py --preset a100
  python3 scripts/run_multigpu_benchmark.py --preset a100 --repeats 5
  python scripts/run_multigpu_benchmark.py --preset k80 --gpu-sets "0;0,2;0,1,2,3"
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable


PRESETS = {
    "k80": {
        "domain": 256,
        "compute_capability": "37",
        "gpu_sets": [[0], [0, 1], [0, 1, 2, 3]],
        "note": (
            "Tesla K80 is compute capability 3.7. CUDA 12 removed Kepler code "
            "generation; use a CUDA 11.x toolkit and a compatible driver."
        ),
    },
    "a100": {
        "domain": 512,
        "compute_capability": "80",
        "gpu_sets": [[0], [0, 1]],
        "note": "The 512^3 global domain is used so the A100s have enough work.",
    },
}

MLUPS_RE = re.compile(
    r"MLUPS:\s*([0-9]+(?:\.[0-9]+)?)\s*"
    r"\(steps=(\d+),\s*nodes=(\d+),\s*time=([0-9]+(?:\.[0-9]+)?)s\)"
)


def command_text(command: Iterable[str]) -> str:
    return " ".join(str(part) for part in command)


def capture_quiet(command: list[str], cwd: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return {
            "command": command_text(command),
            "returncode": completed.returncode,
            "output": completed.stdout.strip(),
        }
    except OSError as exc:
        return {
            "command": command_text(command),
            "returncode": None,
            "output": f"Unable to execute: {exc}",
        }


def run_logged(
    command: list[str], cwd: Path, log_path: Path, env: dict[str, str] | None = None
) -> tuple[int, str]:
    print(f"\n$ {command_text(command)}", flush=True)
    chunks: list[str] = []
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        log.write(f"$ {command_text(command)}\n\n")
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
        except OSError as exc:
            message = f"Unable to execute command: {exc}\n"
            print(message, end="", flush=True)
            log.write(message)
            return 127, message

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            chunks.append(line)
        return process.wait(), "".join(chunks)


def replace_exact(text: str, pattern: str, replacement: str, label: str) -> str:
    result, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label}; found {count}")
    return result


def configured_sources(
    var_text: str,
    constants_text: str,
    gpu_ids: list[int],
    domain: int,
    steps: int,
    overlap: bool,
) -> tuple[str, str]:
    gpu_list = ", ".join(str(gpu_id) for gpu_id in gpu_ids)
    var_text = replace_exact(
        var_text,
        r"^constexpr unsigned int N_GPUS = \d+;[^\r\n]*(?=\r?$)",
        f"constexpr unsigned int N_GPUS = {len(gpu_ids)};"
        "                      // Number of GPUs to use",
        "N_GPUS declaration",
    )
    var_text = replace_exact(
        var_text,
        r"^constexpr unsigned int GPUS_TO_USE\[N_GPUS\] = \{[^}]*\};[^\r\n]*(?=\r?$)",
        f"constexpr unsigned int GPUS_TO_USE[N_GPUS] = {{{gpu_list}}};"
        "       // Which GPUs to use",
        "GPUS_TO_USE declaration",
    )
    constants_text = replace_exact(
        constants_text,
        r"^constexpr int N_STEPS = \d+;[^\r\n]*(?=\r?$)",
        f"constexpr int N_STEPS = {steps};              // Total simulation steps",
        "N_STEPS declaration",
    )
    constants_text = replace_exact(
        constants_text,
        r"^constexpr int N = \d+ \* SCALE;[^\r\n]*(?=\r?$)",
        f"constexpr int N = {domain} * SCALE;"
        "              // Reference grid size",
        "domain-size declaration",
    )
    constants_text = replace_exact(
        constants_text,
        r"^[ \t]*(?://[ \t]*)?#define STEP11_SPLIT_Z_KERNEL[^\r\n]*(?=\r?$)",
        "#define STEP11_SPLIT_Z_KERNEL",
        "STEP11_SPLIT_Z_KERNEL guard",
    )
    step12 = (
        "#define STEP12_OVERLAP_HALOS"
        if overlap
        else "// #define STEP12_OVERLAP_HALOS"
    )
    constants_text = replace_exact(
        constants_text,
        r"^[ \t]*(?://[ \t]*)?#define STEP12_OVERLAP_HALOS[^\r\n]*(?=\r?$)",
        step12,
        "STEP12_OVERLAP_HALOS guard",
    )
    return var_text, constants_text


def parse_gpu_sets(raw: str | None, defaults: list[list[int]]) -> list[list[int]]:
    if raw is None:
        return defaults
    result: list[list[int]] = []
    for group in raw.split(";"):
        ids = [int(item.strip()) for item in group.split(",") if item.strip()]
        if not ids:
            raise ValueError("GPU sets cannot contain an empty group")
        if len(ids) != len(set(ids)):
            raise ValueError(f"GPU set contains a duplicate device: {ids}")
        result.append(ids)
    if not result:
        raise ValueError("At least one GPU set is required")
    return result


def locate_bash(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.name == "nt":
        candidates = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "Git"
            / "bin"
            / "bash.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "Git"
            / "usr"
            / "bin"
            / "bash.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    found = shutil.which("bash")
    if found:
        return found
    raise RuntimeError("bash was not found; install Git Bash or pass --bash PATH")


def locate_nvidia_smi(explicit: str | None) -> str:
    if explicit:
        return explicit
    candidates: list[Path] = []
    if os.name == "nt":
        system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        candidates.append(system_root / "System32" / "nvidia-smi.exe")
        for variable in ("ProgramFiles", "ProgramW6432"):
            base = os.environ.get(variable)
            if base:
                candidates.append(
                    Path(base) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
                )
    else:
        candidates.extend(
            [Path("/usr/bin/nvidia-smi"), Path("/usr/local/cuda/bin/nvidia-smi")]
        )
    candidates.extend(
        Path(item) for item in (shutil.which("nvidia-smi"),) if item
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(
        "nvidia-smi was not found. Add it to PATH or pass --nvidia-smi "
        "PATH (usual Windows path: C:\\Windows\\System32\\nvidia-smi.exe)."
    )


def executable_path(bin_dir: Path, prefix: str, cc: str) -> Path:
    stem = bin_dir / f"{prefix}sim_D3Q19_sm{cc}"
    candidates = [stem.with_suffix(".exe"), stem] if os.name == "nt" else [stem, stem.with_suffix(".exe")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Compiler reported success but executable was not found: {stem}")


def parse_mlups(output: str) -> dict[str, Any] | None:
    matches = list(MLUPS_RE.finditer(output))
    if not matches:
        return None
    value, steps, nodes, seconds = matches[-1].groups()
    return {
        "mlups": float(value),
        "steps": int(steps),
        "nodes": int(nodes),
        "seconds": float(seconds),
    }


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def make_summary(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for experiment in experiments:
        values = [
            run["measurement"]["mlups"]
            for run in experiment["runs"]
            if not run["warmup"]
            and run["returncode"] == 0
            and run.get("measurement") is not None
        ]
        row = {
            "gpu_ids": experiment["gpu_ids"],
            "gpu_count": len(experiment["gpu_ids"]),
            "variant": experiment["variant"],
            "mean_mlups": mean_or_none(values),
            "median_mlups": statistics.median(values) if values else None,
            "stdev_mlups": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
            "successful_runs": len(values),
        }
        summary.append(row)

    baseline = next(
        (
            row["mean_mlups"]
            for row in summary
            if row["gpu_count"] == 1
            and row["variant"] == "serialized"
            and row["mean_mlups"] is not None
        ),
        None,
    )
    serialized_by_set = {
        tuple(row["gpu_ids"]): row["mean_mlups"]
        for row in summary
        if row["variant"] == "serialized"
    }
    for row in summary:
        mean = row["mean_mlups"]
        row["speedup"] = mean / baseline if mean is not None and baseline else None
        row["efficiency"] = (
            row["speedup"] / row["gpu_count"] if row["speedup"] is not None else None
        )
        serialized = serialized_by_set.get(tuple(row["gpu_ids"]))
        row["overlap_delta_percent"] = (
            100.0 * (mean / serialized - 1.0)
            if row["variant"] == "overlap" and mean is not None and serialized
            else None
        )
    return summary


def fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_reports(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result["summary"] = make_summary(result.get("experiments", []))
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8", newline="\n"
    )

    with (output_dir / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "gpu_ids",
                "gpu_count",
                "variant",
                "warmup",
                "run",
                "mlups",
                "seconds",
                "steps",
                "nodes",
                "returncode",
                "log",
            ],
        )
        writer.writeheader()
        for experiment in result.get("experiments", []):
            for run in experiment.get("runs", []):
                measurement = run.get("measurement") or {}
                writer.writerow(
                    {
                        "gpu_ids": ",".join(map(str, experiment["gpu_ids"])),
                        "gpu_count": len(experiment["gpu_ids"]),
                        "variant": experiment["variant"],
                        "warmup": run["warmup"],
                        "run": run["run"],
                        "mlups": measurement.get("mlups", ""),
                        "seconds": measurement.get("seconds", ""),
                        "steps": measurement.get("steps", ""),
                        "nodes": measurement.get("nodes", ""),
                        "returncode": run["returncode"],
                        "log": run["log"],
                    }
                )

    lines = [
        "# HephFlow multi-GPU benchmark",
        "",
        f"- Generated: `{result.get('finished_at', 'incomplete')}`",
        f"- Preset: `{result['configuration']['preset']}`",
        f"- Git commit: `{result['system'].get('git_commit', 'unknown')}`",
        f"- Global domain: `{result['configuration']['domain']}^3`",
        f"- Steps: `{result['configuration']['steps']}`",
        f"- Measured repetitions: `{result['configuration']['repeats']}`",
        f"- Warm-ups per executable: `{result['configuration']['warmups']}`",
        f"- Preset note: {result['configuration']['preset_note']}",
        "",
        "## Summary",
        "",
        "| GPUs | Device IDs | Variant | Mean MLUPS | Median | Std. dev. | Speedup | Efficiency | Overlap delta | Runs |",
        "|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["summary"]:
        overlap = (
            "n/a"
            if row["overlap_delta_percent"] is None
            else f"{row['overlap_delta_percent']:+.3f}%"
        )
        efficiency = (
            "n/a"
            if row["efficiency"] is None
            else f"{row['efficiency'] * 100.0:.2f}%"
        )
        lines.append(
            f"| {row['gpu_count']} | `{','.join(map(str, row['gpu_ids']))}` | "
            f"{row['variant']} | {fmt(row['mean_mlups'])} | "
            f"{fmt(row['median_mlups'])} | {fmt(row['stdev_mlups'])} | "
            f"{fmt(row['speedup'])} | {efficiency} | "
            f"{overlap} | {row['successful_runs']} |"
        )

    lines += ["", "## Hardware and software", ""]
    for key in ("platform", "python"):
        lines.append(f"- {key}: `{result['system'].get(key, 'unknown')}`")
    for key in ("nvidia_smi", "driver_model", "nvcc", "git_status"):
        item = result["system"].get(key, {})
        lines += ["", f"### {key}", "", "```text", item.get("output", "not captured"), "```"]
    topology = result["system"].get("topology", {})
    lines += ["", "## GPU topology", "", "```text", topology.get("output", "not captured"), "```"]

    lines += ["", "## Individual runs", ""]
    for experiment in result.get("experiments", []):
        lines.append(
            f"### GPUs {','.join(map(str, experiment['gpu_ids']))} - {experiment['variant']}"
        )
        lines.append("")
        lines.append(f"- Build return code: `{experiment.get('build_returncode')}`")
        lines.append(f"- Build log: `{experiment.get('build_log', 'n/a')}`")
        for run in experiment.get("runs", []):
            kind = "warm-up" if run["warmup"] else f"run {run['run']}"
            measurement = run.get("measurement")
            value = f"{measurement['mlups']:.6f} MLUPS" if measurement else "no MLUPS parsed"
            lines.append(
                f"- {kind}: {value}, return code `{run['returncode']}`, log `{run['log']}`"
            )
        lines.append("")

    if result.get("fatal_error"):
        lines += ["## Fatal error", "", "```text", result["fatal_error"], "```", ""]
    lines += [
        "## Interpretation notes",
        "",
        "- Speedup uses the one-GPU serialized mean for the same fixed global domain.",
        "- Efficiency is `speedup / GPU count`.",
        "- Overlap delta compares Step 12 against serialized Step 11 on the same GPU set.",
        "- Check the topology and run logs for `P2P access enabled` or access warnings before interpreting scaling.",
        "- On Windows, verify the Tesla devices use TCC rather than WDDM mode; the driver-model query is captured above.",
        "- Initialization and final output are outside HephFlow's reported MLUPS timer.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def self_test(repo_root: Path) -> int:
    var_path = repo_root / "src" / "var.h"
    constants_path = repo_root / "src" / "cases" / "000_benchmark" / "constants.inc"
    var_original = var_path.read_bytes().decode("utf-8")
    constants_original = constants_path.read_bytes().decode("utf-8")
    var_new, constants_new = configured_sources(
        var_original, constants_original, [0, 2, 3, 5], 512, 1234, True
    )
    assert "N_GPUS = 4" in var_new
    assert "{0, 2, 3, 5}" in var_new
    assert "N_STEPS = 1234" in constants_new
    assert "N = 512 * SCALE" in constants_new
    assert re.search(r"^#define STEP12_OVERLAP_HALOS\r?$", constants_new, re.MULTILINE)
    assert parse_mlups("MLUPS: 1234.5  (steps=10, nodes=42, time=0.123s)") == {
        "mlups": 1234.5,
        "steps": 10,
        "nodes": 42,
        "seconds": 0.123,
    }
    assert parse_gpu_sets("0;0,2;0,1,2,3", []) == [[0], [0, 2], [0, 1, 2, 3]]
    print("Self-test PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Hardware/domain preset")
    parser.add_argument("--domain", type=int, help="Override cubic global-domain edge")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--gpu-sets",
        help='Semicolon-separated device sets, e.g. "0;0,1;0,1,2,3"',
    )
    parser.add_argument("--compute-capability", help="Override target, e.g. 37 or 80")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("serialized", "overlap"),
        default=("serialized", "overlap"),
    )
    parser.add_argument("--bash", help="Path to bash/Git Bash")
    parser.add_argument("--nvidia-smi", help="Path to nvidia-smi executable")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.self_test:
        return self_test(repo_root)
    if not args.preset:
        raise SystemExit("--preset is required unless --self-test is used")
    if args.steps <= 0 or args.repeats <= 0 or args.warmups < 0:
        raise SystemExit("steps/repeats must be positive and warmups cannot be negative")

    preset = PRESETS[args.preset]
    domain = args.domain or preset["domain"]
    cc = args.compute_capability or preset["compute_capability"]
    gpu_sets = parse_gpu_sets(args.gpu_sets, preset["gpu_sets"])
    if domain <= 0 or any(domain % len(gpu_ids) for gpu_ids in gpu_sets):
        raise SystemExit("The global domain must be positive and divisible by every GPU count")

    timestamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or repo_root / "benchmark_results" / f"{args.preset}_{timestamp}").resolve()
    print(f"Preset note: {preset['note']}")
    print(f"Result directory: {output_dir}")
    print("Planned configurations:")
    for gpu_ids in gpu_sets:
        variants = ["serialized"] if len(gpu_ids) == 1 else list(args.variants)
        print(f"  GPUs {gpu_ids}: {', '.join(variants)}")
    if args.dry_run:
        return 0

    bash = locate_bash(args.bash)
    nvidia_smi = locate_nvidia_smi(args.nvidia_smi)
    src_dir = repo_root / "src"
    bin_dir = repo_root / "bin"
    bin_dir.mkdir(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    log_dir = output_dir / "logs"
    log_dir.mkdir()

    var_path = src_dir / "var.h"
    constants_path = src_dir / "cases" / "000_benchmark" / "constants.inc"
    originals = {
        var_path: var_path.read_bytes(),
        constants_path: constants_path.read_bytes(),
    }
    backup_dir = output_dir / "source_backup"
    backup_dir.mkdir()
    (backup_dir / "var.h").write_bytes(originals[var_path])
    (backup_dir / "constants.inc").write_bytes(originals[constants_path])
    var_original = originals[var_path].decode("utf-8")
    constants_original = originals[constants_path].decode("utf-8")

    system = {
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "git_commit": capture_quiet(["git", "rev-parse", "HEAD"], repo_root)["output"],
        "git_status": capture_quiet(["git", "status", "--short"], repo_root),
        "nvidia_smi": capture_quiet(
            [
                nvidia_smi,
                "--query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            repo_root,
        ),
        "topology": capture_quiet([nvidia_smi, "topo", "-m"], repo_root),
        "driver_model": capture_quiet(
            [nvidia_smi, "--query-gpu=index,driver_model.current", "--format=csv,noheader"],
            repo_root,
        ),
        "nvcc": capture_quiet(["nvcc", "--version"], repo_root),
    }
    if system["topology"]["returncode"] != 0:
        system["topology"]["output"] += (
            "\n\nThe topology matrix is not supported by this nvidia-smi build. "
            "PCI bus IDs are recorded above; HephFlow's run logs still report "
            "the result of cudaDeviceCanAccessPeer for every device pair."
        )
    visible_ids = set()
    for line in system["nvidia_smi"]["output"].splitlines():
        first = line.split(",", 1)[0].strip()
        if first.isdigit():
            visible_ids.add(int(first))
    requested_ids = {gpu_id for gpu_ids in gpu_sets for gpu_id in gpu_ids}
    missing = sorted(requested_ids - visible_ids)
    if missing:
        raise SystemExit(f"Requested GPU IDs are not visible to nvidia-smi: {missing}")
    if args.preset == "k80":
        nvcc_output = system["nvcc"]["output"]
        release = re.search(r"release\s+(\d+)", nvcc_output)
        if release and int(release.group(1)) >= 12:
            raise SystemExit("The K80 preset requires CUDA 11.x; this nvcc is CUDA 12 or newer")

    result: dict[str, Any] = {
        "started_at": dt.datetime.now().astimezone().isoformat(),
        "configuration": {
            "preset": args.preset,
            "preset_note": preset["note"],
            "domain": domain,
            "steps": args.steps,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "gpu_sets": gpu_sets,
            "variants": list(args.variants),
            "compute_capability": cc,
        },
        "system": system,
        "experiments": [],
    }
    failed = False

    try:
        compiled: list[dict[str, Any]] = []
        for gpu_ids in gpu_sets:
            variants = ["serialized"] if len(gpu_ids) == 1 else list(args.variants)
            for variant in variants:
                overlap = variant == "overlap"
                var_new, constants_new = configured_sources(
                    var_original, constants_original, gpu_ids, domain, args.steps, overlap
                )
                var_path.write_bytes(var_new.encode("utf-8"))
                constants_path.write_bytes(constants_new.encode("utf-8"))
                ids_label = "-".join(map(str, gpu_ids))
                prefix = f"mgpu_{args.preset}_g{len(gpu_ids)}_{ids_label}_{variant}_"
                build_log = log_dir / f"build_g{len(gpu_ids)}_{ids_label}_{variant}.log"
                build_env = os.environ.copy()
                build_env["CC"] = cc
                nvidia_dir = str(Path(nvidia_smi).resolve().parent)
                build_env["PATH"] = nvidia_dir + os.pathsep + build_env.get("PATH", "")
                if args.preset == "k80":
                    prior_flags = build_env.get("NVCC_PREPEND_FLAGS", "").strip()
                    compatibility_flag = "-DHEPHFLOW_SKIP_CONSTEXPR_MATH_TESTS"
                    build_env["NVCC_PREPEND_FLAGS"] = (
                        f"{prior_flags} {compatibility_flag}".strip()
                    )
                returncode, _ = run_logged(
                    [bash, "./compile.sh", prefix], src_dir, build_log, build_env
                )
                experiment = {
                    "gpu_ids": gpu_ids,
                    "variant": variant,
                    "build_returncode": returncode,
                    "build_log": str(build_log.relative_to(output_dir)),
                    "runs": [],
                }
                result["experiments"].append(experiment)
                if returncode != 0:
                    failed = True
                    if not args.continue_on_error:
                        raise RuntimeError(f"Build failed for GPUs {gpu_ids}, {variant}")
                    continue
                experiment["executable"] = str(executable_path(bin_dir, prefix, cc))
                compiled.append(experiment)

        for experiment in compiled:
            exe = Path(experiment["executable"])
            ids_label = "-".join(map(str, experiment["gpu_ids"]))
            label = f"g{len(experiment['gpu_ids'])}_{ids_label}_{experiment['variant']}"
            for warmup_index in range(1, args.warmups + 1):
                log_path = log_dir / f"run_{label}_warmup{warmup_index}.log"
                returncode, output = run_logged([str(exe)], bin_dir, log_path)
                measurement = parse_mlups(output)
                experiment["runs"].append(
                    {
                        "warmup": True,
                        "run": warmup_index,
                        "returncode": returncode,
                        "measurement": measurement,
                        "log": str(log_path.relative_to(output_dir)),
                    }
                )
                if returncode != 0 or measurement is None:
                    failed = True
                    if not args.continue_on_error:
                        raise RuntimeError(f"Warm-up failed for {label}")

        # Alternate variant order at each repetition to reduce order bias.
        for gpu_ids in gpu_sets:
            group = [exp for exp in compiled if exp["gpu_ids"] == gpu_ids]
            for repetition in range(1, args.repeats + 1):
                ordered = group if repetition % 2 else list(reversed(group))
                for experiment in ordered:
                    exe = Path(experiment["executable"])
                    ids_label = "-".join(map(str, gpu_ids))
                    label = f"g{len(gpu_ids)}_{ids_label}_{experiment['variant']}"
                    log_path = log_dir / f"run_{label}_{repetition}.log"
                    returncode, output = run_logged([str(exe)], bin_dir, log_path)
                    measurement = parse_mlups(output)
                    experiment["runs"].append(
                        {
                            "warmup": False,
                            "run": repetition,
                            "returncode": returncode,
                            "measurement": measurement,
                            "log": str(log_path.relative_to(output_dir)),
                        }
                    )
                    if returncode != 0 or measurement is None:
                        failed = True
                        if not args.continue_on_error:
                            raise RuntimeError(f"Measured run failed for {label}")
    except BaseException:
        result["fatal_error"] = traceback.format_exc()
        failed = True
    finally:
        for path, content in originals.items():
            path.write_bytes(content)
        result["sources_restored"] = all(
            path.read_bytes() == content for path, content in originals.items()
        )
        result["finished_at"] = dt.datetime.now().astimezone().isoformat()
        write_reports(result, output_dir)
        print(f"\nSources restored: {result['sources_restored']}")
        print(f"Markdown report: {output_dir / 'report.md'}")
        print(f"JSON results:   {output_dir / 'results.json'}")
        print(f"CSV runs:       {output_dir / 'runs.csv'}")

    return 1 if failed or not result["sources_restored"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
