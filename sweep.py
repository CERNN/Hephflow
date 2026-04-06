#!/usr/bin/env python3
"""
Hephflow Parameter Sweep
========================
Edits compile-time constants in a case, recompiles, and runs the binary
for each parameter set in a sweep.

Quick start (in-file config)
----------------------------
1) Edit CASE_NAME and SWEEP_PARAMETERS in this file.
2) Run:
    python sweep.py

Usage
-----
Single variable:
    python sweep.py 007_twoLayerChannel N=32,64,128,256,512

Multiple variables — cartesian product (default):
    python sweep.py 007_twoLayerChannel N=32,64,128 TAU=0.6,0.8
    → runs: (32,0.6), (32,0.8), (64,0.6), (64,0.8), (128,0.6), (128,0.8)

Multiple variables — paired/zip mode:
    python sweep.py 007_twoLayerChannel N=32,64 TAU=0.6,0.8 --zip
    → runs: (32,0.6), (64,0.8)

Flags
-----
--prefix PREFIX   Binary prefix (default: leading digits of case, e.g. "007")
--compile-only    Edit and compile but do not run the binary
--run-only        Skip compilation and run existing binaries (must exist)
--dry-run         Print what would be done without modifying any files
--no-id-patch     Do not update ID_SIM in output.inc (all runs overwrite same folder)
--log-dir DIR     Directory to store per-run compile/run logs (default: sweep_logs/)
"""

import csv
import re
import sys
import argparse
import itertools
import subprocess
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths (relative to this script, which lives at the workspace root)
# ---------------------------------------------------------------------------
ROOT_DIR  = Path(__file__).resolve().parent
SRC_DIR   = ROOT_DIR / "src"
BIN_DIR   = ROOT_DIR / "bin"
CASES_DIR = SRC_DIR  / "cases"
VAR_H     = SRC_DIR  / "var.h"


# ---------------------------------------------------------------------------
# In-file sweep configuration (used by default with: python sweep.py)
# ---------------------------------------------------------------------------

CASE_NAME = "007_twoLayerChannel"

# PARAM_GROUPS defines the sweep axes.
# - Variables in the SAME dict are LOCKED (zipped together, move as a pair).
# - Cartesian product is taken ACROSS dicts (independent axes).
#
# Example below: 5 N/FZ pairs × 5 viscosity ratios × 5 PHI_DIFF_REF = 125 runs
PARAM_GROUPS = [
    # Group 1 — mesh resolution (N and FZ locked so U_max stays constant)
    # FZ = FZ_ref * (N_ref/N)²,  ref: N=128, FZ=1e-6
    {
        "N":  ["32",      "64",    "128",   "256",     "512"],
        "FZ": ["1.6e-5",  "4e-6",  "1e-6",  "2.5e-7",  "6.25e-8"],
    },
    # Group 2 — viscosity ratio of phase 2 relative to phase 1 (TAU1 = 0.8, nu1 = 0.1)
    # TAU_PHASE2 = 0.5 + 3 * ratio * nu1   →  ratios 1, 2, 4, 8, 16
    {
        "TAU_PHASE2": ["0.8", "1.1", "1.7", "2.9", "5.3"],
    },
    # Group 3 — phase-field diffusivity
    {
        "PHI_DIFF_REF": ["0.0000033333", "0.000033333", "0.00033333", "0.0033333", "0.033333"],
    },
]


# ---------------------------------------------------------------------------
# File patching helpers
# ---------------------------------------------------------------------------

def _replace_once(pattern: str, replacement: str, text: str, label: str) -> str:
    """Apply a regex substitution exactly once, or raise if not found."""
    new_text, count = re.subn(pattern, replacement, text, count=1)
    if count == 0:
        raise ValueError(f"Pattern not found for '{label}'")
    return new_text


def patch_constexpr(text: str, var: str, value: str) -> str:
    """Replace the RHS of:  constexpr <type> VAR = <old_value>;"""
    pattern = rf'(constexpr\s+\S+\s+{re.escape(var)}\s*=\s*)([^;]+)(;)'
    return _replace_once(pattern, rf'\g<1>{value}\3', text, f"constexpr {var}")


def patch_define(text: str, var: str, value: str) -> str:
    """Replace the value in:  #define VAR <old_value>"""
    pattern = rf'(#define\s+{re.escape(var)}\s+)(\S+)'
    return _replace_once(pattern, rf'\g<1>{value}', text, f"#define {var}")


def patch_define_quoted(text: str, var: str, value: str) -> str:
    """Replace the value in:  #define VAR "old_value"  (preserves quotes)."""
    pattern = rf'(#define\s+{re.escape(var)}\s+"[^"]*")'
    replacement = f'#define {var} "{value}"'
    new_text, count = re.subn(pattern, replacement, text, count=1)
    if count == 0:
        raise ValueError(f"Quoted #define not found for '{var}'")
    return new_text


def patch_variable(file_path: Path, var: str, value: str, dry_run: bool = False) -> None:
    """
    Patch a variable in *file_path*.  Tries constexpr first, then #define.
    """
    text = file_path.read_text(encoding="utf-8")
    patched = None

    # Try constexpr
    try:
        patched = patch_constexpr(text, var, value)
    except ValueError:
        pass

    # Try #define (unquoted)
    if patched is None:
        try:
            patched = patch_define(text, var, value)
        except ValueError:
            pass

    if patched is None:
        raise ValueError(
            f"Variable '{var}' not found as 'constexpr' or '#define' in {file_path}"
        )

    if not dry_run:
        file_path.write_text(patched, encoding="utf-8")


def set_bc_problem(case_name: str, dry_run: bool = False) -> None:
    """Update BC_PROBLEM in var.h."""
    text = VAR_H.read_text(encoding="utf-8")
    patched = patch_define(text, "BC_PROBLEM", case_name)
    if not dry_run:
        VAR_H.write_text(patched, encoding="utf-8")


def set_id_sim(output_inc: Path, id_sim: str, dry_run: bool = False) -> None:
    """Update ID_SIM in output.inc."""
    text = output_inc.read_text(encoding="utf-8")
    patched = patch_define_quoted(text, "ID_SIM", id_sim)
    if not dry_run:
        output_inc.write_text(patched, encoding="utf-8")


# ---------------------------------------------------------------------------
# Compilation and execution
# ---------------------------------------------------------------------------

def compile_case(prefix: str, log_path: Path, dry_run: bool = False) -> int:
    """
    Run  bash compile.sh <prefix>  in SRC_DIR.
    Streams output to stdout and also writes it to *log_path*.
    Returns the process exit code.
    """
    cmd = ["bash", "compile.sh", prefix]
    print(f"  [compile] {' '.join(cmd)}")

    if dry_run:
        print("  [dry-run] skipping compilation")
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(SRC_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log_fh.write(line)
        proc.wait()

    return proc.returncode


def find_binary(prefix: str) -> Path | None:
    """Find the compiled binary matching  <prefix>sim_*  in BIN_DIR."""
    matches = sorted(BIN_DIR.glob(f"{prefix}sim_*"))
    # Filter out .exp / .lib build artifacts
    exes = [p for p in matches if p.suffix not in (".exp", ".lib", ".pdb")]
    return exes[0] if exes else None


def run_simulation(binary: Path, log_path: Path, dry_run: bool = False) -> int:
    """
    Execute *binary* from BIN_DIR.
    Returns the process exit code.
    """
    print(f"  [run] {binary.name}")

    if dry_run:
        print("  [dry-run] skipping run")
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            [str(binary)],
            cwd=str(BIN_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log_fh.write(line)
        proc.wait()

    return proc.returncode


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def make_id(param_set: dict) -> str:
    """
    Build a short, filesystem-safe identifier from a parameter set.
    e.g.  {"N": "32", "TAU": "0.8"}  →  "N_32_TAU_0.8"
    """
    parts = []
    for k, v in param_set.items():
        safe_v = str(v).replace(".", "p").replace("-", "m").replace("+", "")
        parts.append(f"{k}_{safe_v}")
    return "__".join(parts)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_param(s: str) -> tuple[str, list[str]]:
    """
    Parse  "VAR=v1,v2,v3"  →  ("VAR", ["v1", "v2", "v3"])
    """
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"Parameter must be in  VAR=v1,v2,...  format, got: '{s}'"
        )
    var, _, raw_values = s.partition("=")
    values = [v.strip() for v in raw_values.split(",") if v.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"No values provided for '{var}'")
    return var.strip(), values


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hephflow parameter sweep: patch constants, compile, run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "case",
        nargs="?",
        default=None,
        help="Case folder name, e.g. 007_twoLayerChannel (optional if using in-file config)",
    )
    p.add_argument(
        "params",
        nargs="*",
        metavar="VAR=v1,v2,...",
        help="One or more parameter sweeps in VAR=value,... format (optional if using in-file config)",
    )
    p.add_argument(
        "--prefix",
        default=None,
        help="Binary prefix (default: leading digits of case name)",
    )
    p.add_argument(
        "--zip",
        action="store_true",
        help="Zip parameters instead of cartesian product",
    )
    p.add_argument("--compile-only", action="store_true", help="Compile but do not run")
    p.add_argument("--run-only",     action="store_true", help="Run without recompiling")
    p.add_argument("--dry-run",      action="store_true", help="Print plan without changes")
    p.add_argument(
        "--no-id-patch",
        action="store_true",
        help="Do not update ID_SIM in output.inc",
    )
    p.add_argument(
        "--log-dir",
        default="sweep_logs",
        help="Directory for per-run logs (default: sweep_logs/)",
    )
    p.add_argument(
        "--file",
        default="constants.inc",
        help="File to patch inside the case folder (default: constants.inc)",
    )
    return p


def load_sweep_definition(args: argparse.Namespace) -> tuple[str, list[str], list[dict], str]:
    """
    Returns (case_name, all_var_names, list_of_param_dicts, mode_label).
    - CLI mode:    python sweep.py <case> VAR=v1,v2 ...
    - Config mode: python sweep.py  (uses PARAM_GROUPS)
    """
    if args.case is not None:
        if not args.params:
            raise ValueError("If a case is provided on CLI, provide at least one VAR=v1,v2,... parameter.")
        parsed = [parse_param(p) for p in args.params]
        case_name = args.case
        var_names = [v for v, _ in parsed]
        value_lists = [vals for _, vals in parsed]
        if args.zip:
            if len(set(len(v) for v in value_lists)) > 1:
                raise ValueError("--zip requires all parameter lists to have equal length")
            combos = [dict(zip(var_names, c)) for c in zip(*value_lists)]
            mode_label = "zip"
        else:
            combos = [dict(zip(var_names, c)) for c in itertools.product(*value_lists)]
            mode_label = "cartesian product"
        return case_name, var_names, combos, mode_label

    # Config mode — use PARAM_GROUPS
    if not CASE_NAME:
        raise ValueError("CASE_NAME is empty in sweep.py")
    if not PARAM_GROUPS:
        raise ValueError("PARAM_GROUPS is empty in sweep.py")

    all_var_names: list[str] = []
    group_rows: list[list[dict]] = []
    for group in PARAM_GROUPS:
        vars_in_group = list(group.keys())
        values_in_group = [group[v] for v in vars_in_group]
        if len(set(len(v) for v in values_in_group)) > 1:
            raise ValueError(
                f"Variables {vars_in_group} in the same PARAM_GROUPS entry must have equal length"
            )
        rows = [dict(zip(vars_in_group, combo)) for combo in zip(*values_in_group)]
        group_rows.append(rows)
        all_var_names.extend(vars_in_group)

    combos = [
        {k: v for d in combination for k, v in d.items()}
        for combination in itertools.product(*group_rows)
    ]
    group_descs = ["(" + "+".join(g.keys()) + ")" for g in PARAM_GROUPS]
    mode_label = "grouped cartesian: " + " × ".join(group_descs)
    return CASE_NAME, all_var_names, combos, mode_label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        case_name, var_names, combos, mode_label = load_sweep_definition(args)
    except (argparse.ArgumentTypeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # --- Validate case ---
    case_dir = CASES_DIR / case_name
    if not case_dir.is_dir():
        print(f"Error: case directory not found: {case_dir}", file=sys.stderr)
        return 1

    target_file = case_dir / args.file
    if not target_file.is_file():
        print(f"Error: target file not found: {target_file}", file=sys.stderr)
        return 1

    output_inc = case_dir / "output.inc"

    # --- Determine prefix ---
    prefix = args.prefix
    if prefix is None:
        m = re.match(r'^(\d+)', case_name)
        prefix = m.group(1) if m else "000"

    total = len(combos)
    log_dir = ROOT_DIR / args.log_dir / case_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = log_dir / "sweep_runs.csv"

    print(f"\nHephflow sweep: {case_name}")
    print(f"  Target file : {target_file.relative_to(ROOT_DIR)}")
    print(f"  Variables   : {var_names}")
    print(f"  Combinations: {total}")
    print(f"  Mode        : {mode_label}")
    print(f"  Prefix      : {prefix}")
    print(f"  Logs        : {log_dir.relative_to(ROOT_DIR)}")
    print(f"  CSV summary : {csv_path.relative_to(ROOT_DIR)}")
    print()

    # --- Backup originals ---
    original_constants = target_file.read_text(encoding="utf-8")
    original_var_h     = VAR_H.read_text(encoding="utf-8")
    original_output    = output_inc.read_text(encoding="utf-8") if output_inc.is_file() else None

    def restore_originals():
        target_file.write_text(original_constants, encoding="utf-8")
        VAR_H.write_text(original_var_h, encoding="utf-8")
        if original_output is not None and output_inc.is_file():
            output_inc.write_text(original_output, encoding="utf-8")

    failed_runs = []

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_fields = ["run_index", "sim_id", *var_names, "status", "compile_rc", "run_rc"]
        with csv_path.open("w", newline="", encoding="utf-8") as csv_fh:
            writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
            writer.writeheader()

            for i, param_set in enumerate(combos, 1):
                run_id = make_id(param_set)
                compile_rc = ""
                run_rc = ""
                status = "success"

                print(f"{'='*60}")
                print(f"Run {i}/{total}: {param_set}  ->  ID={run_id}")
                print(f"{'='*60}")

                # 1. Patch constants file
                if not args.run_only:
                    try:
                        for var, val in param_set.items():
                            print(f"  [patch] {var} = {val} in {target_file.name}")
                            patch_variable(target_file, var, val, dry_run=args.dry_run)
                    except ValueError as e:
                        print(f"  Error: {e}", file=sys.stderr)
                        restore_originals()
                        return 1

                # 2. Patch ID_SIM in output.inc
                if not args.no_id_patch and output_inc.is_file() and not args.run_only:
                    print(f"  [patch] ID_SIM = \"{run_id}\" in output.inc")
                    try:
                        set_id_sim(output_inc, run_id, dry_run=args.dry_run)
                    except ValueError as e:
                        print(f"  Warning: could not patch ID_SIM: {e}", file=sys.stderr)

                # 3. Set BC_PROBLEM in var.h
                if not args.run_only:
                    print(f"  [patch] BC_PROBLEM = {case_name} in var.h")
                    set_bc_problem(case_name, dry_run=args.dry_run)

                # 4. Compile
                if not args.run_only:
                    compile_log = log_dir / f"{run_id}_compile.log"
                    rc = compile_case(prefix, compile_log, dry_run=args.dry_run)
                    compile_rc = str(rc)
                    if rc != 0:
                        status = "compile_failed"
                        print(f"  Compilation FAILED (exit {rc}) — skipping run", file=sys.stderr)
                        failed_runs.append((run_id, "compile", rc))
                        row = {"run_index": i, "sim_id": run_id, "status": status, "compile_rc": compile_rc, "run_rc": run_rc}
                        row.update(param_set)
                        writer.writerow(row)
                        # Restore constants for next iteration
                        target_file.write_text(original_constants, encoding="utf-8")
                        if original_output is not None:
                            output_inc.write_text(original_output, encoding="utf-8")
                        continue

                # 5. Run
                if not args.compile_only:
                    binary = find_binary(prefix)
                    if binary is None:
                        status = "binary_not_found"
                        print(f"  Error: binary '{prefix}sim_*' not found in {BIN_DIR}", file=sys.stderr)
                        failed_runs.append((run_id, "binary_not_found", -1))
                        row = {"run_index": i, "sim_id": run_id, "status": status, "compile_rc": compile_rc, "run_rc": run_rc}
                        row.update(param_set)
                        writer.writerow(row)
                        continue

                    run_log = log_dir / f"{run_id}_run.log"
                    rc = run_simulation(binary, run_log, dry_run=args.dry_run)
                    run_rc = str(rc)
                    if rc != 0:
                        status = "run_failed"
                        print(f"  Simulation FAILED (exit {rc})", file=sys.stderr)
                        failed_runs.append((run_id, "run", rc))
                    else:
                        print("  Run completed successfully.")
                elif not args.run_only:
                    status = "compiled_only"

                row = {"run_index": i, "sim_id": run_id, "status": status, "compile_rc": compile_rc, "run_rc": run_rc}
                row.update(param_set)
                writer.writerow(row)
                print()

    except KeyboardInterrupt:
        print("\nInterrupted by user. Restoring original files...", file=sys.stderr)
        restore_originals()
        return 130

    finally:
        restore_originals()

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Sweep complete: {total - len(failed_runs)}/{total} runs successful")
    print(f"CSV summary written to: {csv_path}")
    if failed_runs:
        print("Failed runs:")
        for run_id, stage, rc in failed_runs:
            print(f"  {run_id}  [{stage}]  exit={rc}")

    return 1 if failed_runs else 0


if __name__ == "__main__":
    sys.exit(main())
