#!/usr/bin/env python3
"""Convert Hephflow raw binary field snapshots to legacy VTK files.

The converter uses only the Python standard library.  It reads the simulation
metadata and field filenames produced by ``saveData.cu``, groups known vector
and symmetric-tensor components, and writes point-centered structured data.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from array import array
from pathlib import Path
from typing import BinaryIO, Iterable


INFO_KEYS = {"Simulation ID", "Precision", "NX", "NY", "NZ", "NZ_TOTAL", "BX", "BY", "BZ"}
VECTOR_GROUPS = (
    ("velocity", ("ux", "uy", "uz"), 3.0),
    ("forces", ("fx", "fy", "fz"), 1.0),
    ("local_forces", ("local_fx", "local_fy", "local_fz"), 1.0),
)
TENSOR_GROUPS = (
    ("Aij", ("Axx", "Axy", "Axz", "Ayy", "Ayz", "Azz")),
    ("source_Gij", ("source_Gxx", "source_Gxy", "source_Gxz", "source_Gyy", "source_Gyz", "source_Gzz")),
)
CHUNK_VALUES = 1 << 18
REQUIRED_FIELDS = {"rho", "ux", "uy", "uz"}


class ConversionError(RuntimeError):
    """Report an invalid or incomplete Hephflow output set."""


def parse_info(directory: Path) -> dict[str, str]:
    info_files = sorted(directory.glob("*info*.txt"))
    if len(info_files) != 1:
        raise ConversionError(f"expected one *info*.txt file in {directory}, found {len(info_files)}")

    values: dict[str, str] = {}
    line_pattern = re.compile(r"^\s*([^:]+):\s*(\S+)\s*$")
    with info_files[0].open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = line_pattern.match(line)
            if match and match.group(1).strip() in INFO_KEYS:
                values[match.group(1).strip()] = match.group(2)

    required = {"Simulation ID", "Precision", "NX", "NY", "NZ"}
    missing = sorted(required - values.keys())
    if missing:
        raise ConversionError(f"{info_files[0]} is missing: {', '.join(missing)}")
    if values["Precision"] not in {"float", "double"}:
        raise ConversionError(f"unsupported precision {values['Precision']!r}")
    return values


def discover_snapshots(directory: Path, simulation_id: str) -> dict[int, dict[str, Path]]:
    pattern = re.compile(rf"^{re.escape(simulation_id)}_(.+?)(\d{{6,}})\.bin$")
    snapshots: dict[int, dict[str, Path]] = {}
    for path in sorted(directory.glob("*.bin")):
        match = pattern.match(path.name)
        if match:
            field, step_text = match.groups()
            snapshots.setdefault(int(step_text), {})[field] = path
    if not snapshots:
        raise ConversionError(f"no {simulation_id}_<field><step>.bin snapshots found in {directory}")
    return snapshots


def resolve_input(path: Path) -> Path:
    if path.is_dir():
        return path.resolve()
    repository_relative = Path(__file__).resolve().parents[1] / "bin" / path
    if repository_relative.is_dir():
        return repository_relative.resolve()
    raise ConversionError(f"input directory does not exist: {path}")


def select_steps(available: Iterable[int], requested: list[int] | None) -> list[int]:
    available_set = set(available)
    if not requested:
        return sorted(available_set)
    missing = sorted(set(requested) - available_set)
    if missing:
        raise ConversionError(f"requested steps not found: {', '.join(map(str, missing))}")
    return sorted(set(requested))


def read_chunk(stream: BinaryIO, typecode: str, count: int) -> array:
    values = array(typecode)
    try:
        values.fromfile(stream, count)
    except EOFError as error:
        raise ConversionError("binary field ended before the expected point count") from error
    return values


def write_big_endian(output: BinaryIO, values: array) -> None:
    if sys.byteorder == "little":
        values.byteswap()
    output.write(values.tobytes())


def write_scalar(output: BinaryIO, name: str, path: Path, point_count: int, typecode: str) -> None:
    vtk_type = "unsigned_int" if typecode == "I" else ("float" if typecode == "f" else "double")
    output.write(f"SCALARS {name} {vtk_type} 1\nLOOKUP_TABLE default\n".encode("ascii"))
    with path.open("rb") as source:
        remaining = point_count
        while remaining:
            count = min(remaining, CHUNK_VALUES)
            write_big_endian(output, read_chunk(source, typecode, count))
            remaining -= count
    output.write(b"\n")


def write_vector(
    output: BinaryIO,
    name: str,
    paths: tuple[Path, Path, Path],
    point_count: int,
    typecode: str,
    scale: float,
) -> None:
    vtk_type = "float" if typecode == "f" else "double"
    output.write(f"VECTORS {name} {vtk_type}\n".encode("ascii"))
    streams = tuple(path.open("rb") for path in paths)
    try:
        remaining = point_count
        while remaining:
            count = min(remaining, CHUNK_VALUES)
            components = [read_chunk(stream, typecode, count) for stream in streams]
            packed = array(typecode)
            if scale == 1.0:
                packed.extend(value for values in zip(*components) for value in values)
            else:
                packed.extend(value / scale for values in zip(*components) for value in values)
            write_big_endian(output, packed)
            remaining -= count
    finally:
        for stream in streams:
            stream.close()
    output.write(b"\n")


def write_tensor(
    output: BinaryIO,
    name: str,
    paths: tuple[Path, Path, Path, Path, Path, Path],
    point_count: int,
    typecode: str,
) -> None:
    vtk_type = "float" if typecode == "f" else "double"
    output.write(f"TENSORS {name} {vtk_type}\n".encode("ascii"))
    streams = tuple(path.open("rb") for path in paths)
    try:
        remaining = point_count
        while remaining:
            count = min(remaining, CHUNK_VALUES // 3)
            components = [read_chunk(stream, typecode, count) for stream in streams]
            packed = array(typecode)
            for xx, xy, xz, yy, yz, zz in zip(*components):
                packed.extend((xx, xy, xz, xy, yy, yz, xz, yz, zz))
            write_big_endian(output, packed)
            remaining -= count
    finally:
        for stream in streams:
            stream.close()
    output.write(b"\n")


def validate_fields(fields: dict[str, Path], selected: set[str], point_count: int, item_size: int) -> None:
    for name in selected:
        path = fields[name]
        expected = point_count * (4 if name == "bc" else item_size)
        if path.stat().st_size < expected:
            raise ConversionError(f"{path} is {path.stat().st_size} bytes; expected at least {expected}")


def convert_step(
    destination: Path,
    fields: dict[str, Path],
    selected: set[str],
    dimensions: tuple[int, int, int],
    precision: str,
) -> None:
    nx, ny, nz = dimensions
    point_count = nx * ny * nz
    typecode = "f" if precision == "float" else "d"
    validate_fields(fields, selected, point_count, array(typecode).itemsize)

    temporary = destination.with_name(destination.name + ".tmp")
    consumed: set[str] = set()
    try:
        with temporary.open("wb") as output:
            output.write(
                ("# vtk DataFile Version 3.0\n"
                 "Hephflow binary conversion\n"
                 "BINARY\n"
                 "DATASET STRUCTURED_POINTS\n"
                 f"DIMENSIONS {nx} {ny} {nz}\n"
                 "ORIGIN 0 0 0\n"
                 "SPACING 1 1 1\n"
                 f"POINT_DATA {point_count}\n").encode("ascii")
            )

            for name, components, scale in VECTOR_GROUPS:
                if set(components) <= selected:
                    write_vector(output, name, tuple(fields[field] for field in components), point_count, typecode, scale)
                    consumed.update(components)
            for name, components in TENSOR_GROUPS:
                if set(components) <= selected:
                    write_tensor(output, name, tuple(fields[field] for field in components), point_count, typecode)
                    consumed.update(components)
            for name in sorted(selected - consumed):
                write_scalar(output, name, fields[name], point_count, "I" if name == "bc" else typecode)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="output directory, or its path relative to bin/")
    parser.add_argument("--output-dir", type=Path, help="destination directory (default: input directory)")
    parser.add_argument("--step", type=int, action="append", help="convert only this step; may be repeated")
    parser.add_argument("--field", action="append", help="include only this field; may be repeated")
    parser.add_argument("--force", action="store_true", help="replace existing converted VTK files")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        input_directory = resolve_input(args.input)
        info = parse_info(input_directory)
        simulation_id = info["Simulation ID"]
        snapshots = discover_snapshots(input_directory, simulation_id)
        steps = select_steps(snapshots, args.step)
        dimensions = tuple(int(info[key]) for key in ("NX", "NY", "NZ"))
        if any(size <= 0 for size in dimensions):
            raise ConversionError(f"invalid dimensions: {dimensions}")

        output_directory = (args.output_dir or input_directory).resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        for step in steps:
            fields = snapshots[step]
            missing_required = sorted(REQUIRED_FIELDS - fields.keys())
            if missing_required:
                raise ConversionError(f"step {step} is incomplete; missing fields: {', '.join(missing_required)}")
            selected = set(args.field or fields)
            missing = sorted(selected - fields.keys())
            if missing:
                raise ConversionError(f"step {step} is missing fields: {', '.join(missing)}")
            destination = output_directory / f"{simulation_id}_vtk_from_bin{step:06d}.vtk"
            if destination.exists() and not args.force:
                raise ConversionError(f"output exists (use --force to replace it): {destination}")
            print(f"Converting step {step}: {', '.join(sorted(selected))}")
            convert_step(destination, fields, selected, dimensions, info["Precision"])
            print(f"Wrote {destination}")
        return 0
    except (ConversionError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
