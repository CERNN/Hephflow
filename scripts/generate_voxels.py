#!/usr/bin/env python3
"""Generate voxel-coordinate CSV files for Hephflow geometry cases.

The output contains one zero-based ``x,y,z`` solid-node coordinate per line,
which is the format consumed by ``read_xyz_file`` in ``lbmInitialization.cu``.
Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TextIO


Coordinate = tuple[int, int, int]
Point = tuple[float, float, float]
Triangle = tuple[Point, Point, Point]


class GeometryError(ValueError):
    """Report invalid geometry parameters."""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def centered_start(extent: int, size: int) -> int:
    if size > extent:
        raise GeometryError(f"size {size} exceeds domain extent {extent}")
    return (extent - size) // 2


def cube_coordinates(args: argparse.Namespace) -> Iterator[Coordinate]:
    nx, ny, nz = args.domain
    size_x, size_y, size_z = args.size
    start_x = centered_start(nx, size_x) if args.origin is None else args.origin[0]
    start_y = centered_start(ny, size_y) if args.origin is None else args.origin[1]
    start_z = centered_start(nz, size_z) if args.origin is None else args.origin[2]
    validate_box((start_x, start_y, start_z), args.size, args.domain)

    for y in range(start_y, start_y + size_y):
        for x in range(start_x, start_x + size_x):
            for z in range(start_z, start_z + size_z):
                yield x, y, z


def mounted_cube_coordinates(args: argparse.Namespace) -> Iterator[Coordinate]:
    nx, ny, nz = args.domain
    size = args.size
    start_x = centered_start(nx, size)
    start_z = centered_start(nz, size)
    start_y = args.base_y + 1
    validate_box((start_x, start_y, start_z), (size, size, size), args.domain)

    plate_start_x = start_x - args.plate_padding
    plate_start_z = start_z - args.plate_padding
    plate_size = size + 2 * args.plate_padding
    validate_box(
        (plate_start_x, args.base_y, plate_start_z),
        (plate_size, 1, plate_size),
        args.domain,
    )

    for y in range(start_y, start_y + size):
        for x in range(start_x, start_x + size):
            for z in range(start_z, start_z + size):
                yield x, y, z
    for x in range(plate_start_x, plate_start_x + plate_size):
        for z in range(plate_start_z, plate_start_z + plate_size):
            yield x, args.base_y, z


def parallel_plate_coordinates(args: argparse.Namespace) -> Iterator[Coordinate]:
    nx, ny, nz = args.domain
    extents = {"x": nx, "y": ny, "z": nz}
    positions = (0, extents[args.axis] - 1)

    if args.axis == "x":
        for x in positions:
            for y in range(ny):
                for z in range(nz):
                    yield x, y, z
    elif args.axis == "y":
        for y in positions:
            for x in range(nx):
                for z in range(nz):
                    yield x, y, z
    else:
        for z in positions:
            for x in range(nx):
                for y in range(ny):
                    yield x, y, z


def sphere_coordinates(args: argparse.Namespace) -> Iterator[Coordinate]:
    nx, ny, nz = args.domain
    center = args.center or ((nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2)
    radius_squared = args.diameter * args.diameter / 4

    for x in range(nx):
        dx_squared = (x - center[0]) ** 2
        for y in range(ny):
            dxy_squared = dx_squared + (y - center[1]) ** 2
            for z in range(nz):
                if dxy_squared + (z - center[2]) ** 2 <= radius_squared:
                    yield x, y, z


def read_stl(path: Path) -> list[Triangle]:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise GeometryError(f"cannot read STL file {path}: {error}") from error

    if len(data) >= 84:
        facet_count = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + 50 * facet_count:
            triangles = []
            for offset in range(84, len(data), 50):
                values = struct.unpack_from("<12fH", data, offset)
                triangles.append(
                    (
                        (values[3], values[4], values[5]),
                        (values[6], values[7], values[8]),
                        (values[9], values[10], values[11]),
                    )
                )
            return validate_triangles(triangles, path)

    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as error:
        raise GeometryError(f"{path} is neither a valid binary nor ASCII STL file") from error

    vertices: list[Point] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].lower() != "vertex":
            continue
        if len(fields) != 4:
            raise GeometryError(f"invalid STL vertex at {path}:{line_number}")
        try:
            vertices.append(tuple(map(float, fields[1:4])))
        except ValueError as error:
            raise GeometryError(f"invalid STL vertex at {path}:{line_number}") from error
    if len(vertices) % 3:
        raise GeometryError(f"ASCII STL {path} has an incomplete triangle")
    triangles = [tuple(vertices[index : index + 3]) for index in range(0, len(vertices), 3)]
    return validate_triangles(triangles, path)


def validate_triangles(triangles: list[Triangle], path: Path) -> list[Triangle]:
    if not triangles:
        raise GeometryError(f"STL file {path} contains no triangles")
    for triangle in triangles:
        if not all(math.isfinite(value) for vertex in triangle for value in vertex):
            raise GeometryError(f"STL file {path} contains a non-finite coordinate")
    return triangles


def mesh_bounds(triangles: Iterable[Triangle]) -> tuple[Point, Point]:
    vertices = [vertex for triangle in triangles for vertex in triangle]
    minimum = tuple(min(vertex[axis] for vertex in vertices) for axis in range(3))
    maximum = tuple(max(vertex[axis] for vertex in vertices) for axis in range(3))
    return minimum, maximum


def check_watertight(triangles: list[Triangle]) -> None:
    minimum, maximum = mesh_bounds(triangles)
    diagonal = math.sqrt(sum((high - low) ** 2 for low, high in zip(minimum, maximum)))
    tolerance = max(diagonal * 1e-9, 1e-12)

    def vertex_key(vertex: Point) -> tuple[int, int, int]:
        return tuple(round(value / tolerance) for value in vertex)

    edge_counts: Counter[tuple[tuple[int, int, int], tuple[int, int, int]]] = Counter()
    for triangle in triangles:
        keys = [vertex_key(vertex) for vertex in triangle]
        for start, end in ((keys[0], keys[1]), (keys[1], keys[2]), (keys[2], keys[0])):
            edge_counts[tuple(sorted((start, end)))] += 1
    invalid_edges = sum(count != 2 for count in edge_counts.values())
    if invalid_edges:
        raise GeometryError(
            f"STL mesh is not watertight: {invalid_edges} edges do not belong to exactly two facets; "
            "repair the mesh or pass --skip-watertight-check"
        )


def fit_mesh(
    triangles: list[Triangle], domain: Coordinate, padding: int
) -> list[Triangle]:
    if any(2 * padding >= extent for extent in domain):
        raise GeometryError("STL padding must leave at least one voxel on every axis")
    minimum, maximum = mesh_bounds(triangles)
    spans = tuple(high - low for low, high in zip(minimum, maximum))
    if any(span <= 0 for span in spans):
        raise GeometryError("STL mesh must have nonzero extent on all three axes")
    scale = min((extent - 2 * padding) / span for extent, span in zip(domain, spans))
    source_center = tuple((low + high) / 2 for low, high in zip(minimum, maximum))
    target_center = tuple(extent / 2 for extent in domain)

    return [
        tuple(
            tuple(
                (vertex[axis] - source_center[axis]) * scale + target_center[axis]
                for axis in range(3)
            )
            for vertex in triangle
        )
        for triangle in triangles
    ]


def ray_intersection_x(triangle: Triangle, y: float, z: float) -> float | None:
    first, second, third = triangle
    denominator = (second[1] - third[1]) * (first[2] - third[2]) + (
        third[2] - second[2]
    ) * (first[1] - third[1])
    if abs(denominator) < 1e-15:
        return None
    first_weight = (
        (second[1] - third[1]) * (z - third[2])
        + (third[2] - second[2]) * (y - third[1])
    ) / denominator
    second_weight = (
        (third[1] - first[1]) * (z - third[2])
        + (first[2] - third[2]) * (y - third[1])
    ) / denominator
    third_weight = 1.0 - first_weight - second_weight
    tolerance = 1e-10
    if min(first_weight, second_weight, third_weight) < -tolerance:
        return None
    return (
        first_weight * first[0]
        + second_weight * second[0]
        + third_weight * third[0]
    )


def voxelize_mesh(triangles: list[Triangle], domain: Coordinate) -> Iterator[Coordinate]:
    nx, ny, nz = domain
    intersections: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
    jitter = 1e-7

    for triangle in triangles:
        minimum_y = min(vertex[1] for vertex in triangle)
        maximum_y = max(vertex[1] for vertex in triangle)
        minimum_z = min(vertex[2] for vertex in triangle)
        maximum_z = max(vertex[2] for vertex in triangle)
        start_y = max(0, math.floor(minimum_y - 0.5))
        end_y = min(ny - 1, math.ceil(maximum_y - 0.5))
        start_z = max(0, math.floor(minimum_z - 0.5))
        end_z = min(nz - 1, math.ceil(maximum_z - 0.5))
        for y in range(start_y, end_y + 1):
            sample_y = y + 0.5 + jitter
            for z in range(start_z, end_z + 1):
                sample_z = z + 0.5 + 0.61803398875 * jitter
                intersection = ray_intersection_x(triangle, sample_y, sample_z)
                if intersection is not None:
                    intersections[y, z].append(intersection)

    odd_scanlines = 0
    for (y, z), values in sorted(intersections.items()):
        values.sort()
        unique_values: list[float] = []
        for value in values:
            if not unique_values or abs(value - unique_values[-1]) > 1e-7:
                unique_values.append(value)
        if len(unique_values) % 2:
            odd_scanlines += 1
            continue
        for index in range(0, len(unique_values), 2):
            lower, upper = unique_values[index : index + 2]
            start_x = max(0, math.ceil(lower - 0.5 - 1e-9))
            end_x = min(nx - 1, math.floor(upper - 0.5 + 1e-9))
            for x in range(start_x, end_x + 1):
                yield x, y, z
    if odd_scanlines:
        raise GeometryError(
            f"STL voxelization encountered {odd_scanlines} scanlines with an odd number of "
            "surface crossings; check the mesh for overlaps or self-intersections"
        )


def stl_coordinates(args: argparse.Namespace) -> Iterator[Coordinate]:
    triangles = read_stl(args.input)
    if not args.skip_watertight_check:
        check_watertight(triangles)
    if args.fit:
        triangles = fit_mesh(triangles, args.domain, args.padding)
    yield from voxelize_mesh(triangles, args.domain)


def validate_box(origin: Coordinate, size: Coordinate, domain: Coordinate) -> None:
    axis_names = "xyz"
    for axis, start, length, extent in zip(axis_names, origin, size, domain):
        if start < 0 or start + length > extent:
            raise GeometryError(
                f"{axis}-range [{start}, {start + length}) is outside domain [0, {extent})"
            )


def write_coordinates(coordinates: Iterable[Coordinate], stream: TextIO) -> int:
    count = 0
    for x, y, z in coordinates:
        stream.write(f"{x},{y},{z}\n")
        count += 1
    return count


def write_output(args: argparse.Namespace, coordinates: Iterable[Coordinate]) -> int:
    if args.output == Path("-"):
        count = write_coordinates(coordinates, sys.stdout)
        print(f"wrote {count} solid nodes to standard output", file=sys.stderr)
        return count

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="ascii", newline="\n") as stream:
            count = write_coordinates(coordinates, stream)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"wrote {count} solid nodes to {output}")
    return count


def add_domain_argument(parser: argparse.ArgumentParser, default: tuple[int, int, int]) -> None:
    parser.add_argument(
        "--domain",
        nargs=3,
        type=positive_int,
        default=default,
        metavar=("NX", "NY", "NZ"),
        help=f"lattice dimensions (default: {' '.join(map(str, default))})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        metavar="CSV",
        help="output CSV path, or - for standard output",
    )
    subparsers = parser.add_subparsers(dest="shape", required=True)

    cube = subparsers.add_parser("cube", help="generate a centered or positioned cuboid")
    add_domain_argument(cube, (256, 256, 256))
    cube.add_argument(
        "--size",
        nargs=3,
        type=positive_int,
        default=(32, 32, 32),
        metavar=("LX", "LY", "LZ"),
        help="cuboid size (default: 32 32 32)",
    )
    cube.add_argument(
        "--origin",
        nargs=3,
        type=nonnegative_int,
        metavar=("X", "Y", "Z"),
        help="lowest cuboid coordinate; omit to center it",
    )
    cube.set_defaults(generate=cube_coordinates)

    mounted = subparsers.add_parser(
        "mounted-cube", help="generate a centered cube mounted above a square base plate"
    )
    add_domain_argument(mounted, (256, 256, 256))
    mounted.add_argument("--size", type=positive_int, default=64, help="cube side (default: 64)")
    mounted.add_argument(
        "--base-y", type=nonnegative_int, default=0, help="base plate y coordinate (default: 0)"
    )
    mounted.add_argument(
        "--plate-padding",
        type=nonnegative_int,
        default=1,
        help="base extension beyond each cube edge (default: 1)",
    )
    mounted.set_defaults(generate=mounted_cube_coordinates)

    plates = subparsers.add_parser("parallel-plates", help="generate two opposite domain faces")
    add_domain_argument(plates, (32, 32, 32))
    plates.add_argument(
        "--axis", choices=("x", "y", "z"), default="y", help="face-normal axis (default: y)"
    )
    plates.set_defaults(generate=parallel_plate_coordinates)

    sphere = subparsers.add_parser("sphere", help="generate solid nodes inside a sphere")
    add_domain_argument(sphere, (256, 256, 256))
    sphere.add_argument(
        "--diameter", type=positive_int, default=32, help="sphere diameter in lattice units (default: 32)"
    )
    sphere.add_argument(
        "--center",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="sphere center; omit to use the domain center",
    )
    sphere.set_defaults(generate=sphere_coordinates)

    stl = subparsers.add_parser("stl", help="voxelize a closed binary or ASCII STL mesh")
    stl.add_argument("input", type=Path, metavar="STL", help="input binary or ASCII STL file")
    add_domain_argument(stl, (256, 256, 256))
    stl.add_argument(
        "--padding",
        type=nonnegative_int,
        default=1,
        help="minimum empty border when fitting the mesh (default: 1)",
    )
    stl.add_argument(
        "--no-fit",
        action="store_false",
        dest="fit",
        help="use STL coordinates directly instead of fitting and centering the mesh",
    )
    stl.add_argument(
        "--skip-watertight-check",
        action="store_true",
        help="attempt conversion even when mesh edge topology is not closed",
    )
    stl.set_defaults(generate=stl_coordinates)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        write_output(args, args.generate(args))
    except (GeometryError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
