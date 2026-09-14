#!/usr/bin/env python3
"""Check that local quoted includes resolve with exact filesystem casing."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


SOURCE_SUFFIXES = {".cu", ".cuh", ".h", ".hpp"}
EXCLUDED_DIRS = {".git", "bin", "build", "dist", "__pycache__"}
INCLUDE_RE = re.compile(r"^\s*#\s*include\s*\"([^\"]+)\"")


def relative_parts(root: Path, path: Path) -> list[str] | None:
    # abspath normalizes ./ and ../ while preserving the spelling of components.
    root_absolute = Path(os.path.abspath(root))
    path_absolute = Path(os.path.abspath(path))
    try:
        relative = path_absolute.relative_to(root_absolute)
    except ValueError:
        return None
    return list(relative.parts)


def exact_path(root: Path, path: Path) -> bool:
    """Return whether every path component matches the directory entry exactly."""
    parts = relative_parts(root, path)
    if parts is None:
        return False
    current = root
    for component in parts:
        entries = {entry.name for entry in current.iterdir()}
        if component not in entries:
            return False
        current /= component
    return current.is_file()


def case_insensitive_match(root: Path, path: Path) -> Path | None:
    """Find the existing file reached by path when component case is ignored."""
    parts = relative_parts(root, path)
    if parts is None:
        return None
    current = root
    for component in parts:
        matches = [entry for entry in current.iterdir() if entry.name.casefold() == component.casefold()]
        if len(matches) != 1:
            return None
        current = matches[0]
    return current if current.is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    failures = 0

    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix in SOURCE_SUFFIXES
        and not any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts)
    )
    for source in files:
        for line_number, line in enumerate(source.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            match = INCLUDE_RE.match(line)
            if not match:
                continue
            include = match.group(1)
            if include.startswith("/") or (len(include) > 1 and include[1] == ":"):
                continue

            candidate = source.parent / include
            if exact_path(root, candidate):
                continue

            actual = case_insensitive_match(root, candidate)
            if actual is not None:
                print(f"{source.relative_to(root)}:{line_number}: {include!r} -> {actual.relative_to(root)!s}")
                failures += 1
            else:
                # Bare quoted includes may be found through the compiler's include path.
                matches = []
                for include_root in (root, root / "src"):
                    fallback = include_root / include
                    if exact_path(root, fallback):
                        # The compiler can resolve bare quoted includes through -I roots.
                        matches.append((fallback, True))
                    else:
                        fallback_actual = case_insensitive_match(root, fallback)
                        if fallback_actual is not None:
                            matches.append((fallback_actual, False))
                if matches:
                    actual, exact = matches[0]
                    if not exact:
                        print(f"{source.relative_to(root)}:{line_number}: {include!r} -> {actual.relative_to(root)!s}")
                        failures += 1
                else:
                    # System/third-party headers are outside this local-file audit.
                    continue

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
