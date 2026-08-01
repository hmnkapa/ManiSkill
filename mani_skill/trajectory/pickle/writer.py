"""Atomic pickle and LZMA-pickle I/O."""

from __future__ import annotations

import lzma
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

from .validator import validate_trajectory


def _compression(path: Path) -> bool:
    name = path.name
    if name.endswith(".pkl.xz"):
        return True
    if name.endswith(".pkl"):
        return False
    raise ValueError("Pickle output path must end in .pkl or .pkl.xz")


def write_trajectory(
    trajectory: Any,
    path: str | os.PathLike,
    *,
    overwrite: bool = False,
    validate: bool = True,
) -> Path:
    """Validate and atomically write one trajectory with highest protocol."""

    path = Path(path)
    compressed = _compression(path)
    if validate:
        validate_trajectory(trajectory)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing pickle: {path}")

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.tmp-"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as raw_file:
            if compressed:
                with lzma.LZMAFile(raw_file, mode="wb") as pickle_file:
                    pickle.dump(
                        trajectory, pickle_file, protocol=pickle.HIGHEST_PROTOCOL
                    )
            else:
                pickle.dump(trajectory, raw_file, protocol=pickle.HIGHEST_PROTOCOL)
            raw_file.flush()
            os.fsync(raw_file.fileno())

        # Recheck immediately before replacement to retain the default
        # no-overwrite contract if a target appeared during serialization.
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing pickle: {path}")
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return path


def read_trajectory(path: str | os.PathLike) -> Any:
    """Read either supported pickle suffix."""

    path = Path(path)
    compressed = _compression(path)
    opener = lzma.open if compressed else open
    with opener(path, "rb") as pickle_file:
        return pickle.load(pickle_file)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    directory_fd = os.open(directory, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


__all__ = ["read_trajectory", "write_trajectory"]
