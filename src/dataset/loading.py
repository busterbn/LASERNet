from __future__ import annotations

import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class Plane(str, Enum):
    """Enumeration of orthogonal planes that can be extracted from the point cloud."""

    XY = "xy"
    YZ = "yz"
    XZ = "xz"


class FieldType(str, Enum):
    """Dataset field options that can be fetched for a given timestep."""

    TEMPERATURE = "temperature"
    MICROSTRUCTURE = "microstructure"


AXIS_TO_COLUMN: Dict[str, str] = {"x": "Points:0", "y": "Points:1", "z": "Points:2"}
MICROSTRUCTURE_COLUMNS: Tuple[str, ...] = (
    "ipf_x:0",
    "ipf_x:1",
    "ipf_x:2",
    "ipf_y:0",
    "ipf_y:1",
    "ipf_y:2",
    "ipf_z:0",
    "ipf_z:1",
    "ipf_z:2",
    "ori_inds",
)
TEMPERATURE_COLUMNS: Tuple[str, ...] = ("T",)
TIMESTEP_PATTERN = re.compile(r"(\d+)(?!.*\d)")


def _plane_axes(plane: Plane) -> Tuple[str, str, str]:
    if plane == Plane.XY:
        return "x", "y", "z"
    if plane == Plane.YZ:
        return "z", "y", "x"
    if plane == Plane.XZ:
        return "x", "z", "y"
    raise ValueError(f"Unsupported plane {plane!r}")


@dataclass(frozen=True)
class PlaneSpec:
    plane: Plane
    coordinate: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinate", float(self.coordinate))


@dataclass(frozen=True)
class PlaneSelection:
    plane: Plane
    coordinate: Optional[float] = None
    index: Optional[int] = None

    def resolve(self, axis_values: np.ndarray) -> PlaneSpec:
        if axis_values.size == 0:
            raise ValueError("Axis value array is empty; cannot resolve plane selection")
        if self.coordinate is not None:
            return PlaneSpec(self.plane, float(self.coordinate))
        if self.index is not None:
            idx = self.index
            if idx < 0:
                idx = axis_values.size + idx
            if idx < 0 or idx >= axis_values.size:
                raise IndexError(
                    f"Plane index {self.index} out of range for axis with {axis_values.size} values"
                )
            return PlaneSpec(self.plane, float(axis_values[idx]))
        mid = axis_values.size // 2
        return PlaneSpec(self.plane, float(axis_values[mid]))


@dataclass(frozen=True)
class LaserNetFrame:
    data: torch.Tensor
    mask: torch.Tensor
    plane: PlaneSpec


class LaserNetDataRepository:
    """Manages discovery and cached access to LASERNet raw timesteps."""

    def __init__(
        self,
        data_dir: Optional[Union[str, os.PathLike[str]]] = None,
        pattern: str = "Alldata_withpoints_*.csv",
        chunk_size: int = 500_000,
        frame_cache_size: int = 8,
        axis_scan_files: Optional[int] = 1,
    ) -> None:
        self.data_dir = self._resolve_data_dir(data_dir)
        self.pattern = pattern
        self.chunk_size = chunk_size
        self.frame_cache_size = max(frame_cache_size, 0)
        self.file_paths = self._discover_files()
        if not self.file_paths:
            raise FileNotFoundError(
                f"No CSV files found under {self.data_dir} matching pattern {self.pattern!r}"
            )
        self.axis_values, self.axis_lookup, self.axis_tol = self._build_axis_metadata(axis_scan_files)
        self._frame_cache: OrderedDict[Tuple[str, FieldType, PlaneSpec], LaserNetFrame]
        self._frame_cache = OrderedDict()

    def _resolve_data_dir(
        self, data_dir: Optional[Union[str, os.PathLike[str]]]
    ) -> Path:
        if data_dir is not None:
            candidate = Path(data_dir).expanduser()
            if not candidate.exists():
                raise FileNotFoundError(f"Provided data directory {candidate} does not exist")
            return candidate
        root_env = os.environ.get("BLACKHOLE")
        if not root_env:
            raise ValueError(
                "BLACKHOLE environment variable is not set and no data path was provided"
            )
        for name in ("data", "Data"):
            candidate = Path(root_env) / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not locate dataset under $BLACKHOLE/data or $BLACKHOLE/Data (root {root_env})"
        )

    def _timestep_key(self, path: Path) -> Tuple[int, Union[int, str]]:
        match = TIMESTEP_PATTERN.search(path.stem)
        if match:
            return 0, int(match.group(1))
        return 1, path.name

    def _discover_files(self) -> List[Path]:
        candidates = list(self.data_dir.glob(self.pattern))
        candidates.sort(key=self._timestep_key)
        return candidates

    def _build_axis_metadata(
        self, axis_scan_files: Optional[int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[float, int]], Dict[str, float]]:
        axis_cols = list(AXIS_TO_COLUMN.values())
        uniques: Dict[str, set[float]] = {axis: set() for axis in AXIS_TO_COLUMN.keys()}
        paths = self.file_paths
        if axis_scan_files is not None and axis_scan_files > 0:
            paths = paths[:axis_scan_files]
        for csv_path in paths:
            for chunk in pd.read_csv(csv_path, usecols=axis_cols, chunksize=self.chunk_size):
                for axis, col in AXIS_TO_COLUMN.items():
                    uniques[axis].update(map(float, chunk[col].unique()))
        axis_values: Dict[str, np.ndarray] = {}
        axis_lookup: Dict[str, Dict[float, int]] = {}
        axis_tol: Dict[str, float] = {}
        for axis, values in uniques.items():
            if not values:
                raise ValueError(f"No coordinate values collected for axis {axis!r}")
            sorted_vals = np.array(sorted(values), dtype=np.float64)
            axis_values[axis] = sorted_vals
            axis_lookup[axis] = {float(v): idx for idx, v in enumerate(sorted_vals)}
            if sorted_vals.size > 1:
                diffs = np.diff(sorted_vals)
                diffs = diffs[diffs > 0]
                tol = float(diffs.min()) * 0.51 if diffs.size else 1e-9
            else:
                tol = 1e-9
            axis_tol[axis] = tol
        return axis_values, axis_lookup, axis_tol

    @property
    def num_timesteps(self) -> int:
        return len(self.file_paths)

    def field_columns(self, field: FieldType) -> Tuple[str, ...]:
        if field == FieldType.TEMPERATURE:
            return TEMPERATURE_COLUMNS
        if field == FieldType.MICROSTRUCTURE:
            return MICROSTRUCTURE_COLUMNS
        raise ValueError(f"Unsupported field {field!r}")

    def resolve_plane(self, selection: Union[PlaneSelection, PlaneSpec]) -> PlaneSpec:
        if isinstance(selection, PlaneSpec):
            return selection
        _, _, fixed_axis = _plane_axes(selection.plane)
        axis_values = self.axis_values[fixed_axis]
        return selection.resolve(axis_values)

    def split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, List[int]]:
        ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)
        if np.any(ratios < 0):
            raise ValueError("Split ratios must be non-negative")
        if ratios.sum() == 0:
            raise ValueError("At least one split ratio must be positive")
        ratios = ratios / ratios.sum()
        total = self.num_timesteps
        counts = np.floor(ratios * total).astype(int)
        shortfall = total - counts.sum()
        for i in range(shortfall):
            counts[i % counts.size] += 1
        splits: Dict[str, List[int]] = {}
        start = 0
        for name, count in zip(("train", "val", "test"), counts.tolist()):
            end = min(total, start + count)
            splits[name] = list(range(start, end)) if end > start else []
            start = end
        for name, ratio in zip(("train", "val", "test"), ratios.tolist()):
            if ratio > 0 and not splits[name]:
                raise ValueError(f"Split {name} received ratio {ratio} but has no timesteps")
        return splits

    def get_frame(
        self,
        timestep: Union[int, Path],
        field: FieldType,
        plane: PlaneSpec,
    ) -> LaserNetFrame:
        if isinstance(timestep, Path):
            path = timestep
        else:
            path = self.file_paths[timestep]
        return self._get_cached_frame(path, field, plane)

    def _get_cached_frame(
        self,
        path: Path,
        field: FieldType,
        plane: PlaneSpec,
    ) -> LaserNetFrame:
        key = (str(path), field, plane)
        if self.frame_cache_size == 0:
            return self._load_frame(path, field, plane)
        cached = self._frame_cache.get(key)
        if cached is not None:
            self._frame_cache.move_to_end(key)
            return cached
        frame = self._load_frame(path, field, plane)
        self._frame_cache[key] = frame
        if len(self._frame_cache) > self.frame_cache_size:
            self._frame_cache.popitem(last=False)
        return frame

    def _load_frame(
        self,
        path: Path,
        field: FieldType,
        plane: PlaneSpec,
    ) -> LaserNetFrame:
        width_axis, height_axis, fixed_axis = _plane_axes(plane.plane)
        width_vals = self.axis_values[width_axis]
        height_vals = self.axis_values[height_axis]
        width_lookup = self.axis_lookup[width_axis]
        height_lookup = self.axis_lookup[height_axis]
        tol = self.axis_tol[fixed_axis]
        width_col = AXIS_TO_COLUMN[width_axis]
        height_col = AXIS_TO_COLUMN[height_axis]
        fixed_col = AXIS_TO_COLUMN[fixed_axis]
        columns = list(self.field_columns(field))
        usecols = columns + [width_col, height_col, fixed_col]
        reader = pd.read_csv(path, usecols=usecols, chunksize=self.chunk_size)
        channels = len(columns)
        frame_tensor = torch.full(
            (channels, height_vals.size, width_vals.size),
            float("nan"),
            dtype=torch.float32,
        )
        mask_tensor = torch.zeros((height_vals.size, width_vals.size), dtype=torch.bool)
        found = False
        for chunk in reader:
            plane_chunk = chunk[np.isclose(chunk[fixed_col], plane.coordinate, atol=tol)]
            if plane_chunk.empty:
                continue
            width_idx = plane_chunk[width_col].map(width_lookup)
            height_idx = plane_chunk[height_col].map(height_lookup)
            if width_idx.isna().any() or height_idx.isna().any():
                raise ValueError(
                    f"Encountered unseen coordinates when loading {path.name}; "
                    "consider rebuilding axis metadata with axis_scan_files=None"
                )
            x_idx = width_idx.to_numpy(dtype=np.int64)
            y_idx = height_idx.to_numpy(dtype=np.int64)
            values_np = plane_chunk[columns].to_numpy(dtype=np.float32, copy=False)
            if values_np.ndim == 1:
                values_np = values_np[:, np.newaxis]
            values_np = np.ascontiguousarray(values_np.T)
            values_t = torch.from_numpy(values_np)
            x_tensor = torch.from_numpy(x_idx)
            y_tensor = torch.from_numpy(y_idx)
            frame_tensor[:, y_tensor, x_tensor] = values_t
            mask_tensor[y_tensor, x_tensor] = True
            found = True
        if not found:
            raise ValueError(
                f"No samples found for plane {plane.plane.value} at {plane.coordinate} in file {path.name}"
            )
        return LaserNetFrame(frame_tensor, mask_tensor, plane)


class LaserNetSequenceDataset(Dataset):
    """Temporal dataset that returns context/target tensors for LASERNet."""

    def __init__(
        self,
        repository: LaserNetDataRepository,
        timesteps: Sequence[int],
        sequence_length: int,
        field: FieldType,
        plane: Union[PlaneSelection, PlaneSpec],
        target_offset: int = 1,
        return_target: bool = True,
        padding_value: Optional[float] = 0.0,
        frame_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        sample_transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
        copy_tensors: bool = True,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.repository = repository
        self.timesteps = sorted(int(t) for t in timesteps)
        if not self.timesteps:
            raise ValueError("timesteps sequence cannot be empty")
        self.sequence_length = sequence_length
        self.field = field
        self.return_target = return_target
        self.target_offset = target_offset if return_target else 0
        if self.return_target and self.target_offset <= 0:
            raise ValueError("target_offset must be >= 1 when return_target is True")
        self.padding_value = padding_value
        self.frame_transform = frame_transform
        self.target_transform = target_transform
        self.sample_transform = sample_transform
        self.copy_tensors = copy_tensors
        self.plane_spec = repository.resolve_plane(plane)
        self.width_axis, self.height_axis, _ = _plane_axes(self.plane_spec.plane)
        self.width_coords = torch.from_numpy(repository.axis_values[self.width_axis].astype(np.float32))
        self.height_coords = torch.from_numpy(repository.axis_values[self.height_axis].astype(np.float32))
        min_length = self.sequence_length + self.target_offset
        if len(self.timesteps) < min_length:
            raise ValueError(
                f"Not enough timesteps ({len(self.timesteps)}) to satisfy sequence_length="
                f"{self.sequence_length} with target_offset={self.target_offset}"
            )
        self.num_items = len(self.timesteps) - min_length + 1

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < 0 or index >= self.num_items:
            raise IndexError(index)
        start = index
        end = index + self.sequence_length
        context_indices = self.timesteps[start:end]
        context_frames = [self._prepare_frame(timestep) for timestep in context_indices]
        context_data = torch.stack([item[0] for item in context_frames], dim=0)
        context_masks = torch.stack([item[1] for item in context_frames], dim=0)
        sample: Dict[str, torch.Tensor] = {
            "context": context_data,
            "context_mask": context_masks,
            "context_indices": torch.as_tensor(context_indices, dtype=torch.long),
            "width_coords": self.width_coords,
            "height_coords": self.height_coords,
            "plane_coordinate": torch.tensor(self.plane_spec.coordinate, dtype=torch.float32),
        }
        if self.return_target:
            target_pos = end - 1 + self.target_offset
            target_timestep = self.timesteps[target_pos]
            target_data, target_mask = self._prepare_frame(target_timestep, is_target=True)
            sample.update(
                {
                    "target": target_data,
                    "target_mask": target_mask,
                    "target_index": torch.tensor(target_timestep, dtype=torch.long),
                }
            )
        if self.sample_transform is not None:
            sample = self.sample_transform(sample)
        return sample

    def _prepare_frame(
        self,
        timestep: int,
        *,
        is_target: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        frame = self.repository.get_frame(timestep, self.field, self.plane_spec)
        data = frame.data.clone() if self.copy_tensors else frame.data
        mask = frame.mask.clone() if self.copy_tensors else frame.mask
        if self.padding_value is not None:
            filler = torch.full_like(data, float(self.padding_value))
            data = torch.where(mask.unsqueeze(0), data, filler)
        transform = self.target_transform if is_target else self.frame_transform
        if transform is not None:
            data = transform(data)
        return data, mask


def build_datasets(
    repository: LaserNetDataRepository,
    sequence_length: int,
    field: FieldType,
    plane: Union[PlaneSelection, PlaneSpec],
    target_offset: int = 1,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    timesteps_splits: Optional[Dict[str, Sequence[int]]] = None,
    **dataset_kwargs: Any,
) -> Dict[str, LaserNetSequenceDataset]:
    if timesteps_splits is None:
        splits = repository.split(*split_ratios)
    else:
        splits = {name: sorted(int(t) for t in indices) for name, indices in timesteps_splits.items()}
    datasets: Dict[str, LaserNetSequenceDataset] = {}
    for name, indices in splits.items():
        if not indices:
            continue
        datasets[name] = LaserNetSequenceDataset(
            repository=repository,
            timesteps=indices,
            sequence_length=sequence_length,
            field=field,
            plane=plane,
            target_offset=target_offset,
            **dataset_kwargs,
        )
    return datasets


def build_dataloaders(
    datasets: Dict[str, LaserNetSequenceDataset],
    batch_size: int,
    shuffle_train: bool = True,
    num_workers: int = 0,
    persistent_workers: bool = False,
    **loader_kwargs: Any,
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for name, dataset in datasets.items():
        shuffle = shuffle_train if name == "train" else False
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            **loader_kwargs,
        )
    return loaders


__all__ = [
    "Plane",
    "FieldType",
    "PlaneSpec",
    "PlaneSelection",
    "LaserNetFrame",
    "LaserNetDataRepository",
    "LaserNetSequenceDataset",
    "build_datasets",
    "build_dataloaders",
]
