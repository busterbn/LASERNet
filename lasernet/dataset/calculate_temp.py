"""
Calculate temperature statistics from the dataset.

This module provides functions to compute min/max temperature values
dynamically from the training data for normalization purposes.
"""

import torch
from typing import Tuple, Optional
from .loading import SliceSequenceDataset


def calculate_temp_stats(
    plane: str = "xz",
    split: str = "train",
    sequence_length: int = 4,
    target_offset: int = 1,
    preload: bool = True,
    max_samples: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Calculate global min/max temperature values from the dataset.

    Args:
        plane: Plane orientation ("xy", "xz", or "yz")
        split: Dataset split ("train", "val", or "test")
        sequence_length: Number of timesteps in each sequence
        target_offset: Offset for target frame
        preload: Whether to preload data into memory
        max_samples: Optional limit on number of samples to scan (for faster computation)

    Returns:
        Tuple of (temp_min, temp_max)
    """
    dataset = SliceSequenceDataset(
        field="temperature",
        plane=plane,
        split=split,
        sequence_length=sequence_length,
        target_offset=target_offset,
        preload=preload,
    )

    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    print(f"Scanning {num_samples} samples for temperature min/max...")

    global_min = float("inf")
    global_max = float("-inf")

    for i in range(num_samples):
        sample = dataset[i]
        ctx = sample["context"]  # [seq_len, 1, H, W]
        tgt = sample["target"]   # [1, H, W]

        # Combine context + target
        all_frames = torch.cat([ctx, tgt.unsqueeze(0)], dim=0)

        frame_min = all_frames.min().item()
        frame_max = all_frames.max().item()

        global_min = min(global_min, frame_min)
        global_max = max(global_max, frame_max)

    print(f"Temperature MIN: {global_min}")
    print(f"Temperature MAX: {global_max}")

    return global_min, global_max


def calculate_temp_stats_fast(
    plane: str = "xz",
    split: str = "train",
    sequence_length: int = 3,
    target_offset: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_samples: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Calculate global min/max temperature values from FastSliceSequenceDataset.

    This version works with preprocessed .pt files for faster loading.

    Args:
        plane: Plane orientation ("xy", "xz", or "yz")
        split: Dataset split ("train", "val", or "test")
        sequence_length: Number of timesteps in each sequence
        target_offset: Offset for target frame
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        max_samples: Optional limit on number of samples to scan (for faster computation)

    Returns:
        Tuple of (temp_min, temp_max)
    """
    from ..micronet.dataset.fast_loading import FastSliceSequenceDataset

    dataset = FastSliceSequenceDataset(
        plane=plane,
        split=split,
        sequence_length=sequence_length,
        target_offset=target_offset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    print(f"Scanning {num_samples} temperature samples for min/max...")

    global_min = float("inf")
    global_max = float("-inf")

    for i in range(num_samples):
        sample = dataset[i]
        ctx = sample["context"]  # [seq_len, 1, H, W]
        tgt = sample["target"]   # [1, H, W]

        # Combine context + target
        all_frames = torch.cat([ctx, tgt.unsqueeze(0)], dim=0)

        frame_min = all_frames.min().item()
        frame_max = all_frames.max().item()

        global_min = min(global_min, frame_min)
        global_max = max(global_max, frame_max)

    print(f"✓ Temperature MIN: {global_min:.2f} K")
    print(f"✓ Temperature MAX: {global_max:.2f} K")

    return global_min, global_max


def get_default_temp_range() -> Tuple[float, float]:
    """
    Get default temperature range based on physics.

    Returns:
        Tuple of (temp_min, temp_max) where:
        - temp_min: Room temperature baseline (300K)
        - temp_max: Pre-calculated max from full dataset (4652.05K)
    """
    return 300.0, 4652.0498046875


def get_temp_range_from_checkpoint(checkpoint_path: str) -> Tuple[float, float]:
    """
    Extract temperature range from a saved model checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file

    Returns:
        Tuple of (temp_min, temp_max) extracted from the model's buffers

    Raises:
        KeyError: If temperature buffers are not found in the checkpoint
        FileNotFoundError: If checkpoint file doesn't exist
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract from model state dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if "temp_min" not in state_dict or "temp_max" not in state_dict:
        raise KeyError(
            f"Temperature buffers not found in checkpoint. "
            f"Available keys: {list(state_dict.keys())[:10]}..."
        )

    temp_min = float(state_dict["temp_min"].item())
    temp_max = float(state_dict["temp_max"].item())

    return temp_min, temp_max


# Script mode: calculate and display stats
if __name__ == "__main__":
    temp_min, temp_max = calculate_temp_stats(
        plane="xz",
        split="train",
        sequence_length=4,
        target_offset=1,
        preload=True,
    )

    print("\n==== RESULTS ====")
    print(f"Temperature MIN: {temp_min}")
    print(f"Temperature MAX: {temp_max}")
    print("\nUse these values in your model initialization:")