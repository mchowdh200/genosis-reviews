import os
import random
from glob import glob


def get_segments(
    segment_directory: str,
    segment_offset: int,
    gt_extension: str,
    delimiter: str,
) -> list[str]:
    """
    Parse segment numbers from genotype files in given directory.
    """
    segments = [
        x.split(delimiter)[segment_offset]
        for x in glob(f"{segment_directory}/*.{gt_extension}")
        if x.split(delimiter)[segment_offset] != "168"  # this segment had issues
    ]

    if len(segments) == 0:
        raise ValueError(
            f"No segments found\n"
            f"{segment_directory=}\n"
            f"{gt_extension=}\n"
            f"{delimiter=}\n"
            f"{segment_offset=}\n"
        )
    return segments


def train_val_test_split(
    random_seed: int, segments: list[str]
) -> tuple[list[str], list[str], list[str]]:
    random.seed(random_seed)
    shuffled_segments = random.sample(segments, len(segments))

    train_ratio = 0.8  # TODO make this a config option
    val_ratio = test_ratio = (1.0 - train_ratio) / 2.0

    train_segments = shuffled_segments[: int(train_ratio * len(segments))]
    val_segments = shuffled_segments[
        int(train_ratio * len(segments)) : int(
            (train_ratio + val_ratio) * len(segments)
        )
    ]
    test_segments = shuffled_segments[int((train_ratio + val_ratio) * len(segments)) :]

    # any empty sets?
    if not train_segments or not val_segments or not test_segments:
        raise ValueError(
            f"Train, val, test split failed\n"
            f"{train_ratio=}\n"
            f"{val_ratio=}\n"
            f"{test_ratio=}\n"
            f"{train_segments=}\n"
            f"{val_segments=}\n"
            f"{test_segments=}\n"
        )

    # check if the sets are disjoint
    if len(set(train_segments) & set(val_segments) & set(test_segments)) > 0:
        raise ValueError(
            f"Train/val/test sets are not disjoint\n"
            f"{train_segments=}\n"
            f"{val_segments=}\n"
            f"{test_segments=}\n"
        )
    return train_segments, val_segments, test_segments


def get_wandb_api_key() -> str:
    """
    Get the Weights & Biases API key from environment variable.
    """
    key = os.getenv("WANDB_API_KEY")
    if key is None:
        raise ValueError(
            "WANDB_API_KEY environment variable not set. "
            "Please set it to your Weights & Biases API key."
        )
    return key
