"""The data acquisition module for the forest_fires package."""

import shutil
from pathlib import Path
from typing import Any

import kagglehub  # type: ignore

from src.forest_fires.utils.log import get_logger

logger = get_logger(__name__)


def download_dataset() -> Any:
    """Download and copy the forest fire dataset .wav files into class_1 folder."""
    dataset_dir = kagglehub.dataset_download("forestprotection/forest-wild-fire-sound-dataset")
    dataset_dir = Path(dataset_dir)

    target_dir = Path("data/forest_fire_dataset/class_1")
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy all wav files into class_1
    for wav_file in dataset_dir.rglob("*.wav"):
        shutil.copy(str(wav_file), target_dir / wav_file.name)

    return target_dir


if __name__ == "__main__":
    path = download_dataset()
    logger.info("Dataset downloaded to: %s", path)
