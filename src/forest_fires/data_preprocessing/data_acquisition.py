"""The data acquisition module for the forest_fires package."""

import shutil
from typing import Any

import kagglehub  # type: ignore

from src.forest_fires.utils.log import get_logger

logger = get_logger(__name__)


def download_dataset() -> Any:
    """Download the latest version of the forest fire dataset."""
    path = kagglehub.dataset_download("forestprotection/forest-wild-fire-sound-dataset")
    # Move the downloaded dataset to the appropriate directory
    shutil.move(path, "data/forest_fire_dataset.zip")
    return path


if __name__ == "__main__":
    path = download_dataset()
    logger.info("Dataset downloaded to: %s", path)
