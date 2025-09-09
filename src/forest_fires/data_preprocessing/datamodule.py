"""Define the Dataset Module with labels."""

from pathlib import Path
from typing import Any, Callable

import torch  # type: ignore
import torchaudio  # type: ignore
from torch.utils.data import DataLoader, Dataset  # type: ignore

from src.forest_fires.utils.collate_fn import ast_collate_fn


class AudioDataset(Dataset):  # type: ignore
    """Dataset for loading audio files, converting to mel spectrograms, and attaching labels."""

    def __init__(
        self,
        files: list[Path],
        class_to_idx: dict[str, int],
        sample_rate: int = 16_000,
        n_fft: int = 512,
        n_mels: int = 128,
        f_max: int = 8000,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Initialize the AudioDataset.

        Args:
            files: List of audio file paths.
            class_to_idx: Mapping from class names to indices.
            sample_rate: Target sample rate for audio files.
            n_fft: The size of FFT.
            n_mels: Number of mel bands.
            f_max: Maximum frequency for mel spectrogram.
            transform: Optional transform to be applied to the mel spectrogram.
        """
        self.files = files
        self.class_to_idx = class_to_idx
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=int(0.025 * sample_rate),
            hop_length=int(0.010 * sample_rate),  # 10ms hop
            window_fn=torch.hamming_window,
            n_mels=n_mels,
            f_max=f_max,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of audio files in the dataset."""
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a mel spectrogram and its label for a given index."""
        file_path = self.files[idx]
        waveform, sr = torchaudio.load(file_path)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # convert to mono
        mel_spec = self.mel_transform(waveform)
        mel_db = self.db_transform(mel_spec)

        if self.transform:
            mel_db = self.transform(mel_db)

        # Label from parent folder
        label_str = file_path.parent.name
        label = self.class_to_idx[label_str]

        return mel_db, label


def get_dataloader(
    data_dir: str | Path,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn: Callable[
        [list[tuple[torch.Tensor, int]]], tuple[torch.Tensor, torch.Tensor]
    ] = ast_collate_fn,
    **dataset_kwargs: Any,
) -> DataLoader:
    """Create DataLoader for all audio files in a directory.

    Assumes directory structure: data_dir/class_x/xxx.wav

    Args:
        data_dir: Directory containing audio files organized in subdirectories by class.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset.
        num_workers: Number of worker processes for data loading.
        collate_fn: Function to collate samples into a batch.
        **dataset_kwargs: Additional arguments to pass to the AudioDataset.

    Returns:
        DataLoader: DataLoader for the audio dataset.
    """
    data_dir = Path(data_dir)
    files = list(data_dir.rglob("*.wav"))

    # build class mapping from folder names
    classes = sorted({p.parent.name for p in files})
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    dataset = AudioDataset(files, class_to_idx=class_to_idx, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


# Example usage:
if __name__ == "__main__":
    loader = get_dataloader("data/forest_fire_dataset", batch_size=8)
    for batch, labels in loader:
        print(batch.shape, labels)
        break
