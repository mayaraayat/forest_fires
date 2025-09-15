"""Datasets for audio classification using waveforms and AST embeddings."""

from glob import glob
from typing import List, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import ASTModel, AutoFeatureExtractor


class AudioDatasetWave(Dataset):  # type: ignore
    """Dataset for loading raw waveform audio files with labels.

    Attributes:
        sample_rate (int): Target sample rate for resampling audio.
        data (List[str]): List of audio file paths.
        labels (torch.Tensor): Tensor containing labels (0 or 1).
    """

    def __init__(self, root_dir: str, sample_rate: int = 16000) -> None:
        """Initialize the dataset by scanning class directories for WAV files.

        Args:
            root_dir (str): Root directory containing class_0/ and class_1/ folders.
            sample_rate (int, optional): Target sample rate for audio. Defaults to 16000.
        """
        self.sample_rate: int = sample_rate
        self.data: List[str] = []
        self.labels: List[int] = []

        for i, class_name in enumerate(["class_0", "class_1"]):
            files = glob(f"{root_dir}/{class_name}/*.wav")
            self.data.extend(files)
            self.labels.extend([i] * len(files))

        self.labels = torch.tensor(self.labels)

    def __len__(self) -> int:
        """Return the total number of audio samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load a waveform and its label.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, int]: waveform and label
        """
        waveform, sr = torchaudio.load(self.data[idx])
        waveform = waveform.mean(dim=0)  # Convert to mono
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return waveform, self.labels[idx].item()  # type: ignore


class AudioDatasetAST(Dataset):  # type: ignore
    """Dataset for extracting AST CLS embeddings from waveforms.

    Attributes:
        wave_dataset (AudioDatasetWave): Raw waveform dataset.
        ast_model (ASTModel): Pretrained AST model.
        feature_extractor (AutoFeatureExtractor): Feature extractor for AST.
        device (str): Device to run AST model on ('cpu' or 'cuda').
    """

    def __init__(
        self,
        wave_dataset: AudioDatasetWave,
        ast_model: ASTModel,
        feature_extractor: AutoFeatureExtractor,
        device: str = "cpu",
    ) -> None:
        """Initialize AST dataset with frozen AST model.

        Args:
            wave_dataset (AudioDatasetWave): Waveform dataset.
            ast_model (ASTModel): Pretrained AST model.
            feature_extractor (AutoFeatureExtractor): AST feature extractor.
            device (str, optional): Device for computation. Defaults to "cpu".
        """
        self.wave_dataset: AudioDatasetWave = wave_dataset
        self.ast_model: ASTModel = ast_model.to(device)
        self.feature_extractor: AutoFeatureExtractor = feature_extractor
        self.device: str = device

        self.ast_model.eval()
        for param in self.ast_model.parameters():
            param.requires_grad = False

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.wave_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Extract CLS embedding for a waveform and return with label.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, int]: CLS embedding and label
        """
        waveform, label = self.wave_dataset[idx]
        inputs = self.feature_extractor(
            waveform.numpy(), sampling_rate=16000, return_tensors="pt", do_normalize=True
        )
        input_values: torch.Tensor = inputs["input_values"].to(self.device)
        with torch.no_grad():
            outputs = self.ast_model(input_values=input_values)
            cls_embedding: torch.Tensor = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.squeeze(0), label
