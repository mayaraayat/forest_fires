"""Define the collate function for AST model."""

from typing import Any

import torch


def ast_collate_fn(batch: Any, max_frames: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function to pad/crop mel spectrograms to a fixed number of frames.

    Args:
        batch: List of tuples (mel_spectrogram, label).
        max_frames: Maximum number of frames to pad/crop to.

    Returns:
        A tuple of:
            - features: A tensor of shape (batch_size, max_frames, n_mels).
            - labels: A tensor of shape (batch_size,).
    """
    features, labels = zip(*batch, strict=False)  # features: [B, 1, 128, T]

    processed = []
    for mel in features:
        mel = mel.squeeze(0).transpose(0, 1)  # [T, 128]

        T = mel.size(0)
        if T < max_frames:
            # pad
            pad_amount = max_frames - T
            mel = torch.nn.functional.pad(mel, (0, 0, 0, pad_amount))
        else:
            # crop
            mel = mel[:max_frames, :]

        processed.append(mel)

    features = torch.stack(processed)  # [B, max_frames, 128]
    labels = torch.tensor(labels)

    return features, labels
