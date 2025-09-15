"""
This module defines the MLP classifier and PyTorch Lightning wrapper for audio embeddings.
"""

import torch
import torch.nn as nn
import lightning.pytorch as pl

class MLPClassifier(nn.Module):
    """
    Simple multi-layer perceptron for classification.

    Attributes:
        mlp (nn.Sequential): Feed-forward layers.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        """
        Initialize the MLP.

        Args:
            input_dim (int): Dimensionality of input embeddings.
            hidden_dim (int, optional): Hidden layer size. Defaults to 256.
        """
        super().__init__()
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Output logits of shape (batch_size,)
        """
        return self.mlp(x).squeeze()


class LitMLP(pl.LightningModule):
    """
    PyTorch Lightning module for training MLPClassifier.

    Attributes:
        model (MLPClassifier): The MLP classifier.
        criterion (nn.BCEWithLogitsLoss): Binary cross-entropy loss.
        lr (float): Learning rate.
    """
    def __init__(self, input_dim: int, lr: float = 1e-3) -> None:
        """
        Initialize the Lightning module.

        Args:
            input_dim (int): Dimensionality of input embeddings.
            lr (float, optional): Learning rate. Defaults to 1e-3.
        """
        super().__init__()
        self.model: MLPClassifier = MLPClassifier(input_dim)
        self.criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        self.lr: float = lr

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform a training step.

        Args:
            batch (tuple): (inputs, labels)
            batch_idx (int): Batch index (unused)

        Returns:
            torch.Tensor: Computed loss
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device).float()
        logits: torch.Tensor = self.model(x)
        loss: torch.Tensor = self.criterion(logits, y)
        preds: torch.Tensor = (torch.sigmoid(logits) > 0.5).long()
        acc: torch.Tensor = (preds == y.long()).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Perform a validation step.

        Args:
            batch (tuple): (inputs, labels)
            batch_idx (int): Batch index (unused)
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device).float()
        logits: torch.Tensor = self.model(x)
        loss: torch.Tensor = self.criterion(logits, y)
        preds: torch.Tensor = (torch.sigmoid(logits) > 0.5).long()
        acc: torch.Tensor = (preds == y.long()).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return optimizer for training."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)
