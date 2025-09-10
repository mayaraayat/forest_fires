"""Audio Spectrogram Transformer (AST) with LoRA adapters."""

from typing import Any

import lightning.pytorch as pl  # type: ignore
import torch  # type: ignore
from peft import LoraConfig, get_peft_model  # type: ignore
from transformers import ASTModel  # type: ignore

from src.forest_fires.utils.log import get_logger

logger = get_logger(__name__)


def build_ast_lora_model(
    pretrained_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
) -> ASTModel:
    """Build an AST model with LoRA adapters for classification.

    Args:
        pretrained_model: Hugging Face model ID or local path.
        r: Rank of the LoRA update matrices.
        alpha: Scaling factor for LoRA layers.
        dropout: Dropout probability on LoRA layers.

    Returns:
        ASTForAudioClassification model wrapped with LoRA adapters.
    """
    base_model = ASTModel.from_pretrained(
        pretrained_model,
        attn_implementation="sdpa",
        dtype=torch.float32,
    )

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["query", "value"],
        lora_dropout=dropout,
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)
    return model


class LoRaASTClassifier(pl.LightningModule):  # type: ignore
    """Class for fine-tuning AST with LoRA adapters."""

    def __init__(self, num_labels: int, lr: float = 1e-4) -> None:
        """Initialize the LoRaASTClassifier.

        Args:
            num_labels: Number of output classes for classification.
            lr: Learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()
        self.ast_lora = build_ast_lora_model()
        hidden_size = self.ast_lora.config.hidden_size
        self.classifier = torch.nn.Linear(hidden_size, num_labels, dtype=torch.float32)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward method.

        Args:
            x: Input tensor of shape (batch_size, n_mels, time_frames).

        Returns:
            Logits tensor of shape (batch_size, num_labels).
        """
        outputs = self.ast_lora(input_values=x)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_dim]
        logits = self.classifier(cls_emb)
        return logits

    def training_step(self, batch: Any, batch_idx: Any) -> torch.Tensor:
        """Training step.

        Args:
            batch: A batch of data.
            batch_idx: Index of the batch.

        Returns:
            The training loss.
        """
        features, labels = batch
        features = features.to(torch.float32)
        outputs = self(features)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

        loss = self.loss(logits.float(), labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: Any) -> torch.Tensor:
        """Validation step.

        Args:
            batch: A batch of data.
            batch_idx: Index of the batch.

        Returns:
            The validation loss.
        """
        features, labels = batch
        features = features.to(torch.float32)
        outputs = self(features)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
        loss = self.loss(logits.float(), labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers for training.

        Returns:
            An AdamW optimizer.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    model = LoRaASTClassifier(num_labels=2)
    dummy_input = torch.randn(2, 128, 1024)  # batch=2, mel=128, frames=1024
    logits = model(dummy_input)
    print("Logits:", logits.shape)  # torch.Size([2, 2])
