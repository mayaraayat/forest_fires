"""Training script for ASTLightning with LoRA adapters."""

import torch  # type: ignore
from lightning.pytorch import Trainer, seed_everything  # type: ignore
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore

from src.forest_fires.data_preprocessing.datamodule import get_dataloader
from src.forest_fires.models.ast_lora import LoRaASTClassifier


def main() -> None:
    """Main training function."""
    seed_everything(42)
    batch_size = 2
    # Build dataloaders
    train_loader = get_dataloader(
        "data/forest_fire_dataset/train", batch_size=batch_size, shuffle=True
    )
    val_loader = get_dataloader(
        "data/forest_fire_dataset/val", batch_size=batch_size, shuffle=False
    )

    # Detect number of classes from dataset
    num_labels = len(train_loader.dataset.class_to_idx)

    # Model
    model = LoRaASTClassifier(num_labels=num_labels, lr=1e-4)

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints",
        filename="ast-lora-{epoch:02d}-{val_acc:.2f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )
    earlystop_cb = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=5,
    )

    # Trainer
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=20,
        precision=32,
        log_every_n_steps=1,
        callbacks=[checkpoint_cb, earlystop_cb],
    )

    # Train
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
