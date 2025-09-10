"""Training script for ASTLightning with LoRA adapters."""

import torch  # type: ignore
from lightning.pytorch import Trainer, seed_everything  # type: ignore
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore

from src.forest_fires.data_preprocessing.datamodule import get_dataloader
from src.forest_fires.models.ast_lora import LoRaASTClassifier


def main(batch_size: int, lr: float = 1e-4, num_epochs: int = 20) -> None:
    """Main training function.

    Args:
        batch_size: Batch size for training and validation.
        lr: Learning rate for the optimizer.
        num_epochs: Number of epochs to train.
    """
    seed_everything(42)
    train_loader = get_dataloader(
        "data/forest_fire_dataset/train", batch_size=batch_size, shuffle=True
    )
    val_loader = get_dataloader(
        "data/forest_fire_dataset/val", batch_size=batch_size, shuffle=False
    )

    # Quick dataset sanity check
    xb, yb = next(iter(train_loader))
    print("Batch:", xb.shape, yb.shape)
    print("nan?", torch.isnan(xb).any().item(), "inf?", torch.isinf(xb).any().item())

    # ----------------- MODEL -----------------
    num_labels = len(train_loader.dataset.class_to_idx)
    model = LoRaASTClassifier(num_labels=num_labels, lr=lr)

    # Quick model forward check
    with torch.no_grad():
        out = model(xb)  # expect shape [B, num_labels]
        print("Model output:", out.shape)
        print("nan?", torch.isnan(out).any().item(), "inf?", torch.isinf(out).any().item())

        # Quick loss check
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(out, yb)
        print("Loss sample:", loss.item())

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

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=num_epochs,
        precision=32,
        log_every_n_steps=1,
        callbacks=[checkpoint_cb, earlystop_cb],
    )

    # Train
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main(batch_size=2)
