"""Main module for the forest_fires package."""

from pathlib import Path

import lightning.pytorch as pl
import torch
from data_preprocessing.data_downloader import download_esc50_class0, download_forestfire_class1
from data_preprocessing.feature_extractor_ast import AudioDatasetAST, AudioDatasetWave
from model_evaluation.evaluate_ast_mlp import evaluate_model
from models.mlp_classifier import LitMLP
from torch.utils.data import DataLoader, random_split
from transformers import ASTModel, AutoFeatureExtractor


def main(
    data_dir: str = "data/forest_fire_dataset",
    batch_size: int = 8,
    max_epochs: int = 5,
    device: str | None = None,
    checkpoint_path: str = "checkpoints/lit_mlp_checkpoint.ckpt",
) -> None:
    """Run the full audio classification pipeline.

    Steps:
        1. Download and prepare datasets.
        2. Split dataset into train and validation sets.
        3. Initialize AST model and feature extractor.
        4. Create Lightning MLP model.
        5. Train model using PyTorch Lightning.
        6. Evaluate model performance.
        7. Save model checkpoint.

    Args:
        data_dir (str, optional): Root directory to store datasets.
        Defaults to "data/forest_fire_dataset".
        batch_size (int, optional): Batch size for training. Defaults to 8.
        max_epochs (int, optional): Number of training epochs. Defaults to 5.
        device (str, optional): Device for training ('cpu' or 'cuda'). Defaults to auto-detect.
        checkpoint_path (str, optional): Path to save trained model checkpoint.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Download datasets
    _ = download_esc50_class0()
    _ = download_forestfire_class1()

    # Prepare waveform dataset
    full_dataset = AudioDatasetWave(data_dir)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_wave, val_wave = random_split(full_dataset, [train_size, val_size])

    # Load AST model and feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    ast_model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    # Create AST embedding datasets
    train_dataset = AudioDatasetAST(train_wave, ast_model, feature_extractor, device=device)
    val_dataset = AudioDatasetAST(val_wave, ast_model, feature_extractor, device=device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize Lightning model
    lit_model = LitMLP(input_dim=ast_model.config.hidden_size).to(device)

    # Train
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="auto", devices=1, log_every_n_steps=1)
    trainer.fit(lit_model, train_loader, val_loader)

    # Evaluate
    evaluate_model(lit_model, val_loader, device=device)

    # Save checkpoint
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
