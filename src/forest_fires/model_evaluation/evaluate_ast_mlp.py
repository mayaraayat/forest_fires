"""Evaluate a trained AST-MLP audio classifier."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from models.mlp_classifier import LitMLP
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader


def evaluate_model(model: LitMLP, dataloader: DataLoader, device: str = "cpu") -> None:
    """Evaluate a trained PyTorch Lightning model on a dataset.

    Args:
        model (LitMLP): Trained PyTorch Lightning model.
        dataloader (DataLoader): DataLoader for evaluation data.
        device (str, optional): Device to perform computations ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        None
    """
    model.eval()
    model = model.to(device)
    all_logits, all_labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model.model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y)

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = 1 / (1 + np.exp(-all_logits))  # type: ignore
    all_preds = (all_probs > 0.5).astype(int)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1-score: {f1:.3f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

    return None
