# Forest Fires Audio Classification
This repository contains code for classifying audio recordings of forest fires using advanced machine learning techniques. The project leverages pre-trained Audio Spectrogram Transformer (AST) models and Multi-Layer Perceptron (MLP) classifiers to achieve high accuracy in identifying forest fire sounds.

## Features
- Data preprocessing and feature extraction using AST embeddings.
- Training and fine-tuning of AST-MLP models.
- Evaluation of model performance with visualizations.
- Support for LoRA adapters to optimize model training.
- ViT Fine-tuning for enhanced classification comparison.

## Installation
To set up the environment, use the following command:
```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

## Usage
In the Notebooks directory, you will find Jupyter notebooks that guide you through the ViT fine-tuning and AST-LoRA training processes.
The source code is located in the `src/forest_fires` directory. You can find scripts for data preprocessing, model training, and evaluation.

The details about the project motivation, dataset, and methodology can be found in the [Project Report](docs/FMR-ForestFires.pdf).
