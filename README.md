# Movie Genre Classification

Welcome to the **Movie Genre Classification** project! This repository implements a robust movie genre prediction system using Logistic Regression, Support Vector Machine (SVM), and DistilBERT models. The `train_data.txt` file trains the models, while the `evaluate.py` script generates genre predictions for the unlabeled `test_data.txt`.

## Overview

This project leverages machine learning to classify movie genres based on plot summaries, offering a practical application of natural language processing (NLP). It’s designed for ease of use in a Codespaces environment and is ideal for learning, experimentation, or integration into larger systems.

## Project Structure

- `Movie-Genre-Classification/`
  - `data/`
    - `train_data.txt` # Training data (ID ::: Title ::: Genre ::: Plot)
    - `test_data.txt`  # Test data (ID ::: Title ::: Plot)
  - `models/`          # Trained models (logreg_model.joblib, svm_model.joblib, distilbert_model/)
  - `outputs/`         # Predicted genres (e.g., logreg_predictions.csv)
  - `src/`
    - `preprocess.py`  # Data preprocessing and vectorization
    - `train.py`       # Model training script
    - `evaluate.py`    # Prediction script for test data
  - `README.md`        # This file

## Prerequisites

- **Python 3.8+**
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `transformers`, `torch`, `joblib`, `textblob`, `nltk`

## Installation

1. Clone or set up the repository in Codespaces: `cd /workspaces/ && git clone <your-repo-url> Movie-Genre-Classification && cd Movie-Genre-Classification`
2. Install dependencies: `pip install pandas numpy scikit-learn transformers torch joblib textblob nltk`

## Data Preparation

- Place `train_data.txt` and `test_data.txt` in the `data/` directory.
- **train_data.txt** format: Each line must follow `ID ::: Title ::: Genre ::: Plot` (e.g., `1 ::: Oscar et la dame rose (2009) ::: drama ::: Listening in to a conversation...`).
- **test_data.txt** format: Each line must follow `ID ::: Title ::: Plot` (e.g., `1 ::: Edgar's Lunch (1998) ::: L.R. Brane loves his life...`), with no genre.
- Ensure files use `utf-8` encoding. If issues occur, re-save with a text editor or run: `iconv -f utf-8 -t utf-8 -c data/test_data.txt -o data/temp.txt && mv data/temp.txt data/test_data.txt`

## Usage

### 1. Train Models

Execute the training script with `cd src && python train.py` to train models using `train_data.txt`.

- Trained models are saved in `models/`:
  - `logreg_model.joblib` (Logistic Regression)
  - `svm_model.joblib` (Support Vector Machine)
  - `distilbert_model/` (DistilBERT with tokenizer and label encoder)

### 2. Predict Genres

Run the evaluation script with `python evaluate.py` to predict genres for `test_data.txt`.

- Predictions are saved in `outputs/` as CSV files (e.g., `logreg_predictions.csv`, `svm_predictions.csv`, `distilbert_predictions.csv`) with columns `id`, `title`, `plot`, and `predicted_genre`.

### 3. Verify Output

- Confirm `models/` contains the trained models.
- Verify `outputs/` contains the prediction files.
