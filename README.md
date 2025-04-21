Movie Genre Classification
This project implements a movie genre classification system using Logistic Regression, SVM, and DistilBERT models. The train_data.txt is used for training, and the evaluate.py script predicts genres for the unlabeled test_data.txt.
Project Structure
Movie-Genre-Classification/
│
├── data/
│   ├── train_data.txt    # Training data (ID ::: Title ::: Genre ::: Plot)
│   └── test_data.txt     # Test data (ID ::: Title ::: Plot, no genre)
│
├── models/               # Trained models (logreg_model.joblib, svm_model.joblib, distilbert_model/)
│
├── outputs/              # Predicted genres (e.g., logreg_predictions.csv)
│
├── src/
│   ├── preprocess.py     # Data preprocessing and vectorization
│   ├── train.py          # Model training script
│   └── evaluate.py       # Prediction script for test data
│
└── README.md             # This file

Prerequisites

Python 3.8+
Required libraries: pandas, numpy, scikit-learn, transformers, torch, joblib, textblob, nltk

Installation

Clone or set up the repository in Codespaces:
cd /workspaces/
git clone <your-repo-url> Movie-Genre-Classification
cd Movie-Genre-Classification


Install dependencies:
pip install pandas numpy scikit-learn transformers torch joblib textblob nltk



Data Preparation

Place train_data.txt and test_data.txt in the data/ directory.

train_data.txt format: Each line should be ID ::: Title ::: Genre ::: Plot (e.g., 1 ::: Oscar et la dame rose (2009) ::: drama ::: Listening in to a conversation...).

test_data.txt format: Each line should be ID ::: Title ::: Plot (e.g., 1 ::: Edgar's Lunch (1998) ::: L.R. Brane loves his life...), with no genre.

Ensure files are saved with utf-8 encoding. If issues arise, re-save with a text editor or use:
iconv -f utf-8 -t utf-8 -c data/test_data.txt -o data/temp.txt && mv data/temp.txt data/test_data.txt



Usage
1. Train Models
Run the training script to train models using train_data.txt:
cd src
python train.py


Models are saved in models/:
logreg_model.joblib (Logistic Regression)
svm_model.joblib (Support Vector Machine)
distilbert_model/ (DistilBERT with tokenizer and label encoder)



2. Predict Genres
Run the evaluation script to predict genres for test_data.txt:
python evaluate.py


Predictions are saved in outputs/ as CSV files (e.g., logreg_predictions.csv, svm_predictions.csv, distilbert_predictions.csv) with columns id, title, plot, and predicted_genre.

3. Verify Output

Check models/ for trained models.
Check outputs/ for prediction files.

Troubleshooting

Empty test_df or Errors: If logs show "Skipping malformed line," verify test_data.txt format. Share debug logs (e.g., Raw line: '...', Split parts: [...]) for assistance.
Missing Models: Ensure train.py runs successfully before evaluate.py.
Encoding Issues: If data loading fails, try utf-8-sig encoding by modifying preprocess.py to use encoding='utf-8-sig' in the open() calls.
Prediction Mismatch: Ensure train_data.txt has sufficient genre variety for training.

Notes

The project assumes a Codespaces environment at /workspaces/Movie-Genre-Classification/.
Logs are output to the terminal for debugging (e.g., INFO:src.preprocess:Loaded test_df with columns: [...]).
Predictions are based on trained models; accuracy depends on training data quality.

License
[Add your license here, e.g., MIT License]
