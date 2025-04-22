import pandas as pd
import numpy as np
import joblib
import torch
import os
import logging
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from torch.utils.data import DataLoader

# Import preprocess_data from the same directory
from preprocess import preprocess_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_and_save(model, X, test_df, model_name, output_path, label_encoder=None, tokenizer=None, batch_size=8):
    """Predict genres and save to file."""
    if model_name in ['logreg', 'svm']:
        if X is None:
            raise ValueError(f"{model_name} requires TF-IDF input (X), but None was provided.")
        y_pred = model.predict(X)
    else:  # distilbert
        if tokenizer is None or label_encoder is None:
            raise ValueError("DistilBERT requires both tokenizer and label_encoder.")
        encodings = tokenizer(test_df['plot'].tolist(), truncation=True, padding=True, max_length=512)
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            def __getitem__(self, idx):
                return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            def __len__(self):
                return len(self.encodings['input_ids'])
        dataset = Dataset(encodings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(preds)
        y_pred = label_encoder.inverse_transform(predictions)
    
    # Save predictions
    test_df['predicted_genre'] = y_pred
    output_cols = ['id', 'title', 'plot', 'predicted_genre']
    # Ensure all columns exist in test_df
    missing_cols = [col for col in output_cols if col not in test_df.columns]
    if missing_cols:
        logger.warning(f"Missing columns in test_df: {missing_cols}. Filling with 'Unknown'.")
        for col in missing_cols:
            test_df[col] = 'Unknown'
    test_df[output_cols].to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

def main():
    """Predict genres for test data using trained models."""
    # Set base_path to the project root (parent of src/)
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(base_path, 'outputs')
    models_dir = os.path.join(base_path, 'models')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory {models_dir} does not exist. Ensure models are saved.")
    
    # Load preprocessed data
    train_df, test_df, X_train_tfidf, X_test_tfidf, vectorizer = preprocess_data()
    
    # Load and predict with Logistic Regression
    logreg_path = os.path.join(models_dir, 'logreg_model.joblib')
    if not os.path.exists(logreg_path):
        raise FileNotFoundError(f"Logistic Regression model not found at {logreg_path}")
    logreg = joblib.load(logreg_path)
    predict_and_save(logreg, X_test_tfidf, test_df, 'logreg', os.path.join(output_dir, 'logreg_predictions.csv'))
    
    # Load and predict with SVM
    svm_path = os.path.join(models_dir, 'svm_model.joblib')
    if not os.path.exists(svm_path):
        raise FileNotFoundError(f"SVM model not found at {svm_path}")
    svm = joblib.load(svm_path)
    predict_and_save(svm, X_test_tfidf, test_df, 'svm', os.path.join(output_dir, 'svm_predictions.csv'))
    
    # Load and predict with DistilBERT
    distilbert_path = os.path.join(models_dir, 'distilbert_model')
    if not os.path.exists(distilbert_path):
        raise FileNotFoundError(f"DistilBERT model directory not found at {distilbert_path}")
    for file_name in ['config.json', 'model.safetensors', 'label_encoder.joblib']:
        if not os.path.exists(os.path.join(distilbert_path, file_name)):
            raise FileNotFoundError(f"Required DistilBERT file {file_name} not found in {distilbert_path}")
    distilbert_model = DistilBertForSequenceClassification.from_pretrained(distilbert_path)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    label_encoder = joblib.load(os.path.join(distilbert_path, 'label_encoder.joblib'))
    predict_and_save(
        distilbert_model, 
        None, 
        test_df, 
        'distilbert', 
        os.path.join(output_dir, 'distilbert_predictions.csv'), 
        label_encoder, 
        tokenizer,
        batch_size=8
    )

if __name__ == '__main__':
    main()
