import pandas as pd
import numpy as np
import joblib
import torch
import os
import logging
from src.preprocess import preprocess_data
from transformers import DistilBertForSequenceClassification

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_and_save(model, X, test_df, model_name, output_path, label_encoder=None, tokenizer=None):
    """Predict genres and save to file."""
    if model_name in ['logreg', 'svm']:
        y_pred = model.predict(X)
    else:  # distilbert
        encodings = tokenizer(test_df['plot'].tolist(), truncation=True, padding=True, max_length=512)
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            def __getitem__(self, idx):
                return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            def __len__(self):
                return len(self.encodings['input_ids'])
        dataset = Dataset(encodings)
        model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(len(dataset)):
                inputs = dataset[i]
                outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()})
                pred = torch.argmax(outputs.logits, dim=1).item()
                predictions.append(pred)
        y_pred = label_encoder.inverse_transform(predictions)
    
    # Save predictions
    test_df['predicted_genre'] = y_pred
    test_df[['id', 'title', 'plot', 'predicted_genre']].to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

def main():
    """Predict genres for test data using trained models."""
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(base_path, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    train_df, test_df, X_train_tfidf, X_test_tfidf, vectorizer = preprocess_data()
    
    # Load and predict with models
    logreg = joblib.load(os.path.join(base_path, 'models', 'logreg_model.joblib'))
    predict_and_save(logreg, X_test_tfidf, test_df, 'logreg', os.path.join(output_dir, 'logreg_predictions.csv'))
    
    svm = joblib.load(os.path.join(base_path, 'models', 'svm_model.joblib'))
    predict_and_save(svm, X_test_tfidf, test_df, 'svm', os.path.join(output_dir, 'svm_predictions.csv'))
    
    distilbert_model = DistilBertForSequenceClassification.from_pretrained(os.path.join(base_path, 'models', 'distilbert_model'))
    tokenizer = joblib.load(os.path.join(base_path, 'models', 'distilbert_model', 'tokenizer.joblib'))
    label_encoder = joblib.load(os.path.join(base_path, 'models', 'distilbert_model', 'label_encoder.joblib'))
    predict_and_save(distilbert_model, None, test_df, 'distilbert', os.path.join(output_dir, 'distilbert_predictions.csv'), label_encoder, tokenizer)

if __name__ == '__main__':
    main()