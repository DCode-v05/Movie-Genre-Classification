import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import joblib
from preprocess import preprocess_data

def train_logreg(X, y, model_path):
    """Train and save Logistic Regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model

def train_svm(X, y, model_path):
    """Train and save SVM model."""
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model

def train_distilbert(train_df, y_train, model_path, epochs=3):
    """Train and save DistilBERT model."""
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    # Prepare dataset
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_df['plot'].tolist(), truncation=True, padding=True, max_length=512)
    
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)
    
    train_dataset = Dataset(train_encodings, y_train_encoded)
    
    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=len(label_encoder.classes_)
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(model_path, 'logs'),
        logging_steps=10,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Train
    trainer.train()
    trainer.save_model(model_path)
    joblib.dump(label_encoder, os.path.join(model_path, 'label_encoder.joblib'))
    joblib.dump(tokenizer, os.path.join(model_path, 'tokenizer.joblib'))
    return model, label_encoder, tokenizer

def main():
    """Train models using preprocessed train data."""
    # Load preprocessed data
    train_df, test_df, X_train_tfidf, X_test_tfidf, vectorizer = preprocess_data()
    y_train = train_df['genre']
    
    # Create models directory
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.makedirs(os.path.join(base_path, 'models'), exist_ok=True)
    
    # Train models
    train_logreg(X_train_tfidf, y_train, os.path.join(base_path, 'models', 'logreg_model.joblib'))
    train_svm(X_train_tfidf, y_train, os.path.join(base_path, 'models', 'svm_model.joblib'))
    train_distilbert(train_df, y_train, os.path.join(base_path, 'models', 'distilbert_model'))

if __name__ == '__main__':
    main()
