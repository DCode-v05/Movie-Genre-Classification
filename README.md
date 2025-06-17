# MovieGenrePredictor

Welcome to MovieGenrePredictor! This project is an AI-powered movie genre classification system that predicts genres from plot summaries using Logistic Regression, Support Vector Machine (SVM), and DistilBERT.

## 🚀 Features

- **Multi-Model Prediction:** Combines Logistic Regression, SVM, and DistilBERT for robust genre classification.
- **NLP-Driven:** Leverages natural language processing to analyze movie plot summaries.
- **Easy Setup:** Designed for seamless use in GitHub Codespaces or local environments.
- **Structured Output:** Generates CSV files with predicted genres for test data.

## 🛠️ Tech Stack

- **AI/ML:** Scikit-learn (Logistic Regression, SVM), Transformers (DistilBERT), PyTorch
- **Data Processing:** Pandas, NumPy, TextBlob, NLTK
- **Tools:** Joblib (model serialization), Git, GitHub Codespaces
- **Language:** Python 3.8+
- **Other:** UTF-8 encoded text files for data input

## 🏁 Getting Started

### Prerequisites

- Python 3.8+
- Required libraries: pandas, numpy, scikit-learn, transformers, torch, joblib, textblob, nltk
- UTF-8 encoded `train_data.txt` and `test_data.txt` files in the `data/` directory

### Installation

1. Clone the repository:  
   `git clone https://github.com/Denistanb/Movie-Genre-Classification.git`

2. Navigate to the project directory:  
   `cd Movie-Genre-Classification`

3. Install dependencies:  
   `pip install pandas numpy scikit-learn transformers torch joblib textblob nltk`

4. Ensure data files are in place:  
   - Place `train_data.txt` and `test_data.txt` in the `data/` directory.
   - Verify UTF-8 encoding. If issues occur, run:  
     `iconv -f utf-8 -t utf-8 -c data/test_data.txt -o data/temp.txt && mv data/temp.txt data/test_data.txt`

## Usage

### Train Models

Run the training script to train models using `train_data.txt`:

```bash
cd src && python train.py
```

Trained models are saved in `models/`:
- `logreg_model.joblib` (Logistic Regression)
- `svm_model.joblib` (SVM)
- `distilbert_model/` (DistilBERT with tokenizer and label encoder)

### Predict Genres

Run the evaluation script to predict genres for `test_data.txt`:

```bash
python evaluate.py
```

Predictions are saved in `outputs/` as CSV files (e.g., `logreg_predictions.csv`) with columns: `id`, `title`, `plot`, `predicted_genre`.

### Verify Results

- Check `models/` for trained models.
- Check `outputs/` for prediction CSV files.

## 🧠️ Project Structure

```
MovieGenrePredictor/
├── data/              # Input data
│   ├── train_data.txt    # Training data (ID ::: Title ::: Genre ::: Plot)
│   └── test_data.txt     # Test data (ID ::: Title ::: Plot)
├── models/            # Trained models
│   ├── logreg_model.joblib
│   ├── svm_model.joblib
│   └── distilbert_model/
├── outputs/           # Prediction results (e.g., logreg_predictions.csv)
├── src/               # Source code
│   ├── preprocess.py     # Data preprocessing and vectorization
│   ├── train.py          # Model training script
│   └── evaluate.py       # Prediction script
├── README.md          # This file
```

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## 📬 Contact

Have questions or feedback? Reach out to me:

- **GitHub:** Denistanb
- **Email:** denistanb05@gmail.com
