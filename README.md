# Movie Genre Classification

## Project Description

Movie Genre Classification is an AI-powered system that predicts movie genres from plot summaries using advanced Natural Language Processing (NLP) and machine learning models. The project leverages Logistic Regression, Support Vector Machine (SVM), and DistilBERT to automate and enhance the genre classification process for movie datasets. It is designed for researchers, data scientists, and movie enthusiasts interested in text classification and NLP applications.

## Features

- Predicts movie genres from plot summaries using multiple models (Logistic Regression, SVM, DistilBERT)
- Comprehensive data preprocessing and sentiment analysis
- Automated exploratory data analysis (EDA) with visualizations
- Outputs predictions in structured CSV files for easy analysis
- Modular, extensible codebase for experimentation and research

## Tech Stack

- **Programming Language:** Python 3.8+
- **Machine Learning:** scikit-learn, transformers (DistilBERT), PyTorch
- **NLP & Data Processing:** NLTK, TextBlob, pandas, numpy, spaCy
- **Visualization:** matplotlib, seaborn, wordcloud
- **Utilities:** joblib, shap

All dependencies are listed in `requirements.txt`.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Place your data files (`train_data.txt`, `test_data.txt`) in the `data/` directory. Files must be UTF-8 encoded.

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/TensoRag/Movie-Genre-Classification.git
   cd Movie-Genre-Classification
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare data:**
   - Ensure `data/train_data.txt` and `data/test_data.txt` are present and properly formatted.
   - Data format:
     - Train: `ID ::: TITLE ::: GENRE ::: DESCRIPTION`
     - Test:  `ID ::: TITLE ::: DESCRIPTION`
   - Source: ftp://ftp.fu-berlin.de/pub/misc/movies/database/

## Usage

### 1. Exploratory Data Analysis
Generate and save EDA visualizations:
```bash
python notebooks/EDA.py
```
Visualizations will be saved in the `outputs/` directory (e.g., genre distribution, sentiment analysis, co-occurrence heatmaps).

### 2. Train Models
Train all models using your training data:
```bash
cd src
python train.py
```
Trained models will be saved in the `models/` directory.

### 3. Predict Genres
Generate predictions for your test data:
```bash
python evaluate.py
```
Prediction results will be saved in the `outputs/` directory as CSV files (e.g., `logreg_predictions.csv`, `svm_predictions.csv`, `distilbert_predictions.csv`).

## Project Structure

```
Movie-Genre-Classification/
├── data/                # Input data files
│   ├── train_data.txt
│   ├── test_data.txt
│   ├── test_data_solution.txt
│   └── description.txt
├── models/              # Trained models
│   ├── logreg_model.joblib
│   ├── svm_model.joblib
│   └── distilbert_model/
│       ├── config.json
│       ├── label_encoder.joblib
│       ├── model.safetensors
│       └── tokenizer.joblib
├── outputs/             # Predictions & EDA visualizations
│   ├── *.csv            # Model predictions
│   └── *.png            # EDA plots
├── notebooks/           # EDA and analysis scripts
│   └── EDA.py
├── src/                 # Source code
│   ├── preprocess.py    # Data loading & preprocessing
│   ├── train.py         # Model training
│   └── evaluate.py      # Model evaluation & prediction
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request describing your changes.

## Contact

For questions, suggestions, or collaboration:
- **GitHub:** [TensoRag](https://github.com/TensoRag)
- **Email:** denistanb05@gmail.com
