# Movie Genre Classification

## Project Description

MovieGenrePredictor is an AI-powered system designed to predict movie genres from plot summaries using multiple machine learning models, including Logistic Regression, Support Vector Machine (SVM), and DistilBERT. The project utilizes natural language processing (NLP) techniques to analyze and classify movie plots, making it an effective tool for automated genre classification.

## Features

- Supports multiple models: Logistic Regression, SVM, and DistilBERT.
- Utilizes NLP to process and understand movie plot summaries.
- Simple setup for use on local machines or GitHub Codespaces.
- Outputs predictions in structured CSV files for further analysis.

## Tech Stack

- Python 3.8+
- Scikit-learn (Logistic Regression, SVM)
- Transformers (DistilBERT)
- PyTorch
- Pandas, NumPy, TextBlob, NLTK
- Joblib for model serialization

## Getting Started

This section provides an overview of how to set up and use the project.

### Prerequisites

- Python 3.8 or higher
- Required Python libraries: pandas, numpy, scikit-learn, transformers, torch, joblib, textblob, nltk
- Training and test data files (`train_data.txt` and `test_data.txt`) encoded in UTF-8, placed under the `data/` directory

### Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/Denistanb/Movie-Genre-Classification.git
   ```

2. **Navigate to the project directory:**
   ```
   cd Movie-Genre-Classification
   ```

3. **Install the required dependencies:**
   ```
   pip install pandas numpy scikit-learn transformers torch joblib textblob nltk
   ```

4. **Prepare the data files:**
   - Place `train_data.txt` and `test_data.txt` in the `data/` directory.
   - Ensure files are UTF-8 encoded. To fix encoding issues, you can use:
     ```
     iconv -f utf-8 -t utf-8 -c data/test_data.txt -o data/temp.txt && mv data/temp.txt data/test_data.txt
     ```

## Usage

### Train Models

Run the training script to train the models using your training data:

```
cd src
python train.py
```

The trained models will be saved in the `models/` directory as follows:
- `logreg_model.joblib` for Logistic Regression
- `svm_model.joblib` for SVM
- `distilbert_model/` directory for DistilBERT

### Predict Genres

Run the evaluation script to generate predictions for your test data:

```
python evaluate.py
```

Prediction results will be saved in the `outputs/` directory as CSV files (e.g., `logreg_predictions.csv`), each containing columns: `id`, `title`, `plot`, `predicted_genre`.

### Verify Results

- Trained models are stored in the `models/` directory.
- Prediction outputs are available in the `outputs/` directory.

## Project Structure

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
├── README.md           # Project documentation
```

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository.
2. Create a new branch:
   ```
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```
   git commit -m "Add your feature"
   ```
4. Push to the branch:
   ```
   git push origin feature/your-feature
   ```
5. Open a pull request.

## Contact

For questions or feedback, contact:

- GitHub: [Denistanb](https://github.com/Denistanb)
- Email: denistanb05@gmail.com
