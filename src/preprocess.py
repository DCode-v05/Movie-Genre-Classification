import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import re
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download all required NLTK resources, including punkt_tab
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(train_file, test_file):
    """Load and parse train and test data text files."""
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Load train data (labeled)
    train_data = []
    train_path = os.path.join(base_path, 'data', train_file)
    try:
        with open(train_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                parts = re.split(r'\s*:::\s*', line)
                logger.debug(f"Raw line: '{line}', Split parts: {parts}")
                if len(parts) == 4:
                    train_data.append({
                        'id': parts[0],
                        'title': parts[1],
                        'genre': parts[2],
                        'plot': parts[3]
                    })
                else:
                    logger.warning(f"Skipping malformed line in {train_path}: {line} (parts: {parts})")
        train_df = pd.DataFrame(train_data)
        logger.info(f"Loaded train_df with columns: {train_df.columns.tolist()}")
    except FileNotFoundError:
        logger.error(f"Train file not found: {train_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading train data: {e}")
        raise

    # Load test data (unlabeled, for prediction)
    test_data = []
    test_path = os.path.join(base_path, 'data', test_file)
    try:
        with open(test_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                parts = re.split(r'\s*:::\s*', line)
                logger.debug(f"Raw line: '{line}', Split parts: {parts}")
                if len(parts) >= 2:  # At least ID and Plot (Title optional for prediction)
                    test_data.append({
                        'id': parts[0],
                        'title': parts[1] if len(parts) > 1 else 'Unknown',
                        'plot': parts[2] if len(parts) > 2 else parts[1]  # Plot as last part
                    })
                else:
                    logger.warning(f"Skipping malformed line in {test_path}: {line} (parts: {parts})")
        test_df = pd.DataFrame(test_data)
        logger.info(f"Loaded test_df with columns: {test_df.columns.tolist()}")
    except FileNotFoundError:
        logger.error(f"Test file not found: {test_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise

    return train_df, test_df

def clean_text(text):
    """Clean text by removing special characters, stopwords, and lemmatizing."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def get_sentiment(text):
    """Calculate sentiment polarity using TextBlob."""
    return TextBlob(text).sentiment.polarity

def vectorize_text(texts, max_features=5000, fit=True, vectorizer=None):
    """Convert texts to TF-IDF vectors."""
    if fit:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts)
    else:
        X = vectorizer.transform(texts)
    return X, vectorizer

def preprocess_data(train_file='train_data.txt', test_file='test_data.txt'):
    """Preprocess train and test data: clean, add sentiment, vectorize."""
    # Load data
    train_df, test_df = load_data(train_file, test_file)
    
    # Verify required columns
    required_columns_train = ['id', 'title', 'genre', 'plot']
    required_columns_test = ['id', 'plot']  # Test data requires only id and plot
    for df, name, cols in [(train_df, 'train', required_columns_train), (test_df, 'test', required_columns_test)]:
        missing_cols = [col for col in cols if col not in df.columns]
        if missing_cols:
            if name == 'test' and df.empty:
                logger.warning(f"Test data is empty or malformed, creating placeholder test_df")
                test_df = pd.DataFrame(columns=required_columns_test)
            else:
                logger.error(f"Missing columns in {name}_df: {missing_cols}")
                raise KeyError(f"Missing required columns in {name}_df: {missing_cols}")
    
    # Clean plots
    train_df['cleaned_plot'] = train_df['plot'].apply(clean_text)
    test_df['cleaned_plot'] = test_df['plot'].apply(clean_text)
    
    # Add sentiment
    train_df['sentiment'] = train_df['plot'].apply(get_sentiment)
    test_df['sentiment'] = test_df['plot'].apply(get_sentiment)
    
    # Vectorize
    X_train_tfidf, vectorizer = vectorize_text(train_df['cleaned_plot'], fit=True)
    X_test_tfidf, _ = vectorize_text(test_df['cleaned_plot'], fit=False, vectorizer=vectorizer)
    
    return train_df, test_df, X_train_tfidf, X_test_tfidf, vectorizer