import sys
import os
import re
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from textblob import TextBlob

# Import load_data from preprocess.py
from src.preprocess import load_data

def extract_year(title):
    """Extract year from movie title, e.g., 'Movie (2009)' -> 2009."""
    match = re.search(r'\((\d{4})\)', title)
    return int(match.group(1)) if match else None

def plot_genre_distribution(df, output_dir):
    """Plot and save genre distribution bar chart."""
    plt.figure(figsize=(12, 6))
    genre_counts = df['genre'].value_counts().head(10)
    sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='viridis')
    plt.title('Top 10 Movie Genres in Training Data', fontsize=14, weight='bold')
    plt.xlabel('Number of Movies', fontsize=12)
    plt.ylabel('Genre', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    for i, v in enumerate(genre_counts.values):
        plt.text(v + 0.5, i, str(v), va='center', fontsize=10)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'genre_distribution.png'), dpi=300)
    plt.close()

def plot_length_distribution(df, output_dir):
    """Plot and save distribution of plot description lengths."""
    df['plot_length'] = df['plot'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['plot_length'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Plot Description Lengths', fontsize=14, weight='bold')
    plt.xlabel('Word Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'plot_length_distribution.png'), dpi=300)
    plt.close()

def plot_genre_cooccurrence(df, output_dir):
    """Plot and save genre co-occurrence heatmap."""
    df['genres'] = df['genre'].apply(lambda x: x.split(',') if ',' in x else [x])
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['genres'])
    co_matrix = pd.DataFrame(genre_matrix.T @ genre_matrix, index=mlb.classes_, columns=mlb.classes_)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(co_matrix, cmap='YlGnBu', square=True, annot=True, fmt='d', cbar_kws={'label': 'Co-occurrence Count'})
    plt.title('Genre Co-occurrence Heatmap', fontsize=14, weight='bold')
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Genre', fontsize=12)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'genre_cooccurrence_heatmap.png'), dpi=300)
    plt.close()

def plot_sentiment_distribution(df, output_dir):
    """Plot and save distribution of plot sentiment scores."""
    df['sentiment'] = df['plot'].apply(lambda x: TextBlob(x).sentiment.polarity)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sentiment'], bins=30, kde=True, color='salmon')
    plt.title('Distribution of Plot Sentiment Scores', fontsize=14, weight='bold')
    plt.xlabel('Sentiment Polarity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'), dpi=300)
    plt.close()

def plot_length_by_genre(df, output_dir):
    """Plot and save boxplot of plot lengths by genre."""
    df['plot_length'] = df['plot'].apply(lambda x: len(str(x).split()))
    top_genres = df['genre'].value_counts().head(10).index
    df_top = df[df['genre'].isin(top_genres)]
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='genre', y='plot_length', data=df_top, palette='muted')
    plt.title('Plot Length Distribution by Top 10 Genres', fontsize=14, weight='bold')
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Plot Length (Words)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'plot_length_by_genre.png'), dpi=300)
    plt.close()

def plot_sentiment_by_genre(df, output_dir):
    """Plot and save violin plot of sentiment by genre."""
    df['sentiment'] = df['plot'].apply(lambda x: TextBlob(x).sentiment.polarity)
    top_genres = df['genre'].value_counts().head(10).index
    df_top = df[df['genre'].isin(top_genres)]
    
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='genre', y='sentiment', data=df_top, palette='Set2')
    plt.title('Sentiment Distribution by Top 10 Genres', fontsize=14, weight='bold')
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Sentiment Polarity', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'sentiment_by_genre.png'), dpi=300)
    plt.close()

def plot_genre_over_time(df, output_dir):
    """Plot and save genre frequency over time (by title year) with visible x-axis labels."""
    df['year'] = df['title'].apply(extract_year)
    df = df.dropna(subset=['year'])
    top_genres = df['genre'].value_counts().head(5).index
    df_top = df[df['genre'].isin(top_genres)]
    
    plt.figure(figsize=(16, 8))  # Increased width to accommodate labels
    sns.countplot(x='year', hue='genre', data=df_top, palette='tab10')
    plt.title('Genre Frequency Over Time (Top 5 Genres)', fontsize=14, weight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Movies', fontsize=12)
    plt.legend(title='Genre', title_fontsize=12, labelspacing=1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Limit and rotate x-axis labels for visibility
    years = sorted(df_top['year'].unique())
    step = max(1, len(years) // 10)  # Adjust step size based on number of years
    plt.xticks(ticks=range(0, len(years), step), labels=[str(years[i]) for i in range(0, len(years), step)], rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'genre_over_time.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run exploratory data analysis and save visualizations."""
    # Set Seaborn style for professional look
    sns.set_style('whitegrid')
    
    # Define paths
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(base_path, 'outputs')
    
    # Load train data for EDA
    train_df, _ = load_data('train_data.txt', 'test_data.txt')
    
    # Generate plots
    plot_genre_distribution(train_df, output_dir)
    plot_length_distribution(train_df, output_dir)
    plot_genre_cooccurrence(train_df, output_dir)
    plot_sentiment_distribution(train_df, output_dir)
    plot_length_by_genre(train_df, output_dir)
    plot_sentiment_by_genre(train_df, output_dir)
    plot_genre_over_time(train_df, output_dir)

if __name__ == '__main__':
    main()