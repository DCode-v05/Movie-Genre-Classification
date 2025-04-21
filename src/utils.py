import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.manifold import TSNE
import seaborn as sns
import os
from src.preprocess import preprocess_data

def plot_wordcloud(texts, genre, output_dir):
    """Generate word cloud for a specific genre."""
    text = ' '.join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud - {genre}')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'wordcloud_{genre.lower()}.png'))
    plt.close()

def plot_tsne(X, y, output_dir):
    """Generate t-SNE visualization of TF-IDF features."""
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X.toarray())
    df_tsne = pd.DataFrame({'x': X_tsne[:, 0], 'y': X_tsne[:, 1], 'genre': y})
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='x', y='y', hue='genre', data=df_tsne)
    plt.title('t-SNE Visualization of Movie Plots')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'tsne_genres.png'))
    plt.close()

def main():
    """Generate visualizations for train data."""
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(base_path, 'outputs')
    train_df, _, X_train_tfidf, _, _ = preprocess_data()
    
    # Example: Word cloud for 'drama'
    drama_plots = train_df[train_df['genre'] == 'drama']['cleaned_plot']
    if not drama_plots.empty:
        plot_wordcloud(drama_plots, 'Drama', output_dir)
    
    # t-SNE visualization
    plot_tsne(X_train_tfidf, train_df['genre'], output_dir)

if __name__ == '__main__':
    main()