import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('data/SingerAndSongs.csv')

def clean_lyrics(lyrics):
    if isinstance(lyrics, str):
        lyrics = lyrics.lower()
        return lyrics
    return ""

df['clean_lyrics'] = df['lyrics'].apply(clean_lyrics)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_lyrics'])

joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

X_train, X_test, y_train, y_test = train_test_split(X, df[['acousticness', 'danceability', 'energy']], test_size=0.2, random_state=42)

joblib.dump((X_train, X_test, y_train, y_test), 'data/split_data.pkl')

print("Vectorization completed and data saved.")
