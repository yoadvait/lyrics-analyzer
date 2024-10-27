from lyrics_finder import get_lyrics
from vectorizer import clean_lyrics, vectorize_and_split_data
from model_trainer import train_and_evaluate_models
import pandas as pd

df = pd.read_csv('data/SingerAndSongs.csv')
df['lyrics'] = df.apply(lambda row: get_lyrics(row['Song Name'], row['Singer']), axis=1)
df.to_csv('data/SingerAndSongs_with_lyrics.csv', index=False)
print("Lyrics fetched and saved successfully.")

df['clean_lyrics'] = df['lyrics'].apply(clean_lyrics)
X_train, X_test, y_train, y_test, vectorizer = vectorize_and_split_data(df, 'clean_lyrics', ['acousticness', 'danceability', 'energy'])

import joblib
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
joblib.dump((X_train, X_test, y_train, y_test), 'data/split_data.pkl')
print("Vectorization completed and data saved.")

train_and_evaluate_models(X_train, X_test, y_train, y_test)
print("Training and evaluation completed.")