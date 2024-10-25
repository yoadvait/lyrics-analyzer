# app.py
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import re
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_id = "de68ee68278e466d82ae18ebade5ce7e"
client_secret = "8c0b96f3f3854e8cb352c8cdae666520"
api_key = "31a518f4199f09a6f2a39f4c223346e3"
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

# Load data
df = pd.read_csv('data/SingerAndSongs.csv')

# Step 1: Fetch lyrics using Musixmatch API
def get_lyrics(track_name, artist_name):
    endpoint = "https://api.musixmatch.com/ws/1.1/track.search"
    params = {
        'q_track': track_name,
        'q_artist': artist_name,
        'apikey': api_key,
        'format': 'json'
    }
    response = requests.get(endpoint, params=params)
    data = response.json()
    
    if data['message']['header']['status_code'] == 200 and data['message']['body']['track_list']:
        track_id = data['message']['body']['track_list'][0]['track']['track_id']
        lyrics_response = requests.get(f'https://api.musixmatch.com/ws/1.1/track.lyrics.get?track_id={track_id}&apikey={api_key}')
        lyrics_data = lyrics_response.json()
        
        if 'lyrics' in lyrics_data['message']['body']:
            return lyrics_data['message']['body']['lyrics']['lyrics_body']
    return "No lyrics found."

# Apply lyrics fetching and save data with lyrics
df['lyrics'] = df.apply(lambda row: get_lyrics(row['Song Name'], row['Singer']), axis=1)
df.to_csv('data/SingerAndSongs_with_lyrics.csv', index=False)
print("Lyrics fetched and saved successfully.")

# Step 2: Clean lyrics
def clean_lyrics(lyrics):
    if isinstance(lyrics, str):
        lyrics = lyrics.lower()
        lyrics = re.sub(r'[^a-zA-Z\s]', '', lyrics)
        lyrics = re.sub(r'\s+', ' ', lyrics).strip()
        return lyrics
    return ""

df['clean_lyrics'] = df['lyrics'].apply(clean_lyrics)

# Step 3: Vectorize lyrics
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_lyrics'])
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, df[['acousticness', 'danceability', 'energy']], test_size=0.2, random_state=42)
joblib.dump((X_train, X_test, y_train, y_test), 'data/split_data.pkl')
print("Vectorization completed and data split saved.")

# Step 4: Train models
def train_and_save_model(model, X_train, y_train, label, model_name):
    model.fit(X_train, y_train[label])
    joblib.dump(model, f'models/{model_name}.pkl')
    return model

# Initialize and train models for each label
acousticness_model = train_and_save_model(LinearRegression(), X_train, y_train, 'acousticness', 'acousticness_model')
danceability_model = train_and_save_model(LinearRegression(), X_train, y_train, 'danceability', 'danceability_model')
energy_model = train_and_save_model(LinearRegression(), X_train, y_train, 'energy', 'energy_model')

# Evaluate models
def evaluate_model(model, X_test, y_test, label):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test[label], y_pred))
    print(f"{label.capitalize()} RMSE: {rmse:.4f}")

evaluate_model(acousticness_model, X_test, y_test, 'acousticness')
evaluate_model(danceability_model, X_test, y_test, 'danceability')
evaluate_model(energy_model, X_test, y_test, 'energy')

print("Models trained and saved successfully.")
