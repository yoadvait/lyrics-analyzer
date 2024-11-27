import joblib
import re
from sklearn.linear_model import LinearRegression
import numpy as np

vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
acousticness_model = joblib.load('models/acousticness_model.pkl')
danceability_model = joblib.load('models/danceability_model.pkl')
energy_model = joblib.load('models/energy_model.pkl')

def clean_lyrics(lyrics):
    lyrics = lyrics.lower()
    lyrics = re.sub(r'[^a-zA-Z\s]', '', lyrics)
    lyrics = re.sub(r'\s+', ' ', lyrics).strip()
    return lyrics

def predict_song_features(lyrics):
    clean_lyrics_text = clean_lyrics(lyrics)
    lyrics_vector = vectorizer.transform([clean_lyrics_text])
    
    acousticness = acousticness_model.predict(lyrics_vector)[0]
    danceability = danceability_model.predict(lyrics_vector)[0]
    energy = energy_model.predict(lyrics_vector)[0]
    
    return {
        "Acousticness": round(acousticness, 4),
        "Danceability": round(danceability, 4),
        "Energy": round(energy, 4)
    }

if __name__ == "__main__":
    input_lyrics = """



 """
    predictions = predict_song_features(input_lyrics)
    print("Predicted Features:")
    for feature, value in predictions.items():
        print(f"{feature}: {value}")