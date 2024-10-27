import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import pandas as pd
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client_id = "de68ee68278e466d82ae18ebade5ce7e"
client_secret = "8c0b96f3f3854e8cb352c8cdae666520"
api_key = "31a518f4199f09a6f2a39f4c223346e3"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

df = pd.read_csv('data/SingerAndSongs.csv')

def get_lyrics(track_name, artist_name):
    logging.info(f"Fetching lyrics for track: '{track_name}' by artist: '{artist_name}'")
    
    endpoint = "https://api.musixmatch.com/ws/1.1/track.search"
    params = {
        'q_track': track_name,
        'q_artist': artist_name,
        'apikey': api_key,
        'format': 'json'
    }
    
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        logging.info("API response received successfully.")

        if data['message']['header']['status_code'] == 200 and data['message']['body']['track_list']:
            track_id = data['message']['body']['track_list'][0]['track']['track_id']
            logging.info(f"Track ID found: {track_id}")
            
            lyrics_response = requests.get(f'https://api.musixmatch.com/ws/1.1/track.lyrics.get?track_id={track_id}&apikey={api_key}')
            lyrics_response.raise_for_status()
            lyrics_data = lyrics_response.json()

            if 'lyrics' in lyrics_data['message']['body']:
                lyrics = lyrics_data['message']['body']['lyrics']['lyrics_body']
                lyrics = re.split(r"\n\*+ This Lyrics is NOT for Commercial use \*+\n|\(\d+\)$", lyrics)[0]
                return lyrics.strip()
            else:
                logging.warning(f"No lyrics found for track ID: {track_id}.")
                return None
        else:
            logging.warning(f"No track ID found for '{track_name}' by '{artist_name}'.")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return None

lyrics_list = []

for index, row in df.iterrows():
    lyrics = get_lyrics(row['Song name'], row['Singer'])
    if lyrics is not None:
        lyrics_list.append(lyrics)
    else:
        lyrics_list.append(None)

df['lyrics'] = lyrics_list
df = df[df['lyrics'].notna()]

df = df[['lyrics'] + [col for col in df.columns if col != 'lyrics']]

df.to_csv('data/SingerAndSongs_with_lyrics.csv', index=False)
logging.info("Lyrics fetching complete. Data saved to 'data/SingerAndSongs_with_lyrics.csv'.")