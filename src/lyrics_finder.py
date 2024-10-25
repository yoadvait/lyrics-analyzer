import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import pandas as pd

client_id = "de68ee68278e466d82ae18ebade5ce7e"
client_secret = "8c0b96f3f3854e8cb352c8cdae666520"
api_key = "31a518f4199f09a6f2a39f4c223346e3"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

df = pd.read_csv('data/SingerAndSongs.csv')

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
        else:
            return "No lyrics found."
    else:
        return "No track ID found."

df['lyrics'] = df.apply(lambda row: get_lyrics(row['Song Name'], row['Singer']), axis=1)

df.to_csv('data/SingerAndSongs_with_lyrics.csv', index=False)
print("Lyrics fetched and saved successfully.")
