# Lyrics-Based Acoustic Feature Prediction

This project predicts the acoustic features—**acousticness**, **danceability**, and **energy**—of songs based on their lyrics. Using machine learning models trained on a dataset of song lyrics and corresponding acoustic feature values, the project aims to uncover relationships between text data and audio characteristics. The pipeline also allows for real-time prediction of acoustic features based on user-provided lyrics.

---

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Setup and Installation](#setup-and-installation)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Prediction](#prediction)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Read More](#read-more)

---

## Introduction

This project explores the correlation between song lyrics and their acoustic properties. By using text processing techniques and various machine learning models, it aims to predict three acoustic features:
- **Acousticness**: The likelihood a song is acoustic.
- **Danceability**: How suitable a track is for dancing.
- **Energy**: The intensity and activity of a song.

The trained models can predict these values based solely on lyrics, providing insights into how lyrical content relates to a song's sound.

---

## Project Structure

```bash
├── src/
│   ├── predictor.py                        # Script to predict acoustic features from input lyrics
│   ├── vectorizer.py                       # Script for lyrics preprocessing and vectorization
│   ├── model_trainer(Linear+Others).py     # Model training script for each acoustic feature
├── models/
│   ├── acousticness_model.pkl              # Trained model for acousticness prediction
│   ├── danceability_model.pkl              # Trained model for danceability prediction
│   ├── energy_model.pkl                    # Trained model for energy prediction
├── testing.ipynb                           
├── .gitignore                         
├── requirements.txt
└── README.md
```
## Dataset

The dataset used in this project includes:
1. **Song Metadata**: Basic song information such as the song title and artist.
2. **Acoustic Features**: Acousticness, danceability, and energy values.
3. **Lyrics**: The primary feature used for prediction.

Ensure your dataset, `SingerAndSongs.csv`, is saved in the `data/` folder with columns such as `Song name`, `Singer`, and acoustic features.

---

## Setup and Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/lyrics-acoustic-feature-prediction.git
   cd lyrics-acoustic-feature-prediction
   ```
   
2. **Install Required Packages**: Make sure to have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set Up API Credentials:**
   - Register for Spotify and Musixmatch API keys.
   - Save the credentials in lyrics_finder.py or use environment variables for secure storage.
  
## Feature Extraction

Run the `lyrics_finder.py` script to fetch and add lyrics to the dataset. This script uses the Musixmatch API to retrieve lyrics for each song and saves them to a CSV file:
```bash
python lyrics_finder.py
```
This will create a new dataset file `SingerAndSongs_with_lyrics.csv` with an additional `lyrics` column.

## Model Training

1. **Vectorize Lyrics**: Run `vectorizer.py` to preprocess lyrics and convert them into TF-IDF vectors.

```bash
python vectorizer.py
```
This generates:

  - `tfidf_vectorizer.pkl`: The TF-IDF vectorizer saved for later use.
  - `split_data.pkl`: The vectorized dataset split into train and test sets.
2. **Train Models**: Run `model_trainer.py` to train models for each target feature (acousticness, danceability, energy) using Linear Regression, Ridge Regression, and Gradient Boosting.

```bash
python model_trainer.py
```
Each model and its corresponding RMSE value will be saved in the models folder.

## Prediction
To predict acoustic features for new lyrics, run predictor.py. This script takes lyrics as input and outputs the predicted values for each feature:

```bash
python predictor.py
```
Upon execution, it will prompt for lyrics input and return predicted values for acousticness, danceability, and energy.


## Evaluation
The `model_trainer.py` script prints RMSE values for each model, providing insights into model performance:

Root Mean Squared Error (RMSE) is used to evaluate model accuracy.

## Usage
This project can be used to predict acoustic characteristics from song lyrics, allowing:

 - Lyric-Based Feature Prediction: Using lyrics to estimate acoustic traits.
 - Lyric Analysis: Exploring how lyrics impact a song’s audio profile.
 - Music Recommendations: Filtering songs based on lyrical content to match desired acoustic profiles.

## Results

The models have been tested using the RMSE metric, achieving the following results(for Linear Regression):

 - Acousticness RMSE: 0.5444
 - Danceability RMSE: 0.3982
 - Energy RMSE: 0.4611
   
The Linear, Ridge, and Gradient Boosting models were evaluated, with Ridge and Gradient Boosting showing improvement over Linear Regression.

## Read More

Check out my blog on Hashnode - https://yoadvait.hashnode.dev/lyricsanalyzer-song-analysis-with-nlp