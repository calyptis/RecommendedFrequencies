import os
from src.project_config import DATA_DIR, MODEL_DIR


EUCLIDEAN_FEAT_COLS = [
    # Spotify audio features
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "loudness",
    # Spotify song metadata
    "AlbumReleaseYear"
]

EVERYNOISE_FEAT_COL = "GenreEveryNoiseEmbedding"

GENRE_COL = "GenreList"

CATBOOST_FEATURES = EUCLIDEAN_FEAT_COLS + ["genre_x", "genre_y"]

ML_DATA_FILE = os.path.join(DATA_DIR, "ml_data.csv")

CATBOOST_MODEL_FILE = os.path.join(MODEL_DIR, "Catboost.cbm")
