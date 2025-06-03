import os

from src.project_config import MODEL_DIR, PREPARED_DATA_DIR

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
    "AlbumReleaseYear",
    "GenreEveryNoiseEmbeddingX",
    "GenreEveryNoiseEmbeddingY"
]

EVERYNOISE_FEAT_COL = "GenreEveryNoiseEmbedding"

CATBOOST_FEATURES = EUCLIDEAN_FEAT_COLS

ML_DATA_FILE = os.path.join(PREPARED_DATA_DIR, "ml_data.csv")

CATBOOST_MODEL_FILE = os.path.join(MODEL_DIR, "Catboost.cbm")
