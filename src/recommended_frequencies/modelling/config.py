"""Variables for ML training and inference."""

import os

from recommended_frequencies.config import MODEL_DIR, PREPARED_DATA_DIR

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
    "GenreEveryNoiseEmbeddingY",
]

EVERYNOISE_FEAT_COL = "GenreEveryNoiseEmbedding"

CATBOOST_FEATURES = EUCLIDEAN_FEAT_COLS

ML_DATA_FILE = os.path.join(PREPARED_DATA_DIR, "ml_data.csv")

CATBOOST_MODEL_FILE = os.path.join(MODEL_DIR, "Catboost.cbm")

# Sampling data
ADDTL_POS_FRAC = 0.4
TRAIN_FRAC = 0.8
