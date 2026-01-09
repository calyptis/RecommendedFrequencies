RADIAL_COLS = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "loudness",
    "AlbumReleaseYear",
    "GenreEveryNoiseEmbeddingX",
    "GenreEveryNoiseEmbeddingY",
]

D_MAPPING = {
    "AlbumReleaseYear": "Year of Album Release",
    "GenreEveryNoiseEmbeddingX": "Genre Emb x-dim",
    "GenreEveryNoiseEmbeddingY": "Genre Emb y-dim",
}

RADIAL_COLS_PRETTY = [D_MAPPING.get(i, i.capitalize()) for i in RADIAL_COLS]
