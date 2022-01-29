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
    "AlbumReleaseYear"
]

RADIAL_COLS_PRETTY = [i.capitalize() if i != "AlbumReleaseYear" else "Year of Album Release" for i in RADIAL_COLS]
