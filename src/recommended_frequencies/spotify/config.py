import os

from recommended_frequencies.config import (
    RAW_DATA_DIR,
    PREPARED_DATA_DIR,
    CREATED_DATA_DIR,
)


# Files
TRACK_RAW_FILE = os.path.join(RAW_DATA_DIR, "tracks.pickle")
PREVIEW_FILE = os.path.join(RAW_DATA_DIR, "additional_preview_urls.csv")
GENRE_FILE = os.path.join(RAW_DATA_DIR, "genres.csv")
TRACK_FEAT_FILE = os.path.join(RAW_DATA_DIR, "track_features.csv")
TRACK_PARSED_FILE = os.path.join(PREPARED_DATA_DIR, "tracks.csv")
PLAYLIST_FILE = os.path.join(RAW_DATA_DIR, "playlists.pickle")
SIMILAR_PLAYLISTS_FILE = os.path.join(CREATED_DATA_DIR, "similar_playlists.csv")
MAIN_DATA_FILE = os.path.join(PREPARED_DATA_DIR, "data.pickle")
PLAYLIST_GENRE_FILE = os.path.join(PREPARED_DATA_DIR, "playlist_genres.json")
GENRE_EVERYNOISE_EMBEDDING_FILE = os.path.join(
    PREPARED_DATA_DIR, "genre_artist_embeddings_everynoise.pickle"
)
EVERYNOISE_GENRE_SPACE = os.path.join(RAW_DATA_DIR, "genre_space_everynoise.csv")
ALBUM_COVER_FILE = os.path.join(RAW_DATA_DIR, "playlist_album_covers.pickle")
