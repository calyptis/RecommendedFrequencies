import os

from recommended_frequencies.config import (
    RAW_DATA_DIR,
    PREPARED_DATA_DIR,
    CREATED_DATA_DIR,
)


class SpotifyFiles:
    """Spotify related files."""

    TRACK_RAW_FILE = os.path.join(RAW_DATA_DIR, "tracks.jsonl")
    PREVIEW_FILE = os.path.join(RAW_DATA_DIR, "additional_preview_urls.jsonl")
    GENRE_FILE = os.path.join(RAW_DATA_DIR, "genres.jsonl")
    TRACK_FEAT_FILE = os.path.join(RAW_DATA_DIR, "tracks_features.jsonl")
    TRACK_PARSED_FILE = os.path.join(PREPARED_DATA_DIR, "tracks.jsonl")
    PLAYLIST_FILE = os.path.join(RAW_DATA_DIR, "playlists.pickle")
    SIMILAR_PLAYLISTS_FILE = os.path.join(CREATED_DATA_DIR, "similar_playlists.csv")
    MAIN_DATA_FILE = os.path.join(PREPARED_DATA_DIR, "data.pickle")
    PLAYLIST_GENRE_FILE = os.path.join(PREPARED_DATA_DIR, "playlist_genres.jsonl")
    GENRE_EMBEDDING_SPACE = os.path.join(
        PREPARED_DATA_DIR, "genre_artist_embeddings_everynoise.pickle"
    )
    GENRE_COORDINATES = os.path.join(RAW_DATA_DIR, "genre_space_everynoise.jsonl")
    ALBUM_COVER_FILE = os.path.join(RAW_DATA_DIR, "playlist_album_covers.pickle")


EVERYNOISE_URL = "https://everynoise.com/"
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
