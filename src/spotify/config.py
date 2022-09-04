import os
from src.project_config import DATA_DIR

TRACK_RAW_FILE = os.path.join(DATA_DIR, "tracks_raw.pickle")
PREVIEW_FILE = os.path.join(DATA_DIR, "additional_preview_urls.csv")
GENRE_FILE = os.path.join(DATA_DIR, "genres.csv")
TRACK_FEAT_FILE = os.path.join(DATA_DIR, "track_features.csv")
TRACK_PARSED_FILE = os.path.join(DATA_DIR, "tracks.csv")
PLAYLIST_FILE = os.path.join(DATA_DIR, "playlists.pickle")
SIMILAR_PLAYLISTS_FILE = os.path.join(DATA_DIR, "similar_playlists.csv")
MAIN_DATA_FILE = os.path.join(DATA_DIR, "data.pickle")
PLAYLIST_GENRE_FILE = os.path.join(DATA_DIR, "playlist_genres.json")
GENRE_WORD2VEC_EMBEDDING_FILE = os.path.join(DATA_DIR, "genre_embeddings_word2vec.json")
# Use pickle: Json converts NaN to None
GENRE_EVERYNOISE_EMBEDDING_FILE = os.path.join(DATA_DIR, "genre_embeddings_everynoise.pickle")
EVERYNOISE_GENRE_SPACE = os.path.join(DATA_DIR, "genre_space_everynoise.csv")
CO_OCCURRENCE_TABLE = os.path.join(DATA_DIR, "genre_co_occurrence_lookup_table.csv")
CO_OCCURRENCE_FILE = os.path.join(DATA_DIR, "genre_co_occurrence_values_by_playlist.pickle")
ALBUM_COVER_FILE = os.path.join(DATA_DIR, "playlist_album_covers.pickle")
