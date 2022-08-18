from src.spotify.library import (
    get_tracks, parse_tracks,
    get_genres, get_track_features,
    get_missing_preview_urls, get_playlists,
    parse_playlist_genres, get_spotipy_instance,
    get_album_covers_for_playlists
)
from src.spotify.config import (
    TRACK_RAW_FILE,
    PLAYLIST_FILE, MAIN_DATA_FILE,
    PLAYLIST_GENRE_FILE,
    GENRE_WORD2VEC_EMBEDDING_FILE,
    GENRE_EVERYNOISE_EMBEDDING_FILE,
    EVERYNOISE_GENRE_SPACE,
    CO_OCCURRENCE_TABLE
)
from src.spotify.genre_embeddings import (
    get_word2vec_embeddings, download_everynoise_genre_space,
    get_everynoise_embeddings, get_genre_co_occurrence_model
)
from src.spotify.merge_data import merge_data
import os
import logging
import spotipy


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

MSG = "\n{}\n========"


def pipeline(sp: spotipy.client.Spotify):
    """
    Wrapper of pipeline to obtain all the data consumed by the streamlit dashboard.

    Parameters
    ----------
    sp : spotipy.client.Spotify

    Returns
    -------

    """
    if not os.path.exists(TRACK_RAW_FILE):
        logging.info(msg=MSG.format("Download tracks"))
        get_tracks(sp, verbose=1)
    logging.info(msg=MSG.format("Parsing tracks"))
    parse_tracks()
    # Works incrementally
    logging.info(msg=MSG.format("Download genres"))
    get_genres(sp, verbose=1)
    # Works incrementally
    logging.info(msg=MSG.format("Download audio features"))
    get_track_features(sp, verbose=1)
    # Works incrementally
    logging.info(msg=MSG.format("Download missing previews"))
    get_missing_preview_urls(sp, verbose=1)
    if not os.path.exists(PLAYLIST_FILE):
        logging.info(msg=MSG.format("Download playlists"))
        get_playlists(sp, verbose=1)
    if not os.path.exists(EVERYNOISE_GENRE_SPACE):
        logging.info(msg=MSG.format("Download Every Noise At Once Genre Space"))
        download_everynoise_genre_space()
    if not os.path.exists(GENRE_EVERYNOISE_EMBEDDING_FILE):
        logging.info(msg=MSG.format("Obtain everynoise genre embeddings"))
        get_everynoise_embeddings()
    if not os.path.exists(GENRE_WORD2VEC_EMBEDDING_FILE):
        logging.info(msg=MSG.format("Obtain word2vec genre embeddings"))
        get_word2vec_embeddings()
    if not os.path.exists(MAIN_DATA_FILE):
        logging.info(msg=MSG.format("Merge data"))
        merge_data()
    if not os.path.exists(PLAYLIST_GENRE_FILE):
        logging.info(msg=MSG.format("Obtain genre playlist profiles"))
        parse_playlist_genres(verbose=1)
    # Works incrementally => if songs were removed from a playlist
    # File needs to be overwritten
    logging.info(msg=MSG.format("Getting album cover samples"))
    get_album_covers_for_playlists(verbose=1)
    if not os.path.exists(CO_OCCURRENCE_TABLE):
        logging.info(msg=MSG.format("Get co-occurrence lookup table"))
        get_genre_co_occurrence_model()


if __name__ == '__main__':
    spotipy_instance = get_spotipy_instance()
    pipeline(spotipy_instance)
