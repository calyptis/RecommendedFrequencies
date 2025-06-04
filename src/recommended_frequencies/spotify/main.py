import argparse
import logging
import os

import spotipy

from recommended_frequencies.spotify.config import (EVERYNOISE_GENRE_SPACE,
                                                    GENRE_EVERYNOISE_EMBEDDING_FILE,
                                                    MAIN_DATA_FILE, PLAYLIST_FILE,
                                                    PLAYLIST_GENRE_FILE, TRACK_RAW_FILE)
from recommended_frequencies.spotify.genre_embeddings import (download_everynoise_genre_space,
                                                              get_everynoise_embeddings)
from recommended_frequencies.spotify.library import (get_album_covers_for_playlists, get_genres,
                                                     get_missing_preview_urls,
                                                     get_spotipy_instance, get_track_features,
                                                     get_tracks, parse_playlist_genres,
                                                     parse_tracks, get_playlists)
from recommended_frequencies.spotify.merge_data import merge_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ],
)

MSG = "{}"

parser = argparse.ArgumentParser(
    prog="Download a user's Spotify library",
    description="Download saved tracks, their genres and audio features as well as playlist information.",
)
parser.add_argument('--overwrite', default=False, action=argparse.BooleanOptionalAction)


def pipeline(sp: spotipy.client.Spotify, overwrite: bool = False):
    """
    Wrap all functions into a pipeline to obtain all the data consumed by the streamlit dashboard.

    Parameters
    ----------
    sp : spotipy.client.Spotify
    overwrite : bool :
        Whether to overwrite existing data or not.

    """
    if not os.path.exists(TRACK_RAW_FILE) or overwrite:
        logging.info(msg=MSG.format("Download tracks"))
        get_tracks(sp, verbose=1)
    if not os.path.exists(PLAYLIST_FILE) or overwrite:
        logging.info(msg=MSG.format("Downloading playlists"))
        get_playlists(sp)
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
    if not os.path.exists(EVERYNOISE_GENRE_SPACE) or overwrite:
        logging.info(msg=MSG.format("Download Every Noise At Once Genre Space"))
        download_everynoise_genre_space()
    if not os.path.exists(GENRE_EVERYNOISE_EMBEDDING_FILE) or overwrite:
        logging.info(msg=MSG.format("Obtain Every Noise At Once Genre embeddings"))
        get_everynoise_embeddings()
    if not os.path.exists(MAIN_DATA_FILE) or overwrite:
        logging.info(msg=MSG.format("Merge data"))
        merge_data()
    if not os.path.exists(PLAYLIST_GENRE_FILE) or overwrite:
        logging.info(msg=MSG.format("Obtain genre playlist profiles"))
        parse_playlist_genres(verbose=1)
    # Works incrementally => if songs were removed from a playlist
    # File needs to be overwritten
    logging.info(msg=MSG.format("Getting album cover samples"))
    get_album_covers_for_playlists(verbose=1)


if __name__ == "__main__":
    spotipy_instance = get_spotipy_instance()
    args = parser.parse_args()
    pipeline(spotipy_instance, args.overwrite)
