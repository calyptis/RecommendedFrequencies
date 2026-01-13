import argparse
import os

from loguru import logger
import spotipy

from recommended_frequencies.spotify.config import SpotifyFiles
from recommended_frequencies.spotify.genre_embeddings import (
    download_everynoise_genre_space,
    get_everynoise_embeddings,
)
from recommended_frequencies.spotify.library import (
    get_album_covers_for_playlists,
    get_genres,
    get_missing_preview_urls,
    get_spotipy_instance,
    get_track_features,
    get_tracks,
    parse_playlist_genres,
    parse_tracks,
    get_playlists,
)
from recommended_frequencies.spotify.merge_data import merge_data

parser = argparse.ArgumentParser(
    prog="Download a user's Spotify library",
    description="Download saved tracks, their genres and audio features as well as playlist information.",
)
parser.add_argument("--overwrite", default=False, action=argparse.BooleanOptionalAction)


def pipeline(sp: spotipy.client.Spotify, overwrite: bool = False):
    """
    Wrap all functions into a pipeline to obtain all the data consumed by the streamlit dashboard.

    Parameters
    ----------
    sp : spotipy.client.Spotify
    overwrite : bool :
        Whether to overwrite existing data or not.
    """
    if not os.path.exists(SpotifyFiles.TRACK_RAW_FILE) or overwrite:
        get_tracks(sp)
    if not os.path.exists(SpotifyFiles.PLAYLIST_FILE) or overwrite:
        get_playlists(sp)
    # No need to make it work incrementally => is fast
    parse_tracks()
    # Works incrementally
    get_genres(sp)
    # Works incrementally
    get_track_features(sp)
    # Works incrementally
    logger.info("Download missing previews")
    get_missing_preview_urls(sp)
    if not os.path.exists(SpotifyFiles.GENRE_COORDINATES) or overwrite:
        logger.info("Download Every Noise At Once Genre Space")
        download_everynoise_genre_space()
    if not os.path.exists(SpotifyFiles.GENRE_EMBEDDING_SPACE) or overwrite:
        logger.info("Obtain Every Noise At Once Genre embeddings")
        get_everynoise_embeddings()
    if not os.path.exists(SpotifyFiles.MAIN_DATA_FILE) or overwrite:
        logger.info("Merge data")
        merge_data()
    if not os.path.exists(SpotifyFiles.PLAYLIST_GENRE_FILE) or overwrite:
        logger.info("Obtain genre playlist profiles")
        parse_playlist_genres()
    # Works incrementally however if songs were removed from a playlist this file needs to be overwritten
    logger.info("Getting album cover samples")
    get_album_covers_for_playlists(verbose=1)


if __name__ == "__main__":
    spotipy_instance = get_spotipy_instance()
    args = parser.parse_args()
    pipeline(spotipy_instance, args.overwrite)
