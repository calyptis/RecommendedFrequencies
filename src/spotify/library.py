import time
import pandas as pd
import os
import json
import argparse
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import numpy as np
import pickle
import logging
from skimage.io import imread
from src.project_config import CREDENTIALS, SCOPES
from src.spotify.utils import string_similarity, timeout_wrapper, read_pickle
from src.spotify.config import (
    TRACK_RAW_FILE, PREVIEW_FILE, GENRE_FILE,
    TRACK_FEAT_FILE, PLAYLIST_FILE, MAIN_DATA_FILE,
    PLAYLIST_GENRE_FILE, TRACK_PARSED_FILE,
    ALBUM_COVER_FILE
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)


def get_spotipy_instance():
    """
    Get a spotipy instance used to make Spotify queries.

    Returns
    -------
    sp_instance : spotipy.client.Spotify
    """
    sp_instance = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=CREDENTIALS["client_id"],
            client_secret=CREDENTIALS["client_secret"],
            redirect_uri=CREDENTIALS["redirect_uri"],
            scope=SCOPES
        )
    )
    return sp_instance


def get_tracks(sp, verbose=0):
    """
    Get all songs in a user's library.

    Parameters
    ----------
    sp : spotipy.client.Spotify
    verbose : int

    Returns
    -------

    """
    limit = 20
    offset = 0
    update_interval = 60
    results = []
    while True:
        # Saved tracks == liked songs
        library_extract = timeout_wrapper(sp.current_user_saved_tracks(limit=limit, offset=offset))
        if not library_extract:
            continue
        if len(library_extract["items"]) == 0:
            # Already queried all tracks in the library
            break
        else:
            for entry in library_extract["items"]:
                results.append(entry)
            offset += len(library_extract["items"])
        if offset % update_interval == 0:
            if verbose >= 1:
                logging.info(f"Downloaded {offset} tracks")
            pickle.dump(results, open(TRACK_PARSED_FILE, "ab+"))
            time.sleep(0.2)  # Make sure not too many API calls are made in a short amount of time
            results = []
    pickle.dump(results, open(TRACK_PARSED_FILE, "ab+"))


def parse_tracks():
    """
    Parse track information, i.e. select on the most relevant track attributes.

    Parameters
    ----------

    Returns
    -------

    """
    results_raw = read_pickle(TRACK_RAW_FILE)
    results_parsed = [_parse_track_info(entry) for entry in results_raw]
    library = pd.DataFrame(results_parsed)
    library.to_csv(TRACK_PARSED_FILE, index=False)


def get_missing_preview_urls(sp, verbose=0):
    """
    When getting all tracks in a user's library, sometimes some preview URLs are missing.
    One solution to this is to search for the song again.
    Note that not all songs have a preview, so this function looks for the closes match
    in terms of string similarity that has a preview url.

    Parameters
    ----------
    sp : spotipy.client.Spotify
    verbose : int

    Returns
    -------
    """
    # Read in previously downloaded track information
    library = pd.read_csv(TRACK_PARSED_FILE)
    mask = library.PreviewURL.isnull()
    # Get relevant information on songs with missing preview URL
    df = library.loc[mask, ["ID", "SongName", "Artist"]]
    results = []
    count = 0
    update_interval = 50
    for _, row in df.iterrows():
        count += 1
        query = " ".join([row.SongName, row.Artist.replace(" | ", " ")])
        track = timeout_wrapper(sp.search(query, limit=10, offset=0, type='track', market=None))
        if not track:
            continue
        items = track["tracks"]["items"]
        # Filter out matches without preview url and calculate string similarity
        matched_items = [
            (
                item["preview_url"],
                string_similarity(
                    # Input
                    query,
                    # Parse artist and name from returned item
                    " ".join([item["name"], " ".join(i["name"] for i in item["artists"])]))
            )
            for item in items
            if item["preview_url"] is not None
        ]
        if matched_items:
            # Get match that is most similar to input
            matched_items.sort(key=lambda x: x[1], reverse=True)
            best_match_preview_url = matched_items[0][0]
            results.append((row.ID, best_match_preview_url))
        if count % update_interval == 0:
            if verbose >= 1:
                logging.info(f"Downloaded previews for {count} tracks")
            flag = os.path.exists(PREVIEW_FILE) is False
            df_results = pd.DataFrame(results, columns=["ID", "PreviewURL"])
            df_results.to_csv(PREVIEW_FILE, index=False, mode="a" if ~flag else "w", header=flag)
            results = []  # Clear memory

    df_results = pd.DataFrame(results, columns=["ID", "PreviewURL"])
    df_results.to_csv(PREVIEW_FILE, index=False, mode="a", header=False)


def get_genres(sp, verbose=0):
    """
    Obtain genres of artists in a user's library.

    Parameters
    ----------
    sp : spotipy.client.Spotify
    verbose : int

    Returns
    -------

    """
    library = pd.read_csv(TRACK_PARSED_FILE)
    # Flatten the list of lists of artists in a user's library
    artist_ids = list(set(np.concatenate(library.ArtistID.str.split("|").values)))
    results = []
    count = 0
    update_interval = 50
    for artist_id in artist_ids:
        count += 1
        entry = timeout_wrapper(sp.artist(artist_id))
        if not entry:
            continue
        parsed_entry = {
                "GenreList": entry["genres"],
                "Artist": entry["name"],
                "ArtistID": artist_id
            }
        results.append(parsed_entry)
        if count % update_interval == 0:
            if verbose >= 1:
                logging.info(f"Downloaded genre information for {count} artists")
            flag = os.path.exists(GENRE_FILE) is False
            artist_genres = pd.DataFrame(results)
            artist_genres.to_csv(GENRE_FILE, index=False, header=flag, mode="a" if ~flag else "w")
            results = []  # Clear memory

    artist_genres = pd.DataFrame(results)
    artist_genres.to_csv(GENRE_FILE, index=False, mode="a", header=False)


def get_track_features(sp, library=None, verbose=0):
    """
    Obtain high-level audio features for tracks in a user's library.
    For information on the available features, see:
        https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features

    Parameters
    ----------
    sp : spotipy.client.Spotify
    library : None or pd.DataFrame
        None if querying the entire library, otherwise specify the subset of the library.
    verbose : int

    Returns
    -------

    """
    if library is None:
        library = pd.read_csv(TRACK_PARSED_FILE)

    if os.path.exists(TRACK_FEAT_FILE):
        already_downloaded_track_feat = pd.read_csv(TRACK_FEAT_FILE)
        already_downloaded_ids = already_downloaded_track_feat.ID
    else:
        already_downloaded_ids = []

    to_download_ids = list(set(library.ID) - set(already_downloaded_ids))

    features = []
    count = 0
    update_interval = 50
    for i, track in enumerate(to_download_ids):
        count += 1
        try:
            # For more detailed features, use sp.audio_analysis
            audio_features = timeout_wrapper(sp.audio_features(track))
            if not audio_features:
                continue
            features.append(audio_features)
        except Exception as e:
            if verbose >= 1:
                print(f"Issue {e} with {track}")
            pass
        if count % update_interval == 0:
            if verbose >= 1:
                logging.info(f"Downloaded audio features for {count} tracks")
            features_df = pd.DataFrame([i[0] for i in features if isinstance(i[0], dict)]).rename(columns={"id": "ID"})
            flag = os.path.exists(TRACK_FEAT_FILE) is False
            features_df.to_csv(TRACK_FEAT_FILE, mode='a' if ~flag else "w", header=flag, index=False)
            features = []  # Clear memory
    features_df = pd.DataFrame([i[0] for i in features if isinstance(i[0], dict)]).rename(columns={"id": "ID"})
    features_df.to_csv(TRACK_FEAT_FILE, mode='a', header=False, index=False)


def get_playlists(sp, user_id=None, out_file=PLAYLIST_FILE, verbose=0):
    """
    Obtain all playlists that a user has in their library.

    Parameters
    ----------
    sp : spotipy.client.Spotify
    user_id : str
    out_file : str
    verbose : int

    Returns
    -------

    """
    if not user_id:
        user_id = sp.me()['id']
    playlists = []
    offset = 0
    update_interval = 50
    while True:
        response = timeout_wrapper(sp.user_playlists(user_id, limit=50, offset=offset))
        if not response:
            continue
        items = response["items"]
        if len(items) == 0:
            break
        else:
            for v in items:
                response = _get_playlist_tracks(sp, v["id"])
                tracks = list(map(lambda x: x[0], response))
                artists = list(map(lambda x: x[1], response))
                playlists += [{
                    "name": [v["name"]],
                    "id": v["id"],
                    "tracks": tracks,
                    "artists": artists
                }]
            offset += len(items)
        if (offset % update_interval == 0) & (offset != 0):
            if verbose >= 0:
                logging.info(f"Processed {offset:.0f} playlists")
            pickle.dump(playlists, open(out_file, "ab+"))
            playlists = []  # Clear memory
    pickle.dump(playlists, open(out_file, "ab+"))


def parse_playlist_genres(verbose: int = 0):
    """
    Combine all the genres associated to artists in a playlist.

    Parameters
    ----------
    verbose : int
    Returns
    -------

    """
    playlists = pickle.load(open(PLAYLIST_FILE, "rb"))
    data = pickle.load(open(MAIN_DATA_FILE, "rb"))
    updated_playlist = {}
    for playlist_info in playlists:
        playlist_name = playlist_info["name"][0]
        if verbose >= 1:
            logging.info(f"Parsing genres for {playlist_name}")
        # Identify tracks that are both in the library and playlist
        # Because not saved songs can still be part of playlists
        common_tracks = list(set(playlist_info["tracks"]).intersection(set(data.index)))
        playlist_info["genres"] = list(set(np.concatenate(data.loc[common_tracks].GenreList.values)))
        updated_playlist[playlist_name] = playlist_info
    json.dump(updated_playlist, open(PLAYLIST_GENRE_FILE, "w"))


def get_album_covers_for_playlists(verbose=0):
    playlists = pickle.load(open(PLAYLIST_FILE, "rb"))
    data = pickle.load(open(MAIN_DATA_FILE, "rb"))
    items = []
    count = 0
    update_interval = 3
    for playlist_info in playlists:
        count += 1
        playlist_id = playlist_info["id"]
        playlist_name = playlist_info["name"][0]
        playlist_tracks = list(set(playlist_info["tracks"]).intersection(set(data.index)))
        flag = len(playlist_tracks) < 10
        playlist_tracks_sample = list(np.random.choice(playlist_tracks, size=10, replace=flag))
        playlist_album_cover_urls = data.loc[playlist_tracks_sample, "AlbumCover"].values.tolist()
        playlist_album_cover_images = [imread(url) for url in playlist_album_cover_urls]
        items += [(playlist_name, playlist_id, playlist_album_cover_images)]
        if count % update_interval == 0:
            if verbose >= 1:
                logging.info(f"Downloaded selection of album covers for {count} playlists.")
            pickle.dump(items, open(ALBUM_COVER_FILE, "ab+"))
            items = []  # Clear memory
    pickle.dump(items, open(ALBUM_COVER_FILE, "ab+"))


def _get_playlist_tracks(sp, playlist_id):
    """
    Obtain all track IDs of the songs included in a specified playlist.

    Parameters
    ----------
    sp : spotipy.client.Spotify
    playlist_id : str
        Playlist ID for which all tracks should be obtained.

    Returns
    -------
    playlist_tracks : list
        Track IDs that are contained in the playlist
    """
    playlist_tracks = []
    offset = 0
    while True:
        response = timeout_wrapper(sp.playlist_items(
            playlist_id,
            offset=offset,
            fields="items.track.id,items.track.artists.id",
            additional_types=['track']
        ))
        if not response.get("items"):
            break
        offset += len(response['items'])
        current_track_extract = []
        for i in response["items"]:
            try:
                current_track_extract += [(i["track"]["id"], [j["id"] for j in i["track"]["artists"]])]
            except TypeError:
                pass
        playlist_tracks = playlist_tracks + current_track_extract
    return playlist_tracks


def _parse_track_info(track_info):
    """
    Parses object returned from Spotify's API query.

    Parameters
    ----------
    track_info : dict

    Returns
    -------
    parsed_track_info : dict
        Parsed entry that only keeps the most relevant information from the original return object
    """
    track_info = track_info["track"]
    parsed_track_info = {
        "Artist": " | ".join([m.get("name") for m in track_info.get("artists")]),
        "SongName": track_info.get("name"),
        "Album": track_info.get("album").get("name"),
        "ID": track_info.get("id"),
        "URI": track_info.get("uri"),
        "Popularity": track_info.get("popularity"),
        "AlbumReleaseDate": track_info.get("album").get("release_date"),
        # Combine artists using the pipe symbol in case a song is from a collaboration
        "ArtistID": "|".join([m.get("id") for m in track_info.get("artists")]),
        "PreviewURL": track_info.get("preview_url"),
        # Not all tracks have an album cover image. Get the first one if available.
        "AlbumCover": next(iter(track_info.get("album").get("images")), {"url": None}).get("url")
    }
    return parsed_track_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # noinspection PyTypeChecker
    parser.add_argument("--func", type=str, nargs=None, default="get_tracks")
    args = parser.parse_args()
    spotipy_instance = get_spotipy_instance()
    if args.func not in ["merge_data", "get_playlist_genres", "parse_tracks"]:
        globals()[args.func](sp=spotipy_instance)
    else:
        globals()[args.func]()
