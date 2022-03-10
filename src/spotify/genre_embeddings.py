from typing import Tuple
import fasttext
import pandas as pd
import numpy as np
import os
import re
from urllib import request
from bs4 import BeautifulSoup
from collections import Counter
from functools import reduce
import itertools
import pickle
from src.spotify.config import (
    GENRE_FILE, GENRE_WORD2VEC_EMBEDDING_FILE,
    GENRE_EVERYNOISE_EMBEDDING_FILE, EVERYNOISE_GENRE_SPACE,
    MAIN_DATA_FILE, PLAYLIST_FILE, CO_OCCURRENCE_TABLE,
    CO_OCCURRENCE_FILE
)
from src.project_config import MODEL_DIR, DATA_DIR
from src.spotify.utils import (
    concat_lists, read_pickle
)


def get_word2vec_embeddings() -> None:
    """
    Obtain word2vec embeddings for genres of artists.

    Parameters
    ----------

    Returns
    -------
    Saves genre word vectors to disk.
    """
    genres_df = pd.read_csv(GENRE_FILE)
    genres_df["GenreList"] = genres_df["GenreList"].apply(eval)
    # Create genre string for which to obtain embeddings
    genres_df["GenreString"] = genres_df["GenreList"].apply(" ".join)
    ft = fasttext.load_model(os.path.join(MODEL_DIR, 'cc.en.300.bin'))
    genres_df["GenreWord2VecEmbedding"] = genres_df["GenreString"].apply(lambda x: ft.get_sentence_vector(x))
    # Save as JSON because some column values are lists
    genres_df.to_json(GENRE_WORD2VEC_EMBEDDING_FILE)


def download_everynoise_genre_space() -> None:
    """
    Parse genre positions in the embedded genre space as reported on
        https://everynoise.com/

    Parameters
    ----------

    Returns
    -------

    """
    url_body = request.urlopen("https://everynoise.com/").read()
    soup = BeautifulSoup(url_body, 'html.parser')
    # All the genres in scatter plot
    genres = soup.findAll('div', id=lambda x: x and x.startswith('item'))
    genres_positions = [_get_position(i) for i in genres]
    df_genres_positions = pd.DataFrame(genres_positions, columns=["genre", "x", "y"])
    df_genres_positions.to_csv(EVERYNOISE_GENRE_SPACE, index=False)


def get_everynoise_embeddings() -> None:
    genres_df = pd.read_csv(GENRE_FILE)
    genre_space_df = pd.read_csv(EVERYNOISE_GENRE_SPACE).set_index("genre")
    genres_df["GenreList"] = genres_df["GenreList"].apply(eval)
    genres_df["GenreEveryNoiseEmbedding"] = (
        genres_df["GenreList"].apply(lambda x: np.nanmean(genre_space_df.loc[x].values, axis=0))
    )
    # Save as JSON because some column values are lists
    genres_df.to_pickle(GENRE_EVERYNOISE_EMBEDDING_FILE)


def get_genre_co_occurrence_model(debug: bool = True) -> None:
    """
    Calculate genre co-occurrences:
        - For each playlist
            - Get list of genres of each artist in the playlist
            - Get all combinations of any pair of 2 genres in the playlist
                - Calculate how often each pair occurs
        - Do this for each playlist and sum occurrences together

    Parameters
    ----------

    Returns
    -------
    Saves co-occurrence model as pd.DataFrame to disk.
    """
    data = pd.read_pickle(MAIN_DATA_FILE)
    playlists = read_pickle(PLAYLIST_FILE)
    exclude_playlist_path = os.path.join(DATA_DIR, "exclude_playlists.txt")
    if os.path.exists(exclude_playlist_path):
        with open(exclude_playlist_path, "r") as f:
            exclude_playlists = f.read().split("\n")
    else:
        exclude_playlists = []
    playlists = {
        playlist["name"][0]: playlist
        for playlist in playlists
        if playlist["name"][0] not in exclude_playlists
    }
    playlist_counters = {}
    for k, v in playlists.items():
        genres = data.loc[data.index.intersection(v["tracks"])].GenreSet.apply(list).values
        genres = concat_lists(genres)
        pairs = [tuple(sorted(i)) for i in itertools.combinations(genres, 2)]
        playlist_counters[k] = Counter(pairs)
    counter = dict(reduce(lambda a, b: a + b, playlist_counters.values()))
    co_occurrence_model = pd.DataFrame(counter, index=[0]).T
    co_occurrence_model.to_csv(CO_OCCURRENCE_TABLE)
    if debug:
        pickle.dump(playlist_counters, open(CO_OCCURRENCE_FILE, "wb"))


def _get_position(item: BeautifulSoup.element.Tag) -> Tuple[str, int, int]:
    """
    Parse a genre HTML tag on everynoise.com

    Parameters
    ----------
    item : BeautifulSoup.element.Tag

    Returns
    -------

    """
    genre_name = item.get("onclick").split(",")[1].strip().replace('"', "")
    y = int(re.search("top: (\d+)px", item.get("style")).group(1))
    x = int(re.search("left: (\d+)px", item.get("style")).group(1))
    return genre_name, x, y
