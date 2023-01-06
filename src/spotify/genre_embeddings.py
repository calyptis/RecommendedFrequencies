import re
from typing import Tuple
from urllib import request

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from src.spotify.config import (EVERYNOISE_GENRE_SPACE,
                                GENRE_EVERYNOISE_EMBEDDING_FILE, GENRE_FILE)


def download_everynoise_genre_space() -> None:
    """Parse genre positions in the embedded genre space as reported on https://everynoise.com/."""
    url_body = request.urlopen("https://everynoise.com/").read()
    soup = BeautifulSoup(url_body, "html.parser")
    # All the genres in scatter plot
    genres = soup.findAll("div", id=lambda x: x and x.startswith("item"))
    genres_positions = [_get_position(i) for i in genres]
    df_genres_positions = pd.DataFrame(genres_positions, columns=["genre", "x", "y"])
    df_genres_positions.to_csv(EVERYNOISE_GENRE_SPACE, index=False)


def get_everynoise_embeddings() -> None:
    """Given a list of genres for an artist, calculate the centroid across all their (x, y) coordinates."""
    genres_df = pd.read_csv(GENRE_FILE)
    genre_space_df = pd.read_csv(EVERYNOISE_GENRE_SPACE).set_index("genre")
    genres_df["GenreList"] = genres_df["GenreList"].apply(eval)
    genres_df["GenreEveryNoiseEmbedding"] = genres_df["GenreList"].apply(
        lambda x: np.nanmean(genre_space_df.loc[x].values, axis=0)
    )
    # Save as JSON because some column values are lists
    genres_df.to_pickle(GENRE_EVERYNOISE_EMBEDDING_FILE)


def _get_position(item) -> Tuple[str, int, int]:
    """
    Parse a genre HTML tag on everynoise.com.

    Parameters
    ----------
    item : BeautifulSoup.element.Tag
    """
    genre_name = item.get("onclick").split(",")[1].strip().replace('"', "")
    y = int(re.search("top: (\d+)px", item.get("style")).group(1))
    x = int(re.search("left: (\d+)px", item.get("style")).group(1))
    return genre_name, x, y
