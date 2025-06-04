from functools import reduce

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from recommended_frequencies.spotify.config import (GENRE_EVERYNOISE_EMBEDDING_FILE,
                                                    MAIN_DATA_FILE, PREVIEW_FILE, TRACK_FEAT_FILE,
                                                    TRACK_PARSED_FILE)


def merge_data():
    """Merge all data sources into a single file that can be used to calculate song similarities."""
    genres_everynoise_df = pd.read_pickle(GENRE_EVERYNOISE_EMBEDDING_FILE)

    tracks_df = pd.read_csv(TRACK_PARSED_FILE).set_index("ID")
    tracks_df["AlbumReleaseYear"] = pd.to_datetime(tracks_df.AlbumReleaseDate).dt.year

    features_df = pd.read_csv(TRACK_FEAT_FILE).set_index("ID")
    features_df.drop(
        [
            "type",
            "uri",
            "track_href",
            "analysis_url",
            "duration_ms",
            "time_signature",
            "mode",
            "key",
        ],
        axis=1,
        inplace=True,
    )

    # Merge track data-frame to get year of release
    features_df = features_df.join(
        tracks_df[
            [
                "AlbumReleaseYear",
                "ArtistID",
                "SongName",
                "Artist",
                "PreviewURL",
                "AlbumCover",
            ]
        ],
        how="inner",
    )
    # Note that songs can have more than one artist => get list of artist IDs
    features_df["ArtistIDList"] = features_df.ArtistID.str.split("|")

    # Merge artist genre embeddings
    tmp = (
        features_df.loc[:, ["ArtistIDList"]]
        # One row per artist in list of artist IDs
        .explode("ArtistIDList")
        .rename(columns={"ArtistIDList": "ArtistID"})
        # Note that we can have multiple rows for a given track ID
        .reset_index()
    )
    # Get genre profile of each artist
    tmp = tmp.merge(
        genres_everynoise_df[["ArtistID", "GenreEveryNoiseEmbedding", "GenreList"]],
        on="ArtistID",
        how="left",
    )
    # As a track can be from a collaboration of artists with a different genre profile,
    # stack all the genre embeddings of each artist involved in a track and average them
    tmp = (
        tmp.dropna()
        .groupby("ID")
        .agg(
            GenreEveryNoiseEmbedding=(
                "GenreEveryNoiseEmbedding",
                lambda g: np.stack(g).mean(axis=0),
            ),
            GenreList=("GenreList", lambda x: reduce(lambda a, b: a + b, x)),
        )
    )
    tmp["missing_everynoise_genre"] = tmp.GenreEveryNoiseEmbedding.apply(
        lambda x: pd.isnull(x).sum() > 0
    )
    data = features_df.join(tmp, how="left")
    data.missing_everynoise_genre.fillna(True, inplace=True)
    # If genre information is missing, set the embedding to the null vector
    data["GenreEveryNoiseEmbedding"] = (
        data
        .GenreEveryNoiseEmbedding
        .apply(lambda x: [0, 0] if x is np.nan or np.isnan(x).sum() == 2 else x)
    )

    # Get one column for each dimension in EveryNoise embedding
    data["GenreEveryNoiseEmbeddingX"] = data["GenreEveryNoiseEmbedding"].apply(lambda x: x[0])
    data["GenreEveryNoiseEmbeddingY"] = data["GenreEveryNoiseEmbedding"].apply(lambda x: x[1])
    data.drop(columns=["GenreEveryNoiseEmbedding"], inplace=True)

    # In case of collaborations, only care about unique occurrences of genres
    data["GenreSet"] = data.GenreList.apply(lambda x: set(x) if isinstance(x, list) else {})

    # Normalize features that are not yet normalized
    cols_to_scale = ["tempo", "loudness", "AlbumReleaseYear", "GenreEveryNoiseEmbeddingX", "GenreEveryNoiseEmbeddingY"]
    for c in cols_to_scale:
        data[c] = MinMaxScaler().fit_transform(data[c].values.reshape(-1, 1))

    # Replace missing preview URLs
    additional_preview_urls = (
        pd.read_csv(PREVIEW_FILE)
        .set_index("ID")
        .rename(columns={"PreviewURL": "PreviewURLAdditional"})
    )
    n = len(data)
    data = data.join(additional_preview_urls, how="left")
    assert len(data) == n
    data.PreviewURL.fillna(data.PreviewURLAdditional, inplace=True)
    data.drop("PreviewURLAdditional", axis=1, inplace=True)

    # Noise due to syncing Spotify with Apple Music => duplicated songs
    data.drop_duplicates(subset=["SongName", "Artist"], inplace=True)

    # For some artists, no genre info is available, e.g. John Newman "34v5MVKeQnIo0CWYMbbrPf"
    # remove these songs
    # data = data.query("~missing_everynoise_genre")

    data.to_pickle(MAIN_DATA_FILE)


if __name__ == '__main__':
    merge_data()
