import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_distances as cosine
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import time
from joblib import Parallel, delayed
from src.modelling.utils import (
    set_similarity, co_occurrence_similarity,
)
from src.spotify.utils import (
    concat_lists, get_frequent_genres
)
from src.spotify.config import MAIN_DATA_FILE
from src.modelling.config import (
    EUCLIDEAN_FEAT_COLS, WORD2VEC_FEAT_COLS, GENRE_COL, EVERYNOISE_FEAT_COL
)
from src.project_config import DATA_DIR


def get_baseline(playlist_features: pd.DataFrame, songs_available_for_suggestion_features: pd.DataFrame,
                 genre_similarity: str = None) -> pd.DataFrame:
    """
    Calculate weighted Euclidean distance and genre distance between songs in the library but not in the playlist
    and those in the playlist.
    Euclidean distance weights are the inverse of the standard deviation of playlist features.
    This means that similarity for features that are fairly similar for songs within a playlist is rewarded.
    For genre distance, three similarity metrics are supported:
        - Cosine distance of word2vec embeddings
        - Set similarity
        - Co-occurrence similarity

    Parameters
    ----------
    playlist_features : pd.DataFrame
        Features of songs in the playlist
    songs_available_for_suggestion_features : pd.DataFrame
        Features of songs not in the playlist
    genre_similarity : str
        The kind of similarity metric to use in order to capture genre similarity.
        Currently, only these are supported: word2vec, co-occurrence and set similarity.

    Returns
    -------
    df_similarity : pd.DataFrame
        Songs and their similarity score
    """

    # Euclidean similarity
    # ====================
    playlist_euclidean_centroid = (
        playlist_features
        [EUCLIDEAN_FEAT_COLS]
        .agg(["mean", "std"])
        .values
    )

    euclidean_distances = cdist(
        XA=playlist_euclidean_centroid[:1, :],
        XB=songs_available_for_suggestion_features[EUCLIDEAN_FEAT_COLS].values,
        # If standard deviation is large => then small weight for feature dim
        # Apply log so that features with std close to zero do not outweigh other features
        # in distance metric.
        # Note that features are bound between 0 and 1, so std is never 1.
        w=np.log(1 / (playlist_euclidean_centroid[-1, :] + 1e-7))
    ).flatten()

    # Normalise distance metrics
    euclidean_distances = MinMaxScaler().fit_transform(euclidean_distances.reshape(-1, 1)).flatten()

    # Genre similarity
    # ================
    if genre_similarity is not None:

        if genre_similarity == "word2vec":
            playlist_genre_centroid = (
                np.vstack(playlist_features[WORD2VEC_FEAT_COLS].values.tolist())
                .mean(axis=0)
                .reshape(1, -1)
            )
            genre_distances = cosine(
                X=playlist_genre_centroid,
                Y=np.vstack(songs_available_for_suggestion_features[WORD2VEC_FEAT_COLS].values.tolist())
            ).flatten()

        elif genre_similarity == "everynoise":
            playlist_genre_centroid = np.nanmean(np.vstack(playlist_features[EVERYNOISE_FEAT_COL].values), axis=0)
            playlist_genre_centroid = playlist_genre_centroid.reshape(1, -1)
            song_centroids = np.vstack(songs_available_for_suggestion_features[EVERYNOISE_FEAT_COL].values)
            genre_distances = cdist(
                XA=playlist_genre_centroid,
                XB=song_centroids,
            ).flatten()

        elif genre_similarity == "set-similarity":
            playlist_genre_centroid = set(concat_lists(playlist_features[GENRE_COL].values.tolist()))
            values = songs_available_for_suggestion_features[GENRE_COL].values.tolist()
            genre_distances = np.array(
                [set_similarity(playlist_genre_centroid, i) for i in values]
            )

        elif genre_similarity == "weighted set-similarity":
            # https://mathoverflow.net/questions/123339/weighted-jaccard-similarity
            # genre_occurrence = get_frequent_genres(DATA_DIR, thr=0, return_type="df")
            # playlist_genre_centroid = set(concat_lists(playlist_features[GENRE_COL].values.tolist()))
            # values = songs_available_for_suggestion_features[GENRE_COL].values.tolist()
            # genre_distances = np.array(
            #     [weighted_set_similarity(playlist_genre_centroid, i, genre_occurrence) for i in values]
            # )
            raise Exception("Not yet implemented")

        elif genre_similarity == "co-occurrence":
            # Add some print statements here to understand what could be optimised
            # as this option takes by far the longest.
            t0 = time.time()
            print("Import co-occurrence model")
            co_occurrence_lookup = pd.read_csv(
                os.path.join(DATA_DIR, "song_genre_co_occurrence_lookup_table.csv"),
                index_col=[0, 1]
            )
            co_occurrence_lookup.columns = ["CoOccurrenceValue"]
            frq_genres = get_frequent_genres(MAIN_DATA_FILE)
            print(f"Finished loading co-occurrence model in {time.time() - t0:.2f} sec")
            print("Starting co-occurrence calculations")
            t1 = time.time()
            playlist_genre_centroid = list(
                set(concat_lists(playlist_features[GENRE_COL].values.tolist())).intersection(set(frq_genres))
            )
            values = list(zip(
                songs_available_for_suggestion_features[GENRE_COL].index.tolist(),
                songs_available_for_suggestion_features[GENRE_COL]
                .apply(lambda x: list(set(x).intersection(set(frq_genres))))
                .values.tolist()
            ))
            results = Parallel(n_jobs=8)(
                delayed(co_occurrence_similarity)(co_occurrence_lookup, playlist_genre_centroid, i) for i in values
            )
            # results = [
            #     co_occurrence_similarity(co_occurrence_lookup, playlist_genre_centroid, i)
            #     for i in values
            # ]
            df_results = pd.DataFrame(results, columns=["ID", "GenreCoOccurrence"]).set_index("ID")
            # Sort results so that they align with songs_available_for_suggestion_features.index
            df_results = df_results.loc[songs_available_for_suggestion_features.index]
            genre_distances = df_results.GenreCoOccurrence.values
            print(f"Finished co-occurrence calculations in {time.time() - t1:.2f} sec")

        else:
            raise Exception("Selected genre similarity measure is not supported.")

        # Normalise distance metrics
        genre_distances = np.nan_to_num(genre_distances, nan=0)
        genre_distances = MinMaxScaler().fit_transform(genre_distances.reshape(-1, 1)).flatten()

        if genre_similarity not in ("word2vec", "everynoise"):
            # Measures genre similarity => so need to transform to distance
            genre_distances = 1 - genre_distances

    else:
        genre_distances = np.zeros(euclidean_distances.shape)

    # Combine two metrics
    # ===================
    df_similarity = pd.DataFrame(
        {
            "EuclideanDistance": euclidean_distances,
            "GenreDistance": genre_distances
        },
        index=songs_available_for_suggestion_features.index
    )

    return df_similarity


def get_top_results(results: pd.DataFrame, genre_weight: int = 0, n: int = 10) -> pd.DataFrame:
    """
    Limit the results to a selected number of most similar songs.

    Parameters
    ----------
    results : pd.DataFrame
        Songs and their similarity scores
    genre_weight : float
        How much genre similarity matters for selecting the most similar songs.
    n : int
        Number of songs to show.

    Returns
    -------

    """
    euclidean_weight = 1 - genre_weight
    top_n = (
        results
        .assign(
            Distance=lambda row: (euclidean_weight * row.EuclideanDistance) + (genre_weight * row.GenreDistance)
        )
        .sort_values(by="Distance", ascending=True)
        .head(n)
    )
    return top_n
