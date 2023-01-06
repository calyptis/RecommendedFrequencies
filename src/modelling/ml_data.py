import itertools
import os

import networkx as nx
import numpy as np
import pandas as pd

from src.modelling.config import (CATBOOST_FEATURES, EUCLIDEAN_FEAT_COLS,
                                  EVERYNOISE_FEAT_COL)
from src.spotify.config import (MAIN_DATA_FILE, PLAYLIST_FILE,
                                SIMILAR_PLAYLISTS_FILE, DATA_DIR)
from src.spotify.utils import read_pickle


def create_song_triplets() -> pd.DataFrame:
    """
    Create dataframe holding triplets of songs in addition to the playlist context.

    One song acts as a target and the other two as a negative and positive example, respectively.
    Negative examples are songs that come from very different playlists from the playlist the anchor song
    belongs to (as defined in SIMILAR_PLAYLISTS_FILE).
    This format can be consumed both by a siamese network but also easily converted to a tabular format for
    binary classification.
    Returns
    -------

    """
    # =============================================
    # Load main df source: songs & their attributes
    df = pd.read_pickle(MAIN_DATA_FILE)

    # ================================
    # Load playlists we should exclude
    # For example playlists that do not follow a structure, such as "Most Recently Added"
    exclude_playlist_path = os.path.join(DATA_DIR, "exclude_playlists.txt")
    if os.path.exists(exclude_playlist_path):
        with open(exclude_playlist_path, "r") as f:
            exclude_playlists = f.read().split("\n")
    else:
        exclude_playlists = []

    # ===================
    # Load playlist info
    list_playlists = read_pickle(PLAYLIST_FILE)
    # From: List[Dict[Name, ID, Tracks, Artists]]
    # To: Dict[Name]:Dict[Name, ID, Tracks, Artists]
    playlists = {
        playlist["name"][0]: {k: v for k, v in playlist.items() if k != "name"}
        for playlist in list_playlists
        if playlist["name"][0] not in exclude_playlists
    }

    # ===============================================================
    # Create graph (nodes = songs, edges = if nodes in same playlist)
    graph = create_graph(df, playlists)

    # ========================
    # Create positive examples
    pos_examples = np.array(
        [(i[0], i[1], i[2]["playlist"]) for i in graph.edges(data=True)]
    )
    # Add the same examples with the pairs switched
    switched_pos_examples = np.array(
        [(i[1], i[0], i[2]["playlist"]) for i in graph.edges(data=True)]
    )
    # Add some more positive examples => both inputs are the same song
    frac = 0.4
    tmp = np.random.choice(
        list(graph.nodes()), size=int(pos_examples.shape[0] * frac)
    ).flatten()
    # TODO: Pick a playlist the song appears instead of randomly choosing one
    addtl_pos_examples = np.array(
        [(i, i, np.random.choice(list(playlists.keys()))) for i in tmp]
    )
    pos_examples = np.concatenate(
        (pos_examples, switched_pos_examples, addtl_pos_examples)
    )
    print(f"Positive examples: {pos_examples.shape[0]:,}")

    # ========================
    # Create negative examples
    # Pairs of playlists that are similar
    df_similar_playlists = pd.read_csv(SIMILAR_PLAYLISTS_FILE)
    # Collect all similar playlists for a given playlist
    tmp_a = df_similar_playlists.groupby("playlist1").playlist2.agg(set)
    tmp_b = df_similar_playlists.groupby("playlist2").playlist1.agg(set)
    df_playlists_similarity = pd.concat((tmp_a, tmp_b)).reset_index()
    df_playlists_similarity.columns = ["playlist", "similar_playlists"]
    df_playlists_similarity.drop_duplicates(subset=["playlist"], inplace=True)
    # Find playlists that are different
    df_playlists_similarity[
        "different_playlists"
    ] = df_playlists_similarity.similar_playlists.apply(
        lambda x: list(set(playlists.keys()) - x)
    )
    df_playlists_similarity.set_index("playlist", inplace=True)
    # Finally, sample negative examples
    neg_examples = np.chararray((pos_examples.shape[0], 1), itemsize=100, unicode=True)
    for i in range(pos_examples.shape[0]):
        playlist_context = pos_examples[i, -1]
        if playlist_context in df_playlists_similarity.index:
            different_playlist = np.random.choice(
                df_playlists_similarity.loc[playlist_context, "different_playlists"]
            )
        else:
            different_playlist = np.random.choice(list(playlists.keys()))
        negative_example_song = np.random.choice(
            playlists[different_playlist]["tracks"]
        )
        neg_examples[i] = negative_example_song
    print(f"Negative examples: {neg_examples.shape[0]:,}")

    # =========================================================
    # Combine positive and negative examples into one dataframe
    df_examples = pd.DataFrame(
        np.hstack((pos_examples[:, :2], neg_examples, pos_examples[:, -1:])),
        columns=["anchor", "positive_example", "negative_example", "playlist"],
    )

    return df_examples  # .to_csv(ML_DATA_FILE, index=False)


def create_song_pair_features(df_triplets_examples: pd.DataFrame):
    """
    TODO: add docstring.

    Parameters
    ----------
    df_triplets_examples

    Returns
    -------

    """
    # Load main data source: songs & their attributes
    df = pd.read_pickle(MAIN_DATA_FILE)
    # df_triplets_examples = pd.read_csv(ML_DATA_FILE)
    df_pairs_examples = pd.concat(
        (
            df_triplets_examples[["anchor", "positive_example", "playlist"]]
            .rename(columns={"positive_example": "example"})
            .assign(target=1),
            df_triplets_examples[["anchor", "negative_example", "playlist"]]
            .rename(columns={"negative_example": "example"})
            .assign(target=0),
        )
    )
    # Random shuffle
    df_examples = df_pairs_examples.sample(frac=1)

    audio_features = df[EUCLIDEAN_FEAT_COLS].values
    genre_embedding = np.vstack(df[EVERYNOISE_FEAT_COL].values.tolist())
    features = np.hstack((audio_features, genre_embedding))
    # Fillna for missing genre embeddings
    df_features = pd.DataFrame(
        features, index=df.index, columns=CATBOOST_FEATURES
    ).fillna(0)

    df_songs_a_features = (
        df_features.loc[df_examples.anchor]
        .rename(columns={c: c + "_a" for c in df_features.columns})
        .reset_index()
    )
    df_songs_b_features = (
        df_features.loc[df_examples.example]
        .reset_index()
        .rename(columns={c: c + "_b" for c in df_features.columns})
    )
    df_for_model = pd.concat(
        (
            df_songs_a_features,
            df_songs_b_features,
            df_examples[["playlist"]].reset_index(drop=True),
            df_examples[["target"]].reset_index(drop=True),
        ),
        axis=1,
    ).set_index("ID")

    return df_for_model


def create_graph(df: pd.DataFrame, playlists: dict) -> nx.classes.graph.Graph:
    """
    Create a graph representing relationships amongst songs.

    Songs are nodes and connected via an edge if they belong to the same playlist.

    Parameters
    ----------
    df:
    playlists:

    Returns
    -------

    """
    # Get all edges
    edges = []
    for playlist_name, info in playlists.items():
        tracks = info["tracks"]
        # Only consider songs for which we have features
        available_tracks = set(tracks).intersection(set(df.index))
        info["tracks"] = list(available_tracks)
        playlists[playlist_name] = info
        # Create edges
        pairs = list(itertools.combinations(available_tracks, 2))
        edges.append([(*sorted(i), playlist_name) for i in pairs])
    edges = pd.DataFrame(
        [element for sub_lists in edges for element in sub_lists],
        columns=["song_1", "song_2", "playlist"],
    )

    # Get all nodes in the graph
    nodes = set(edges.song_1).union(set(edges.song_2))
    # There should be as many nodes as songs in our feature dataset
    assert len(nodes - set(df.index)) == 0

    # Create a graph representing the relationship between songs
    # in terms of their affiliation to playlists
    graph = nx.from_pandas_edgelist(
        edges, source="song_1", target="song_2", edge_attr="playlist"
    )
    print(f"Number of nodes: {len(graph.nodes):,}")
    print(f"Number of edges: {len(graph.edges):,}")

    # Node attribute is the list of playlist the song featured in
    node_attributes = (
        pd.concat(
            (
                edges[["song_1", "playlist"]].rename(columns={"song_1": "song"}),
                edges[["song_2", "playlist"]].rename(columns={"song_2": "song"}),
            )
        )
        .groupby("song")
        .playlist.agg(lambda x: "; ".join(set(x)))
        .reset_index()
    )
    nx.set_node_attributes(
        graph, values=node_attributes.set_index("song").to_dict(orient="index")
    )

    return graph
