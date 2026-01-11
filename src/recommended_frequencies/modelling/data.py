import itertools
import os

from loguru import logger
import networkx as nx
import numpy as np
import pandas as pd

from recommended_frequencies.modelling.config import CATBOOST_FEATURES, ADDTL_POS_FRAC
from recommended_frequencies.spotify.config import (
    MAIN_DATA_FILE,
    PLAYLIST_FILE,
    SIMILAR_PLAYLISTS_FILE,
)
from recommended_frequencies.config import CREATED_DATA_DIR
from recommended_frequencies.spotify.utils import read_pickle
from recommended_frequencies.streamlit.config import FILE_ADDITIONAL_TRAINING_EXAMPLES


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
    df_examples : pd.DataFrame :
        The created triplets as a pandas DataFrame.
    """
    # =============================================
    # Load main df source: songs & their attributes
    df = pd.read_pickle(MAIN_DATA_FILE)

    # ================================
    # Load playlists we should exclude
    # For example playlists that do not follow a structure, such as "Most Recently Added"
    exclude_playlist_path = os.path.join(CREATED_DATA_DIR, "exclude_playlists.txt")
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

    tmp = np.random.choice(
        list(graph.nodes()), size=int(pos_examples.shape[0] * ADDTL_POS_FRAC)
    ).flatten()
    # TODO: Pick a playlist the song appears instead of randomly choosing one
    addtl_pos_examples = np.array(
        [(i, i, np.random.choice(list(playlists.keys()))) for i in tmp]
    )
    pos_examples = np.concatenate(
        (pos_examples, switched_pos_examples, addtl_pos_examples)
    )
    logger.info(f"Positive examples: {pos_examples.shape[0]:,}")

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
        if playlists[different_playlist]["tracks"]:
            negative_example_song = np.random.choice(
                playlists[different_playlist]["tracks"]
            )
            neg_examples[i] = negative_example_song
        else:
            # When this error is raised, check for peculiarities in the playlist
            # or if one wishes to simply proceed without considering the playlist
            # add the name of the playlist to data/exclude_playlists.txt
            raise Exception(f"No tracks for playlist {different_playlist}")

    logger.info(f"Negative examples: {neg_examples.shape[0]:,}")

    # =========================================================
    # Combine positive and negative examples into one dataframe
    df_examples = pd.DataFrame(
        np.hstack((pos_examples[:, :2], neg_examples, pos_examples[:, -1:])),
        columns=["anchor", "positive_example", "negative_example", "playlist"],
    )

    # ===============================================================
    # Adding negative examples that were selected in the streamlit app
    # This is useful because users can re-train the model on explicitely selected negative pairs
    df_additional_negative_training_examples = pd.read_csv(
        FILE_ADDITIONAL_TRAINING_EXAMPLES
    ).drop_duplicates()
    # Some house-keeping (over-writing file with removed duplicates)
    df_additional_negative_training_examples.to_csv(
        FILE_ADDITIONAL_TRAINING_EXAMPLES, index=False
    )
    to_add = []
    for playlist, g in df_additional_negative_training_examples.groupby(
        "playlist_name"
    ):
        # For each playlist, get the number of negative examples recorded through the streamlit app
        negative_examples = g["song_id"].values
        # Sample as many positive examples and anchors from the pool of data points we have collected so far
        # NOTE: anchor and positive example will occur more than once, but negative example is new
        tmp = df_examples.query(f"playlist == '{playlist}'").sample(
            n=len(negative_examples)
        )
        tmp["negative_example"] = negative_examples
        to_add.append(tmp)
    df_to_add = pd.concat(to_add)
    logger.info(
        f"Adding {len(df_to_add):,} additional negative examples collected as feedback from the app."
    )

    df_examples = pd.concat((df_to_add, df_examples))

    return df_examples  # .to_csv(ML_DATA_FILE, index=False)


def create_song_pair_features(df_triples_examples: pd.DataFrame) -> pd.DataFrame:
    """
    Convert song triplets to pairwise features for binary classification.

    Transforms triplet examples (anchor, positive, negative) into song pairs with
    binary labels (1 for positive pairs, 0 for negative pairs). Each pair is enriched
    with audio features and genre embeddings for both songs, suffixed with '_a' and '_b'.

    Parameters
    ----------
    df_triples_examples : pd.DataFrame
        DataFrame containing columns: 'anchor', 'positive_example', 'negative_example', 'playlist'.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by song ID with columns for both songs' features (suffixed '_a', '_b'),
        'playlist' context, and 'target' label (1=similar, 0=dissimilar).
    """
    # Load main data source: songs & their attributes
    df = pd.read_pickle(MAIN_DATA_FILE)
    df_pairs_examples = pd.concat(
        (
            df_triples_examples[["anchor", "positive_example", "playlist"]]
            .rename(columns={"positive_example": "example"})
            .assign(target=1),
            df_triples_examples[["anchor", "negative_example", "playlist"]]
            .rename(columns={"negative_example": "example"})
            .assign(target=0),
        )
    )
    # Random shuffle
    df_examples = df_pairs_examples.sample(frac=1)

    features = df[CATBOOST_FEATURES].values

    # Fillna for missing genre embeddings
    df_features = pd.DataFrame(
        features, index=df.index, columns=CATBOOST_FEATURES
    ).fillna(0)

    # Ensure that we only consider examples for which we have features
    # TODO: Figure out why it can happen that a song ID is not in df.index
    df_examples = df_examples.loc[
        lambda x: x["anchor"].isin(df.index) & x["example"].isin(df.index)
    ].copy()

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
    df: pd.DataFrame :
        Dataframe storing song features.
    playlists: dict :
        Dictionary mapping playlist name to song list.

    Returns
    -------
    graph: nx.classes.graph.Graph :
        Networkx graph representing relationships amongst songs.
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
    nodes = set(edges.song_1) | (set(edges.song_2))
    # There should be as many nodes as songs in our feature dataset
    assert len(nodes - set(df.index)) == 0

    # Create a graph representing the relationship between songs
    # in terms of their affiliation to playlists
    graph = nx.from_pandas_edgelist(
        edges, source="song_1", target="song_2", edge_attr="playlist"
    )

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
