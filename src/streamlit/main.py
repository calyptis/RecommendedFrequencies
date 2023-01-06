import os
from io import BytesIO

import catboost
import pandas as pd

import streamlit as st
from src.modelling.config import CATBOOST_MODEL_FILE, EUCLIDEAN_FEAT_COLS
from src.modelling.ml_catboost import (create_train_test_split,
                                       get_catboost_predictions,
                                       train_catboost)
from src.modelling.ml_data import (create_song_pair_features,
                                   create_song_triplets)
from src.modelling.utils import get_top_results
from src.plotting.album_cover_collage import plot_album_covers
from src.plotting.mood_board import plot_mood_board, plot_radial_plot
from src.project_config import DATA_DIR
from src.spotify.config import ALBUM_COVER_FILE, MAIN_DATA_FILE, PLAYLIST_FILE
from src.spotify.utils import read_pickle
from src.streamlit.utils import make_clickable_html

COL_ORDER = ["PreviewURL", "SongName", "Artist", "ID"]

SUGGESTION_ENGINES = {
    "Catboost": get_catboost_predictions,
}

GENRE_SIMILARITY = [
    "everynoise",
]
SIMILAR_SONGS_STEPS = 10

# If certain d_playlists should be excluded from the dashboard
# they need to be listed in the file below
exclude_playlist_path = os.path.join(DATA_DIR, "exclude_playlists.txt")
if os.path.exists(exclude_playlist_path):
    with open(exclude_playlist_path, "r") as f:
        exclude_playlists = f.read().split("\n")
else:
    exclude_playlists = []

# In case it is easier to just list the d_playlists that a user should
# be able to choose from
show_playlist_path = os.path.join(DATA_DIR, "show_playlists.txt")
if os.path.exists(show_playlist_path):
    with open(show_playlist_path, "r") as f:
        show_playlists = f.read().split("\n")
else:
    show_playlists = []


app_title = "Recommended Frequencies"
st.set_page_config(page_title=app_title, page_icon="🎵", layout="wide")


# --------- Data
@st.cache
def get_playlists():
    """Load playlists from disk."""
    list_playlists = read_pickle(PLAYLIST_FILE)
    dict_playlists = {playlist["name"][0]: playlist for playlist in list_playlists}
    return dict_playlists


@st.cache(allow_output_mutation=True)
def get_features():
    """Load features from disk."""
    df_features = pd.read_pickle(MAIN_DATA_FILE)
    df_features.index.name = "ID"
    df_features.Artist = df_features.Artist.apply(lambda x: x.split(" | ")[0])
    return df_features


@st.cache
def get_album_covers():
    """Load album covers from disk."""
    album_covers = read_pickle(ALBUM_COVER_FILE)
    return {i[0]: i[-1] for i in album_covers}


d_playlists = get_playlists()
df_features = get_features()
d_playlist_album_covers = get_album_covers()
df_track_info = df_features[[i for i in COL_ORDER if i != "ID"]].copy()
df_track_info["SongNameArtist"] = df_track_info.SongName + " - " + df_track_info.Artist
all_songs_with_features = set(df_features.index)

if show_playlists:
    playlist_options = show_playlists
else:
    playlist_options = sorted(set(d_playlists.keys()) - set(exclude_playlists))

# --------- First row of page
st.title("Recommended Frequencies: A Recommendation System For Playlists")
st.markdown(
    "**This app allows users to identify songs in their library that may fit well into a selected playlist**"
)
controls = st.expander("Page Settings")
include_audio_preview = controls.checkbox("Include Audio Preview", value=False)
st.markdown("---")
st.markdown("### 1. Choose a playlist")
select_playlist = st.selectbox("", playlist_options)

# --------- Get playlist related info
playlist_tracks = sorted(
    list(
        set(d_playlists[select_playlist]["tracks"]).intersection(
            set(all_songs_with_features)
        )
    )
)
df_playlist_features = df_features.loc[playlist_tracks].copy()
df_playlist_info = df_track_info.loc[playlist_tracks]
mood_board = plot_mood_board(
    df_playlist_features[EUCLIDEAN_FEAT_COLS], title="", inline=False, metrics_version=1
)

# --------- Second row of page: playlist info
col1, col2 = st.columns(2)
col1.markdown(f"#### Some songs in playlist «{select_playlist}»")
if include_audio_preview:
    col1.write(
        (
            df_playlist_info.reset_index()[COL_ORDER]
            .head(10)
            .style.format({"PreviewURL": make_clickable_html})
            .to_html()
        ),
        unsafe_allow_html=True,
    )
else:
    col1.dataframe(
        (df_playlist_info.reset_index()[COL_ORDER].head(10).drop("PreviewURL", axis=1)),
        # height=600
    )

# To copy list of songs into the presentation
# df_playlist_info.reset_index()[COL_ORDER].head(10).to_csv(os.path.join(DATA_DIR, "TMP_PLAYLIST_SONGS.csv"))

col2.markdown(f"#### Song attributes for playlist «{select_playlist}»")
col2.plotly_chart(mood_board)

try:
    # Collage of album covers
    # collage = plot_album_covers(list(df_playlist_features.AlbumCover.values[:10]))
    collage = plot_album_covers(d_playlist_album_covers[select_playlist])
    # Use st.image with image in buffer to allow transparent background, instead of st.pyplot(collage)
    buf = BytesIO()
    collage.savefig(
        buf, format="png", bbox_inches="tight", transparent=True, dpi=3 * collage.dpi
    )
    st.image(buf, use_column_width=True)
except KeyError:
    pass

# --------- Third row of page: similarity metric controls
st.markdown("---")
st.markdown("### 2. Find similar songs")
controls = st.expander("Similarity Settings")
select_suggestion_engine = controls.selectbox(
    "Similarity Features", SUGGESTION_ENGINES.keys()
)
suggestion_engine = SUGGESTION_ENGINES[select_suggestion_engine]

if select_suggestion_engine == "Catboost":
    genre_weight = 0
    genre_similarity = None
    if not os.path.exists(CATBOOST_MODEL_FILE):
        print("Creating song triples")
        df_example_triplets = create_song_triplets()
        print("Obtaining features for song triples")
        df_features_for_model = create_song_pair_features(df_example_triplets)
        df_train, df_test = create_train_test_split(df_features_for_model)
        # TODO: Allow to retrain from within the streamlit app
        model_catboost = train_catboost(df_train, df_test)
        model_catboost.save_model(CATBOOST_MODEL_FILE)
    else:
        model_catboost = catboost.CatBoostClassifier()
        model_catboost.load_model(fname=CATBOOST_MODEL_FILE)
else:
    raise Exception(
        f"Suggestion engine '{select_suggestion_engine}' not yet implemented"
    )

top_n = controls.slider(
    "Nr of Suggestions",
    min_value=SIMILAR_SONGS_STEPS,
    max_value=SIMILAR_SONGS_STEPS * 10,
    value=SIMILAR_SONGS_STEPS * 3,
    step=SIMILAR_SONGS_STEPS,
)


# --------- Find most similar songs
@st.cache
def get_results_wrapper(
    _suggestion_engine,
    _df_playlist_features,
    _df_songs_available_for_suggestion_features,
    _genre_similarity,
    _model_catboost,
    _playlist_name,
):
    """Wrap suggestion_engine into a function in order to use st.cache()."""
    if _genre_similarity == "everynoise":
        _songs_available_for_suggestion_features = (
            _df_songs_available_for_suggestion_features.query(
                "missing_everynoise_genre == False"
            ).copy()
        )
    return _suggestion_engine(
        df_playlist_features=_df_playlist_features,
        df_songs_available_for_suggestion_features=_df_songs_available_for_suggestion_features,
        genre_similarity=_genre_similarity,
        model_catboost=_model_catboost,
        playlist_name=_playlist_name,
    )


@st.cache
def get_top_results_wrapper(
    _select_suggestion_engine, _df_song_similarity, _genre_weight, _top_n
):
    """Wrap get_top_results into a function in order to use st.cache()."""
    if _select_suggestion_engine == "Catboost":
        return _df_song_similarity.head(top_n)
    else:
        return get_top_results(
            df_results=_df_song_similarity, genre_weight=genre_weight, n=_top_n
        )


songs_available_for_suggestion = list(all_songs_with_features - set(playlist_tracks))
print(f"Number of songs considered by the app: ", len(all_songs_with_features))
print("Number of songs available for suggestion given the chosen playlist: ", len(songs_available_for_suggestion))
df_songs_available_for_suggestion_features = df_features.loc[
    songs_available_for_suggestion
].copy()
df_song_similarity = get_results_wrapper(
    _suggestion_engine=suggestion_engine,
    _df_playlist_features=df_playlist_features,
    _df_songs_available_for_suggestion_features=df_songs_available_for_suggestion_features,
    _genre_similarity=genre_similarity,
    _model_catboost=model_catboost,
    _playlist_name=select_playlist,
)
df_suggested_songs = get_top_results_wrapper(
    _select_suggestion_engine=select_suggestion_engine,
    _df_song_similarity=df_song_similarity,
    _genre_weight=genre_weight,
    _top_n=top_n,
)
df_suggested_songs_info = df_track_info.join(
    df_suggested_songs, how="inner"
).sort_values(by="Similarity", ascending=False)
df_suggested_songs_info["SongNameArtist"] = (
    df_suggested_songs_info.SongName + " - " + df_suggested_songs_info.Artist
)


df_suggested_songs_info["Similarity"] = df_suggested_songs_info["Similarity"].map(
    lambda x: "{0:.1f}%".format(x * 100)
)

# --------- Fourth row of page: display results
col1, col2 = st.columns(2)
col1.markdown(f"#### Most similar songs to playlist «{select_playlist}»")

if top_n > 10:
    result_page = col1.selectbox("Result Page # ", range(1, top_n // 10))
else:
    result_page = 1
start = (result_page - 1) * SIMILAR_SONGS_STEPS
end = result_page * SIMILAR_SONGS_STEPS

if include_audio_preview:
    col1.write(
        (
            df_suggested_songs_info.reset_index()
            .iloc[start:end][COL_ORDER + ["Similarity"]]
            .style.format({"PreviewURL": make_clickable_html})
            .to_html()
        ),
        height=600,
        unsafe_allow_html=True,
    )
else:
    col1.dataframe(
        (
            df_suggested_songs_info.reset_index()
            .iloc[start:end][COL_ORDER + ["Similarity"]]
            .drop("PreviewURL", axis=1)
        ),
        # height=600
    )
col2.markdown("#### Visualise similarity of proposed song")
# To copy list of songs into the presentation
# df_suggested_songs_info.reset_index()[COL_ORDER].to_csv(os.path.join(DATA_DIR, "TMP_SUGGESTIONS.csv"))
select_song_type = col2.radio("Search for song by", options=["ID", "Name"])
if select_song_type == "ID":
    select_song_suggested = col2.text_input(
        "ID of song to visualise", df_suggested_songs.index[0]
    )
    selected_song_name, selected_song_artist = df_track_info.loc[
        select_song_suggested, ["SongName", "Artist"]
    ].values
else:
    select_song_suggested_name = col2.selectbox(
        "Name of song to visualise", df_track_info.SongNameArtist.unique()
    )
    tmp = df_track_info.loc[
        lambda x: x.SongNameArtist == select_song_suggested_name, ["SongName", "Artist"]
    ].iloc[:1]
    selected_song_name, selected_song_artist = tmp.values[0]
    select_song_suggested = tmp.index[0]

song_radial_plot_trace = plot_radial_plot(
    df_features.loc[select_song_suggested].copy(),
    title=f"{selected_song_name} by {selected_song_artist}",
    only_return_trace=True,
)
mood_board.add_trace(song_radial_plot_trace)
mood_board.update_layout(
    title=f"Song: {selected_song_name} by {selected_song_artist}"
    + "<br>"
    + f"Playlist: {select_playlist}"
)
col2.plotly_chart(mood_board)
if not include_audio_preview:
    col2.markdown("##### Listen to proposed song")
    preview_audio = df_track_info.loc[select_song_suggested, "PreviewURL"]
    if not pd.isnull(preview_audio):
        col2.audio(preview_audio)
    else:
        col2.write("Audio preview not available")
