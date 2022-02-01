# Recommended Frequencies

# Introduction
«Recommended Frequencies» is a recommendation engine for playlists and currently works for Spotify.
Given a selected playlist in a user's library, 
the app suggests songs from the user's liked songs that may make a good addition to it.
The current goal of this app is to provide recommendations solely using information from a user's library.
Thus, methods like collaborative filtering are outside of the current scope.

Specifically, under the current scope, song suggestions are based on similarities between audio features, song attributes and — optionally — genre.
The audio features used by the app are a subset of those provided by [Spotify's API](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features) 
in addition to the year of the song's album release. All features used are listed [here](#audio_features).

Similarities metrics used by the app are the following:
  - For audio features and year of album release:
    - Euclidean distance 
  - For genres, users can choose between:
    - Jaccard index
    - Cosine distance of `word2vec` word embeddings
    - Co-occurrence
    - Euclidean distance of genre embeddings provided by [Every Noise at Once](https://everynoise.com/)

The app is developed using `streamlit`.

# Dashboard

The first step in the dashboard is to select a desired playlist from one's library.
A selection of songs, the playlist profile as measured by the audio features of its songs
and some album covers are visualised.

<img src="resources/images/page1.png" width="800" />

The second step is to select similarity metrics.
The first option is to simply base similarity off of audio features.
The second is to include genre information.
Details on the available similarity metrics can be found in [here](audio_features).

<img src="resources/images/page2.png" width="800" />

# Installation guide

## 1. Set-up Spotify developer account & register app

These steps have been validated with the website's version as of 2021-01-17.

1. Go to https://developer.spotify.com/dashboard/ and create an account
2. Click on "CREATE AN APP"
3. Provide the app name & description of your choice, tick the terms of service and click "CREATE"
4. Click on "EDIT SETTINGS"
5. Under "Redirect URIs" put `http://localhost:9000/callback/` and `http://localhost:8090`
6. On the left side of the dashboard, underneath the description of your app, you will find your apps' "Client ID".
   Take note of it as you will need it in step 2.2.
7. Below your "Client ID" you will find an option to "SHOW CLIENT SECRET", click on it and take note of the value as you
   you will need it in step 2.2.

## 2. Set up the environment on your machine

### 2.1 Get the code & set environmental variables

The below instructions are for Linux or MacOS.

```commandline
git clone git@github.com:calyptis/RecommendedFrequencies.git
cd RecommendedFrequencies
source prepare_env.sh
```

### 2.2 Specify your credentials

In the folder `credentials` create a file named `credentials.json` 
where you specify the configurations you obtained in step 1.6 & 1.7.

The file has the following structure:

```python
{
	"client_id": "{your_client_id}",
	"client_secret": "{you_client_secret}",
	"redirect_uri": "http://localhost:9000/callback/",
	"redirect_flask_uri": "http://localhost:8090"
}
```

replace your client ID with value from step 1.6 and your client secret from step 1.7.

### 2.3 Obtain data

In order to host the dashboard locally, one must collect all relevant data.
This can be done by running

```commandline
python src/spotify/main.py
```

Note that this will open a Spotify login page in the browser were one must
authenticate themselves.

Once logged in, a `.spotify_caches` will be created in the root folder.

Subsequent API calls will not trigger a log-in page anymore unless this file is deleted.

### 2.4 Run the dashboard

Once all the data has been obtained, one can spin up the dashboard by running

```commandline
streamlit run src/streamlit/main.py
```

# User guide

## 1. Audio features<a name="audio_features"></a>

Documentation of all the available audio features can be found on
[Spotify's official documentation page](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features).

In this project, a subset of those are used and are listed below with their official description for convenience:

1. **Acousticness**: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
2. **Danceability**: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
3. **Energy**: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
4. **Instrumentalness**: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
5. **Liveness**: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
6. **Loudness**: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.
7. **Speechiness**: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words.
8. **Tempo**: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
9. **Valence**: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).

In addition, metadata on songs are considered. Specifically, the 10th audio attribute used in this project is **Year of Album Release**.

All these features are cast to a range of [0, 1] if they are not already.

## 2. Similarity metrics

To measure the similarity between two songs, several similarity metrics are employed in this project.
The specific metric depends on the type of feature.
Two type of features are used in to measure song similarities: 
audio features as outlined [here](#audio_features) and genres.

### 1. For audio features

#### 1) Weighted [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance)

<img src="resources/images/euclidean_distance.png" width="350" />

where `n` is the dimension of the audio feature vectors with `n = 10`.
`s` is the feature vector for a given song.
`p` is a vector containing the averages across all songs in the playlist with one element per feature.
σ is a vector containing the standard deviations across all songs in the playlist with one element per feature.

### 2. For genres

#### 1) [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index)

<img src="resources/images/jaccard_index.png" width="350" />

where `S1` is the set of genres in a playlist and `S2` the set of genres for a given song.

#### 2) [Word2Vec](https://en.wikipedia.org/wiki/Word2vec)

Get the word embeddings of genres using a pre-trained word2vec model from [FastText](https://fasttext.cc).
The `word2vec` technique falls under the domain of Natural Language Processing.

In the case of this project, a song's list of genres is treated as a sentence with an individual genre being a word. 
Thus, the embedding of a song is obtained for this sentence composed of genre terms.
Similarly, the embedding of a playlist is obtained by averaging across all of its songs' embeddings.

To get the similarity between a playlist and a song, the cosine distance between the two embeddings is calculated.

A crucial limitation of this approach inevitably arises:
Pre-trained models are not typically trained on documents that focus on music or that describe relationships of genres.
Thus, the embeddings are likely not to not reflect genre similarities very well.

An illustration will help outline this point:
In one of the typical text book examples of `word2vec`, the word ***kitten*** is more similar to ***cat*** than to ***dog***.
Testing this using genre terms, ***techno*** is not more similar to ***pop*** than to ***jazz*** as one would suppose.
Below is a code example to reproduce this.

```python
import fasttext
from sklearn.metrics.pairwise import cosine_distances as cosine

ft = fasttext.load_model('cc.en.300.bin')

# Text book example: note that lower values imply higher degree of similarity
cosine(ft.get_word_vector("kitten").reshape(1, -1), ft.get_word_vector("cat").reshape(1, -1))
>>> 0.19
cosine(ft.get_word_vector("kitten").reshape(1, -1), ft.get_word_vector("dog").reshape(1, -1))
>>> 0.39

# Music genre example
cosine(ft.get_word_vector("techno").reshape(1, -1), ft.get_word_vector("pop").reshape(1, -1))
>>> 0.63
cosine(ft.get_word_vector("techno").reshape(1, -1), ft.get_word_vector("jazz").reshape(1, -1))
>>> 0.51
```

A more thorough analysis on how `word2vec` genre similarities compare to `everynoise` is available
in [this notebook](notebooks/Word2Vec_vs_EveryNoise.ipynb).

#### 3) Co-occurrences

Based on a user's list of playlists, calculate how often a pair of two genres occurs within a given playlist.
The higher this co-occurrence, the more similar two genres are.
Formally, this can be expressed in the following formula:

<img src="resources/images/co_occurrence_formula.png" width="350" />

with `g1` and `g2` being two genres and `P` the set of playlists. 

#### 4) Everynoise embeddings

The position of a genre in the musical genre space measured by the Every Noise at Once project is used as its 
embedding. Similar genres lie close to each other in this space, of which a visual representation is available on
the [project's website](https://everynoise.com).
Since this musical genre space is two-dimensional, each genre is represented by its `(x, y)` location in the scatter plot.

Genre similarity is then simply the euclidean distance between the two locations.

# Known Bugs:
- All audio previews are sometimes played at the same time when updating the dashboard.

# TODO:
- Get the oldest year of release for a given song. 
  For example, if a song appeared in a recent remastered album, 
  track down the first album the song appears in and use that as a release date.

# Related projects
- https://dubolt.com
- https://newsroom.spotify.com/2021-09-09/get-perfect-song-recommendations-in-the-playlists-you-create-with-enhance/
- https://spotibot.com
- https://www.chosic.com/

# Resources:
- Spotify's song features
  - Sample of data: https://www.kaggle.com/nadintamer/top-tracks-of-2017
  - Description of features: https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features
- Analysis of music genres
  - Genre similarity: https://everynoise.com/
- Music recommendation
  - ML model: https://dl.acm.org/doi/10.1145/3383313.3412248
- Articles on Spotify's music recommendation
  - https://www.popsci.com/technology/spotify-audio-recommendation-research/
