import pandas as pd
import gap_statistic
import spotipy
import sys
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.oauth2 as oauth2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score, v_measure_score
from sklearn.preprocessing import Normalizer, MinMaxScaler
import numpy as np


def feat_eng(current_data_frame):
    audio_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                  'key', 'loudness', 'liveness', 'speechiness', 'tempo', 'valence']
    data = current_data_frame[audio_cols]
    data = data.fillna(0)
    # Scale Tempo (optional)
    tempo = data['tempo'].values.reshape(-1, 1)
    data['tempo'] = MinMaxScaler().fit_transform(tempo)

    # Scale loudness (optional)
    loudness = data['loudness'].values.reshape(-1, 1)
    data['loudness'] = MinMaxScaler().fit_transform(loudness)

    # Normalize audio columns (optional)
    norm = Normalizer()
    data[audio_cols] = norm.fit_transform(data[audio_cols])
    X_norm = data[audio_cols]
    pd.DataFrame(X_norm, columns=audio_cols).hist(
        figsize=(20, 20), density=True)
    return data


if len(sys.argv) < 2:
    print('Need an input file')
    exit(0)

filename = sys.argv[1]
track_list = open(filename)
CLIENT_ID = "90f4267ae9434d7da4154b97afa2594d"
CLIENT_SECRET = "7c9ebb9c291043e3b525fdfc16af6ffd"
credentials = oauth2.SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET)

token = credentials.get_access_token()
spotify = spotipy.Spotify(auth=token)
current_data_frame = None
count = 0
for song in track_list:
    try:
        track_info = song.strip()
        result = spotify.audio_features(track_info)[0]
        if current_data_frame is None:
            current_data_frame = pd.DataFrame(result, index=[0])
        else:
            current_data_frame = current_data_frame.append(
                pd.DataFrame(result, index=[0]), ignore_index=True)
        count = count + 1
    except Exception as e:
        credentials = oauth2.SpotifyClientCredentials(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET)
        token = credentials.get_access_token()
        spotify = spotipy.Spotify(auth=token)

data = feat_eng(current_data_frame)
optimalK = gap_statistic.OptimalK(parallel_backend='rust')
max_value = min(count, 15)
n_clusters = optimalK(np.array(data), cluster_array=np.arange(1, max_value))
print('Number of Clusters ' + str(n_clusters))

cluster = KMeans(n_clusters=n_clusters)
cluster.fit(data)

user_data = pd.read_csv('Subset(1).csv')
x_test = feat_eng(user_data)
clustered = cluster.transform(x_test)
min_dist = 1000
index = 0
for i in range(n_clusters):
    ind = np.argsort(clustered[:, i])[0]
    val = np.sort(clustered[:, i])[0]
    if min_dist > val:
        min_dist = val
        index = ind

print('Track :' + str(user_data.iloc[index]['track']))
print('URI: ' + str(user_data.iloc[index]['uri']))
print('Distance: ' + str(min_dist))
