#%% IMPORTS --------------------------------------------------------------------
import pandas as pd
import numpy as np
import csv
import random
import datetime
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import myFunctions_Dani as dani
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import time
import optuna
import multiprocessing
import pickle
import matplotlib.dates as mdates

#%% LOAD DATA --------------------------------------------------------------------

folder = "Data/Preprocessed_inputs/"

sufix=''
X_train = pd.read_csv(folder + 'X_train' + sufix + '.csv')
Y_train = pd.read_csv(folder + 'Y_train' + sufix + '.csv')
X_test = pd.read_csv(folder + 'X_test' + sufix + '.csv')
Y_test = pd.read_csv(folder + 'Y_test' + sufix + '.csv')

X_train['datetime'] = pd.to_datetime(X_train['datetime'])
X_train.set_index('datetime', inplace=True)
X_test['datetime'] = pd.to_datetime(X_test['datetime'])
X_test.set_index('datetime', inplace=True)
Y_train['datetime'] = pd.to_datetime(Y_train['datetime'])
Y_train.set_index('datetime', inplace=True)
Y_test['datetime'] = pd.to_datetime(Y_test['datetime'])
Y_test.set_index('datetime', inplace=True)

# Original data
file_name = "Data/Other/04_Preprocessing_data_original.pickle"
with open(file_name, "rb") as f:
    data_original = pickle.load(f)

    
del folder, file_name, f

#%% STUDY VARIABILITY

data = Y_train.copy()
data['hour'] = Y_train.index.hour

# Average std within hour groups
std_inHour = data.groupby('hour').std().mean(axis=1).mean()
# Average std
std_total = data.drop('hour', axis=1).std().mean()


#%% SCALE

hour_means = data.groupby('hour').mean()
scaler = StandardScaler()
hour_means2 = scaler.fit_transform(hour_means)
hour_means = pd.DataFrame(hour_means2, index=hour_means.index, columns=hour_means.columns)


#%% ELBOW GRAPH

from sklearn.cluster import KMeans

inertias = []
for k in range(2, 25):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(hour_means)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(range(2, 25), inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.title('Elbow curve')
plt.xticks(range(2, 25))
plt.show()


#%% CLUSTERING

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(hour_means)

hour_groups = pd.DataFrame({'hour': hour_means.index, 'group': kmeans.labels_})
hour_groups['group'].value_counts()

# Save
hour_groups.to_csv('Data/Other/12c_hours_in_groups.csv', index=False)
