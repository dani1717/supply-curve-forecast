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
Y_train = pd.read_csv(folder + 'Y_train' + sufix + '.csv')
Y_train.drop('datetime', axis=1, inplace=True)


#%% SCALE

Y_train = Y_train.T
scaler = StandardScaler()
Y_train = scaler.fit_transform(Y_train)

#%% ELBOW GRAPH

from sklearn.cluster import KMeans

inertias = []
for k in range(2, 25):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(Y_train)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(range(2, 25), inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.title('Elbow curve')
plt.xticks(range(2, 25))
plt.show()
