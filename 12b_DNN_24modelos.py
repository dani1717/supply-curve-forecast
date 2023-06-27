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
import random
random.seed(17)
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import TFM_functions as tfm
import myClasses_Dani as myClasses

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

Y_test_unscaled = Y_test

# Create validation sets
validation_rows = 365*24
X_val = X_train.tail(validation_rows)
Y_val = Y_train.tail(validation_rows)
X_train = X_train.iloc[:-validation_rows]
Y_train = Y_train.iloc[:-validation_rows]

#%% NORMALIZE DATA

# Scale Y
scaler = StandardScaler()
Y_train = scaler.fit_transform(Y_train)
Y_val = scaler.transform(Y_val)
Y_test = scaler.transform(Y_test)

#Convert to dataframe again
Y_train = pd.DataFrame(Y_train)
Y_train.index = X_train.index
Y_train.columns = Y_test_unscaled.columns
Y_val = pd.DataFrame(Y_val)
Y_val.index = X_val.index
Y_val.columns = Y_test_unscaled.columns
Y_test = pd.DataFrame(Y_test)
Y_test.index = X_test.index
Y_test.columns = Y_test_unscaled.columns

# Scale X
scaler2 = StandardScaler()
cols_to_scale = ['mDiario_Q', 'mDiario_price', 'r1', 'r2', 'w1', 'w2', 'w3', 'w4', 'w5',
       'w6', 'w7', 'w8', 'w9', 'w10', 'gas_price', 'day_of_month',
       'lag24_pc1', 'lag24_pc2', 'lag24_pc3', 'lag24_pc4', 'lag24_pc5',
       '2weeksEvol_1', '2weeksEvol_2', '2weeksEvol_3', '2weeksEvol_4']
X_train[cols_to_scale] = scaler2.fit_transform(X_train[cols_to_scale])
X_val[cols_to_scale] = scaler2.transform(X_val[cols_to_scale])
X_test[cols_to_scale] = scaler2.transform(X_test[cols_to_scale])

# Drop  useless columns since we are creating 24 models
cols_to_drop = ['hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7',
       'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13',
       'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19',
       'hour_20', 'hour_21', 'hour_22', 'hour_23']
X_train.drop(cols_to_drop, axis=1, inplace=True)
X_val.drop(cols_to_drop, axis=1, inplace=True)
X_test.drop(cols_to_drop, axis=1, inplace=True)


#%% HPO

# Prepare dataloaders by hour
trainloaders = []
validloaders = []

for hour in range(24):
    # Prepare data from that hour
    dataset_train = myClasses.CustomDataset(X_train.loc[X_train.index.hour == hour], Y_train.loc[Y_train.index.hour == hour])
    trainloader = DataLoader(dataset_train, batch_size=64, shuffle=False)
    dataset_val = myClasses.CustomDataset(X_val.loc[X_val.index.hour == hour], Y_val.loc[Y_val.index.hour == hour])
    validloader = DataLoader(dataset_val, batch_size=64, shuffle=False)
    # Add dataloaders to corresponding list
    trainloaders.append(trainloader)
    validloaders.append(validloader)


# Objective function for optuna
def objective(trial):
    
    n_layers = trial.suggest_int("n_layers",2,5)
    # Dimensions of the layers. For example 2 layers dims should be [73, 128, 64, 50]
    dims = [X_train.shape[1]]
    for i in range(1,n_layers):
        if i == 1:
            dims.append(trial.suggest_int("d"+str(i),128+(i-1)*50,1024))
        else:
            dims.append(trial.suggest_int("d"+str(i),min(128+(i-1)*50,dims[i-1]-1),dims[i-1]))
    dims.append(Y_train.shape[1])
    print('Dimensions of the NN:',dims)
    
    p = trial.suggest_float("p",0.05,0.2)
    dropouts = [False, False, False, True, True, False]
    dropouts = dropouts[-n_layers:]
    max_patience = trial.suggest_int("max_patience",6,25)
    
    valid_loss = 0
    hours = [1,3,5,7,9,11,13,15,17,19,21,23]
    for hour in hours:
        model = myClasses.NN_dynamical(dims, dropouts, epochs=100,lr=0.001, p = p)
        model.find_best_epochs(trainloaders[hour], validloaders[hour], max_patience = max_patience, verbose = False)
        model.trainloop(trainloader, validloader, verbose=False) 
        valid_loss += model.evaluate(validloader)
    return valid_loss / len(hours)

budget = 30
study = optuna.create_study(direction = "minimize")
study.optimize(objective, n_trials = budget)
print("The best set of parameters is: ", study.best_params)
print("The rmse score computed with inner-evaluation is:", study.best_value)
print(f"The optimal value is reached in {study.best_trial.number} iterations")



#%% TRAIN MODEL

# Trial 7 finished with value: 3.5528770312054516 and parameters: {'d1': 1443, 'max_patience': 8}  p=0.14
# Trial 21 finished with value: 3.5670012125413715 and parameters: {'d1': 1321, 'p': 0.13412414559131217, 'max_patience': 11}

dims = [X_train.shape[1], 1443,Y_train.shape[1]]
dropouts = [True, False]
max_patience = 11
p=0.14

models = {}
for hour in range(24):
    
    print(f'================ Fitting model for hour {hour} ================')
    
    # Select data of one hour
    X_train_hour = X_train.loc[X_train.index.hour == hour]
    Y_train_hour = Y_train.loc[Y_train.index.hour == hour]
    X_val_hour = X_val.loc[X_val.index.hour == hour]
    Y_val_hour = Y_val.loc[Y_val.index.hour == hour]
    
    # Create dataloaders
    dataset_train = myClasses.CustomDataset(X_train_hour, Y_train_hour)
    trainloader = DataLoader(dataset_train, batch_size=64, shuffle=False)
    dataset_val = myClasses.CustomDataset(X_val_hour, Y_val_hour)
    validloader = DataLoader(dataset_val, batch_size=64, shuffle=False)
    
    # Fit model
    model = myClasses.NN_dynamical(dims, dropouts, epochs=120,lr=0.001, p = p)
    model.find_best_epochs(trainloader, validloader, max_patience = max_patience, verbose = False)
    model.trainloop(trainloader, validloader, verbose=False) 
    models[f"hour{hour}"] = model

file_name = "Data/Other/12b_DNN_24models.pickle"
with open(file_name, "wb") as f:
    pickle.dump(models, f)
with open(file_name, "rb") as f:
    models = pickle.load(f)

Y_pred = pd.DataFrame()
for hour in range(24):
    print('Predicting hour', hour)
    X_test_hour = X_test.loc[X_test.index.hour == hour]
    X_in = torch.tensor(X_test_hour.values)
    model = models[f"hour{hour}"]
    Y_pred_hour = model.forward(X_in)
    Y_pred_hour = pd.DataFrame(Y_pred_hour.detach().numpy(), index=X_test_hour.index, columns=Y_test.columns)
    Y_pred = pd.concat([Y_pred, Y_pred_hour])
Y_pred = Y_pred.sort_index()


    
#%% COMPARISON AND METRICS

Y_pred_unscaled = scaler.inverse_transform(Y_pred)
Y_pred_unscaled = pd.DataFrame(Y_pred_unscaled)
Y_pred_unscaled.index = Y_test_unscaled.index
Y_pred_unscaled.columns = Y_test_unscaled.columns

tfm.plotPreds(Y_test_unscaled, Y_pred_unscaled, n_days=2, plotLag24=True)

stats, e_mae, e_rmse = tfm.errors_analysis(Y_test_unscaled, Y_pred_unscaled, totals=True)

Y_pred_unscaled2=Y_pred_unscaled.iloc[0:365*24]
Y_test_unscaled2=Y_test_unscaled.iloc[0:365*24]
stats, e_mae, e_rmse = tfm.errors_analysis(Y_test_unscaled2, Y_pred_unscaled2, totals=True)






