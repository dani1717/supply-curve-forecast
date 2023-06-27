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

n_days_train = 365
n_days_test = 365
X_test = X_train.iloc[n_days_train*24:n_days_train*24+n_days_test*24]
Y_test = Y_train.iloc[n_days_train*24:n_days_train*24+n_days_test*24]
X_train = X_train.iloc[0:n_days_train*24]
Y_train = Y_train.iloc[0:n_days_train*24]

Y_test_unscaled = Y_test


#%% NORMALIZE DATA

scaler = StandardScaler()
Y_train = scaler.fit_transform(Y_train)
Y_test = scaler.transform(Y_test)

scaler2 = StandardScaler()
cols_to_scale = ['mDiario_Q', 'mDiario_price', 'r1', 'r2', 'w1', 'w2', 'w3', 'w4', 'w5',
       'w6', 'w7', 'w8', 'w9', 'w10', 'gas_price', 'day_of_month',
       'lag24_pc1', 'lag24_pc2', 'lag24_pc3', 'lag24_pc4', 'lag24_pc5',
       '2weeksEvol_1', '2weeksEvol_2', '2weeksEvol_3', '2weeksEvol_4']
X_train[cols_to_scale] = scaler2.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler2.transform(X_test[cols_to_scale])

#%% RNN DEFINITION

class RNN(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers, num_data_train,sigma):
        
        # input size -> Dimension of the input signal
        # outpusize -> Dimension of the output signal
        # hidden_dim -> Dimension of the rnn state
        # n_layers -> If >1, we are using a stacked RNN
        
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.output_size = output_size
        self.sigma = torch.Tensor(np.array(sigma))
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_dim, num_layers=n_layers, 
                          nonlinearity='relu',batch_first=True)   
                          # batch_first=True means that the first dimension of the input will be the batch_size
        
        # One linear layer to estimate mean
        self.fc1 = nn.Linear(hidden_dim, output_size)
        # One linear layer to estimate log-variance
        self.fc2 = nn.Linear(hidden_dim, output_size)
        
        self.hidden_dim = hidden_dim
        self.num_layers = n_layers
        self.lr = 0.001 #Learning Rate
        self.num_train = num_data_train #Number of training signals
        self.optim = optim.Adam(self.parameters(), self.lr)
        self.criterion = nn.MSELoss()   
        self.loss_during_training = []  # A list to store the loss evolution along training
        
        # Move to GPU if existing
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.to(self.device)
            self.sigma = self.sigma.to(self.device)
            print('GPU available. Training in GPU.')
        else:
            self.device = torch.device('cpu')
            print('GPU not available. Training in CPU.')

    def forward(self, x, h0=None):
        
        '''
        About the shape of the different tensors ...:
        - Input signal x has shape (batch_size, seq_length, input_size)
        - The initialization of the RNN hidden state h0 has shape (n_layers, batch_size, hidden_dim).
          If None value is used, internally it is initialized to zeros.
        - The RNN output (batch_size, seq_length, hidden_size). This output is the RNN state along time  
        '''
        x = x.to(self.device) 
        batch_size = x.size(0) # Number of signals N
        seq_length = x.size(1) # T
        
        r_out, hidden = self.rnn(x, h0)  # r_out is the sequence of states, hidden is just the last state (we will use it for forecasting).

        # shape r_out to be (seq_length, hidden_dim)       
        r_out = r_out.reshape(-1, self.hidden_dim)  # En cada fila está un estado (h_i), que tiene hidden_dim componentes
        
        # We compute the mean
        mean = self.fc1(r_out)
        # We compute the variance
        variance = torch.exp(self.fc2(r_out))+torch.ones(mean.shape).to(self.device)*self.sigma
        # We generate noise of the adecuate variance
        noise = torch.randn_like(mean)*torch.sqrt(variance)
        
        sample = mean+noise
        
        # reshape back to temporal structure
        sample = sample.reshape([-1,seq_length,int(self.output_size)])
        mean = mean.reshape([-1,seq_length,int(self.output_size)])
        variance = variance.reshape([-1,seq_length,int(self.output_size)])
        
        return mean, variance, hidden, sample
    
    def trainloop(self,x,y,num_iter=200, lr=0.001):
        
        self.lr = lr
        seq_length = x.size(1)
        x = torch.Tensor(x).view([self.num_train,seq_length,self.input_size]).to(self.device) 
        y = torch.Tensor(y).view([self.num_train,seq_length,self.output_size]).to(self.device)
        
        for e in range(int(num_iter)):  # SGD Loop
            
            self.optim.zero_grad()   
            with torch.set_grad_enabled(True):   # We only compute gradients in the training moment
                mean,var,hid,sample = self.forward(x) 
                loss = self.criterion(sample,y)           
            loss.backward()
            
            # This code helps to avoid vanishing/exploiting gradients in RNNs
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(self.parameters(), 2.0)                
            self.optim.step()               
            self.loss_during_training.append(loss.detach().item())
            if(e % 10 == 0): # Every 10 iterations
                print("Iteration %d. Training loss: %f" %(e,self.loss_during_training[-1]))   
                
                
#%% FIT AND PREDICTION

Y = Y_train.T
T = X_train.shape[0]
batch_size = int(T/24)
X = X_train.T.values
X_torch = torch.tensor(X, dtype=torch.float32).view([24,batch_size,X_train.shape[1]])
Y_torch = torch.tensor(Y, dtype=torch.float32).view([24,batch_size,Y_train.shape[1]])

my_rnn = RNN(input_size=X_train.shape[1], output_size=Y_train.shape[1], hidden_dim=1024, n_layers=1, num_data_train=X_torch.shape[0], 
             sigma=0.05)
my_rnn.trainloop(X_torch,Y_torch, num_iter=201, lr=1e-6)

plt.plot(my_rnn.loss_during_training,label='Training Loss', )
plt.ylim(0,2.5)
plt.legend()

T = Y_test.shape[0]
X = X_test.T.values
batch_size = int(T/24)
X_torch = torch.tensor(X, dtype=torch.float32).view([24,batch_size,X_test.shape[1]])
Y_pred, var, h, sample = my_rnn.forward(X_torch)
Y_pred = Y_pred.view([1,batch_size*24,Y_test.shape[1]]).cpu()
Y_pred = Y_pred.detach().numpy().squeeze()
Y_pred = pd.DataFrame(Y_pred)
Y_pred.index = Y_test_unscaled.index
Y_pred.columns = Y_test_unscaled.columns


#%% COMPARISON & METRICS

Y_pred_unscaled = scaler.inverse_transform(Y_pred)
Y_pred_unscaled = pd.DataFrame(Y_pred_unscaled)
Y_pred_unscaled.index = Y_test_unscaled.index
Y_pred_unscaled.columns = Y_test_unscaled.columns

def plotPreds(Y_test, Y_pred, serie=None, from_date=None, to_date=None, n_days=None, plotLag24=False):
    # Plots the comparison of prediction and real series
    
    if n_days is None:
        n_days = 7
    if from_date is None:
        from_date = Y_test.index[random.randint(0, len(Y_test)-n_days*24)]  # We sample a random row from Y_test
    else:
        from_date = datetime.datetime.strptime(from_date, "%Y-%m-%d")
    if to_date is None:
            to_date = from_date + datetime.timedelta(days=n_days)
    if serie is None:
        serie = random.choice(list(Y_test.columns))
    
    Y_pred = pd.DataFrame(Y_pred)
    real_values = Y_test.loc[from_date:to_date, serie]
    rows = list(range(Y_test.index.get_loc(from_date), Y_test.index.get_loc(from_date) + len(real_values)))
    col = Y_test.columns.get_loc(serie)
    pred_values = Y_pred.iloc[rows, col]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(real_values, color='blue', label='Real values', linewidth=2)
    ax.plot(real_values.index, pred_values, color='red', label='Predicted values', linestyle='--')
    if plotLag24:
        lag_values = Y_test.loc[from_date-datetime.timedelta(hours=24):to_date-datetime.timedelta(hours=24), serie]
        lag_values.index = real_values.index
        ax.plot(lag_values.index, lag_values, color='green', label='lag24 estimator', linestyle='--')
    ax.set_ylabel(serie)
    ax.set_title('Prediction for ' + serie)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.legend()
    plt.show()

plotPreds(Y_test_unscaled, Y_pred_unscaled, n_days=2, plotLag24=True)

import TFM_functions as tfm
stats, mae, rmse = tfm.errors_analysis(Y_test_unscaled, Y_pred_unscaled, totals=True)

# Creo que RNN quizá no sea bueno para esto, ya que hace predicciones demasiado periódicas

