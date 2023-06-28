import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


#%% Data Loaders
    
class CustomDataset(Dataset):
    # Prepares a Dataset to be converted into a dataloader
    # Example of use:
        # dataset_train = CustomDataset(X_train, Y_train)
        # trainloader = DataLoader(dataset_train, batch_size=64, shuffle=False)
    
    def __init__(self, X, Y):
        
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y)
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        
    def __len__(self):
        
        return len(self.X)
    
    def __getitem__(self, idx):
        
        x = self.X[idx]
        y = self.Y[idx]
        
        return x, y


#%% Neural Network definition

class NN_dynamical(nn.Module):
    # Creates a dense NN for regression
    # dims is like [50 1024 128 2]: input size 50, output size 2, two hidden layers of 1024 and 128
    # dropouts is like [False, True, True]
    
    def __init__(self, dims, dropouts, epochs=100,lr=0.001, p = 0.2):
        
        # dims (length = n_layers+1) is a list like [73, 64, 28, 50] for a 3-layers NN
        # dropouts (length = n_layers) i a list of True and False indicating after which layers we do dropout. Last value should be false
        
        super().__init__()
        
        self.dims = dims
        self.dropouts = dropouts
        self.layers = nn.ModuleList()
        # Creating layers
        for i in range(len(dims)-1): 
            self.layers.append(nn.Linear(self.dims[i],self.dims[i+1]))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)
        
        self.epochs = epochs
        self.lr = lr
        self.optim = optim.Adam(self.parameters(), self.lr)
        self.criterion = nn.MSELoss()
        
        self.loss_during_training = []
        self.valid_loss_during_training = []
        
        self.best_epochs = 0
        
    def forward(self, x):
        
        x = x.float()
        
        i = 0
        for layer in self.layers:
            # For each layer we apply the layer transformation and relu (if not last layer) and dropout if demanded
            x = layer(x)
            if i != len(self.dims) - 2:
                x = self.relu(x)
                if self.dropouts[i]:
                    x = self.dropout(x)
            i+=1       
        return x
          
    def find_best_epochs(self, trainloader, validloader, max_patience = 10, verbose = False):
        # Performs training to discover which is the optimum number of epochs before falling in overfitting
        
        self.best_epochs = 0
        self.loss_during_training = []
        self.valid_loss_during_training = []
        
        best_val_loss = float('inf')
        patience = 0
        
        for e in range(int(self.epochs)):
            running_loss = 0
            val_running_loss = 0
            
            # Predict trainset and append loss
            for x, y in trainloader:          
                self.optim.zero_grad()
                out = self.forward(x)
                loss = self.criterion(out, y)
                running_loss += loss.item() * len(x)
                loss.backward()
                self.optim.step()
            self.loss_during_training.append(running_loss/len(trainloader.dataset))
            
            # Predict validation set and append valid_loss
            with torch.no_grad():
                self.eval()
                for x, y in validloader:          
                    out = self.forward(x)
                    loss = self.criterion(out, y)
                    val_running_loss += loss.item() * len(x)             
                val_loss = val_running_loss/len(validloader.dataset)                 
                self.valid_loss_during_training.append(val_loss)    
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_epochs = e + 1
                    patience = 0
                else:
                    patience += 1                    
            self.train()

            if((e % 5 == 0) & (verbose)):
                print("Training loss after %d epochs: %f" 
                      %(e,self.loss_during_training[-1]))
                
            if patience >= max_patience:
                if(verbose):
                    print(f"Validation loss did not improve for {max_patience} epochs. Stopping training.")
                    print("The best epoch was: ", self.best_epochs)
                    plt.plot(self.loss_during_training, label = "Train loss")
                    plt.plot(self.valid_loss_during_training, label = "Validation loss")
                    plt.legend()
                    plt.show()
                return 
    
    
    def trainloop(self, trainloader, validloader, verbose = False):
        
        self.loss_during_training = []
        self.valid_loss_during_training = []
        
        for layer in self.layers:
            layer.reset_parameters()
            
        for e in range(int(self.best_epochs)):

            running_loss = 0
            val_running_loss = 0
            
            # Predict trainset and save loss
            for x, y in trainloader:          
                self.optim.zero_grad()
                out = self.forward(x)
                loss = self.criterion(out, y)
                running_loss += loss.item() * len(x)
                loss.backward()
                self.optim.step()
            self.loss_during_training.append(running_loss/len(trainloader.dataset))
            
            # Predict validation set and save valid_loss
            with torch.no_grad():               
                self.eval()            
                for x, y in validloader:
                    out = self.forward(x)
                    loss = self.criterion(out, y)
                    val_running_loss += loss.item() * len(x)             
                val_loss = val_running_loss/len(validloader.dataset)                 
                self.valid_loss_during_training.append(val_loss)    
            self.train()

            if((e % 5 == 0) & (verbose)):
                print("Training loss after %d epochs: %f" %(e,self.loss_during_training[-1]))
                
        if verbose:
                print("Trained for: ", self.best_epochs, "epochs with a training loss of", self.loss_during_training[-1])
                plt.plot(self.loss_during_training, label = "Train loss")
                plt.plot(self.valid_loss_during_training, label = "Validation loss")
                plt.legend()
                plt.show()
    
    def evaluate(self, testloader):
        
        eval_loss = 0

        with torch.no_grad():
            self.eval()
            for x, y in testloader:
                out = self.forward(x) 
                loss = self.criterion(out,y)
                eval_loss += loss.item() * len(x)
        self.train()
        return eval_loss / len(testloader.dataset)
    
