import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import networkx as nx
import matplotlib.pyplot as plt

import random

import csv

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method # FGSM
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent # PGD
from cleverhans.torch.attacks.noise import noise # Basic uniform noise perturbation

from split_data import *
# Device configuration
# Always check first if GPU is avaialble
device_used = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# If CUDA is not avaialbe, print message that CPU will be used
if device_used != torch.device('cuda'):
    print(f'CUDA not available, have to use {device_used}')
# Set hyperparameters
seed = 1 # Seed for PRNGs 
N_CLIENTS = 1 # Number of clients
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

iid_type = 'iid'      # 'iid' or 'non_iid'
BATCH_SIZE = 1000     # Batch size while training
N_LOCAL_EPOCHS  = 1   # Number of epochs for local training
N_GLOBAL_EPOCHS = 100 # Number of epochs for global training
N_SERVERS  = 0        # Number of servers
N_CLIENTS  = 2        # Number of clients
dataset_name = 'test' # 'fmnist' or 'cifar10'

# Testing the gradient estimate method
class FashionMNIST_Classifier(nn.Module):
    def __init__(self):
        # Inherit the __init__() method from nn.Module (call __init__() from the parent class of FashionMNIST_Classifier)
        super(FashionMNIST_Classifier, self).__init__()

        # Define layers of the FashionMNIST_Classifier
        # padding = (k - 1) / 2 to preserve size for square kernel of size 3 and stride 1
        self.conv0 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (3, 3), stride = (1, 1), padding = 1)
        self.relu0 = nn.ReLU()
        self.drop0 = nn.Dropout(p = 0.5)
        self.pool0 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))

        # Next add another layers batch: convolutional, batch normalization, relu, max pooling
        self.conv1 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = 1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p = 0.5)
        self.pool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))

        # Finally add last layers batch, that includes flattening the output, and 2 linear layers (only add linear here)
        self.lin0  = nn.Linear(in_features = 64 * 7 * 7, out_features = 128)
        self.drop2 = nn.Dropout(p = 0.5)
        self.lin1  = nn.Linear(in_features = 128, out_features = 10)

    # Push the input through the net
    def forward(self, x):
        # Reshape the input data to have desired shape (specific for given data, change as needed, here 1 channel for MNIST)
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2]) 
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.drop0(x)
        x = self.pool0(x)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.pool1(x)

        x = torch.flatten(x, 1) # Flatten the tensor, except the batch dimension
        x = self.lin0(x)
        x = self.drop2(x)
        x = self.lin1(x)

        return x

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.lin0 = nn.Linear(in_features = 1, out_features = 1)
    def forward(self, x):
        x = x ** 2
        x = self.lin0(x)
        return x

def func_to_find(x):
    return 1 if 2 * x**2 - 50 >= 0 else 0
# Set the model used
if dataset_name == 'fmnist':
    NetBasic = FashionMNIST_Classifier
elif dataset_name == 'cifar10':
    NetBasic = CIFAR10_Classifier
else:
    NetBasic = TestModel

net = NetBasic()

class CompiledModel():
    def __init__(self, model, optimizer, loss_func):
        self.model = model          # Set the class model attribute to passed mode
        self.optimizer = optimizer  # Set optimizer attribute to passed optimizer
        self.loss_func = loss_func  # Set loss function attribute to passed loss function
        # self.local_z_model = copy.deepcopy(self.model)  # Create a copy of the model that will store the z model for push-sum training
        # self.local_z_model_loss = nn.CrossEntropyLoss() # Loss for local z model
        # self.local_z_model_optim = optim.Adam(self.local_z_model.parameters())
    # Training method
    def train(self, train_data, train_labels, train_batch_size, n_epoch, show_progress = False):
        self.model.train() # Put the model in training mode

        # Create a dataloader for training
        data   = train_data.clone().detach()    # Represent data as a tensor
        labels = train_labels.clone().detach()  # Represent labels as a tensor
        train_dataset = data_utils.TensorDataset(data, labels) # Create the training dataset
        # Create the train dataloader
        train_dataloader = DataLoader(dataset = train_dataset, batch_size = train_batch_size, shuffle = True)

        # Next fit the model by running the training loop
        # Run for specified number of epochs
        for epoch in range(n_epoch):
            # Then for each batch of specified batch size, train the model
            for x, y in train_dataloader:
                self.optimizer.zero_grad()       # Reset the gradients
                output = self.model(x)           # Calculate the outputs for batch inputs
                loss = self.loss_func(output, y) # Calculate the loss
                loss.backward()                  # Calculate gradients
                self.optimizer.step()            # Update the parameters
                # Print batch loss 
                if show_progress: 
                    print(f'Epoch: {epoch}, Batch loss: {loss}')

    # Calculate gradients
    def calc_grads(self, train_data, train_labels, train_batch_size, n_epoch, show_progress = False):
        self.model.train() # Put the model in training mode
        local_loss = nn.CrossEntropyLoss()
        local_optim = optim.Adam(self.model.parameters())
        # Create a dataloader for training
        data   = train_data.clone().detach()    # Represent data as a tensor
        labels = train_labels.clone().detach()  # Represent labels as a tensor
        train_dataset = data_utils.TensorDataset(data, labels) # Create the training dataset
        # Create the train dataloader
        train_dataloader = DataLoader(dataset = train_dataset, batch_size = train_batch_size, shuffle = True)
        # Get the batch
        x, y = next(iter(train_dataloader))
        local_optim.zero_grad() # Reset the gradients for optimizer
        self.model.zero_grad()     # Reset the gradients for model 
        output = self.model(x)     # Calculate the outputs for batch inputs
        loss = local_loss(output, y) # Calculate the loss
        loss.backward()                  # Calculate gradients
        # Get the gradients
        gradients = [param.grad.detach().clone() for param in self.model.parameters()]

        return gradients

    def set_grads_and_step(self, gradients, step_size):
        self.model.train() # Put the model in training mode
        local_optim = optim.SGD(self.model.parameters(), lr = step_size)
        # Reset the gradients for optimizer and model
        local_optim.zero_grad() 
        self.model.zero_grad()   
        # Set new gradients
        for model_param, grad_new in zip(self.model.parameters(), gradients):
            model_param.grad = grad_new.clone()
        # Update model
        local_optim.step()
        
    # Validate the model method
    def validate(self, valid_data, valid_labels):
        self.model.eval() # Set model in the evaluation mode
        
        # Create validation dataloader
        data = valid_data.clone().detach()     # Get validation data
        labels = valid_labels.clone().detach() # Get validation labels
        valid_dataset = torch.utils.data.TensorDataset(data, labels) # Create the validation dataset
        # Create validation dataloader
        valid_dataloader = DataLoader(dataset = valid_dataset, shuffle = True)

        # Initialize validation parameters
        total_loss    = 0 # Total loss measured
        total_correct = 0 # Total correct predictions
        total_pts     = len(valid_dataloader.dataset) # Total number of all points
        with torch.no_grad():
            for x, y in valid_dataloader:
                output = self.model(x)                                 # Predict the output
                total_loss += self.loss_func(output, y)                # Calculate loss for given datapoint and increment total loss
                total_correct += (torch.argmax(output).item() == y.item())

        # Find loss and accuracy
        valid_loss  = total_loss / total_pts
        valid_accur = total_correct / total_pts

        # Set model back in training mode
        self.model.train()

        # Return the calculated validation loss and validation accuracy
        return valid_loss, valid_accur

    # Return parameters of the network
    def get_params(self):
        return list(self.model.parameters())

    # Set parameters of the network to some specified parameters of the same network type (i.e. all layers match shapes)
    def set_params(self, param_list):
        # Iterate over new parameters and copy into current ones
        with torch.no_grad():
            for param_curr, param_new in zip(self.model.parameters(), param_list):
                param_curr.data.copy_(param_new)
    

a = CompiledModel(net, optim.Adam(net.parameters()), nn.CrossEntropyLoss)

'''
print(a.optimizer.param_groups[0]['params'])
print([_ for _ in a.model.parameters()])
param_list = a.get_params()
print(param_list)
param_list = [_ / 2 for _ in param_list]
a.set_params(param_list)
print(a.optimizer.param_groups[0]['params'])
print([_ for _ in a.model.parameters()])
'''
feature_list = [torch.tensor([_], dtype = torch.float32, requires_grad = True) for _ in (7.0, 1.0, 4.0, 3.3, 5.7, 0.0, -0.5, -2.5, -6.0, 12, -12, -7, )]
output_list = torch.stack([torch.tensor(func_to_find(_)) for _ in feature_list])
model_out = torch.stack([a.model(_) for _ in feature_list])
print(output_list, model_out)
print(a.optimizer.param_groups[0]['params'])
print(7.0 ** 2 * 0.5153 - 0.4414, model_out[0], a.model(torch.tensor([7.0])))
print(a.loss_func)
print(output_list.view(-1, 1))
loss = a.loss_func(model_out.view(-1, 1), output_list.view(-1, 1))
#print(f'Loss: {loss}')
# TODO: Just use the normal fmnist classifier. Fix all that