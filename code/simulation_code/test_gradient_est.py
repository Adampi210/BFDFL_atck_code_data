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
import copy
import csv

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method # FGSM
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent # PGD
from cleverhans.torch.attacks.noise import noise # Basic uniform noise perturbation

from split_data import *
from nn_FL_1 import *
# Device configuration
# Always check first if GPU is avaialble
device_used = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# If CUDA is not avaialbe, print message that CPU will be used
if device_used != torch.device('cuda'):
    print(f'CUDA not available, have to use {device_used}')
# Set hyperparameters
seed = 0 # Seed for PRNGs 
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

iid_type = 'iid'      # 'iid' or 'non_iid'
BATCH_SIZE = 1000     # Batch size while training
N_LOCAL_EPOCHS  = 1   # Number of epochs for local training
N_GLOBAL_EPOCHS = 100 # Number of epochs for global training
N_SERVERS  = 0        # Number of servers
N_CLIENTS  = 10       # Number of clients
dataset_name = 'fmnist' # 'fmnist' or 'cifar10'

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

# Set the model used
if dataset_name == 'fmnist':
    NetBasic = FashionMNIST_Classifier
elif dataset_name == 'cifar10':
    NetBasic = CIFAR10_Classifier

# Split the data for the specified number of clients and servers
if iid_type == 'iid':
    train_dset_split, valid_dset_split = split_data_iid_excl_server(N_CLIENTS, dataset_name)
elif iid_type == 'non_iid':
    train_dset_split, valid_dset_split = split_data_non_iid_excl_server(N_CLIENTS, dataset_name)


if __name__ == '__main__':
    acc_list = []
    client_list = [client_FL(i) for i in range(N_CLIENTS)]
    [x.get_data(train_dset_split[i], valid_dset_split[i]) for x, i in zip(client_list, range(N_CLIENTS))]
    [x.init_compiled_model(NetBasic()) for x in client_list]
    [client_list[i].add_neighbor(client_list[i - 1]) for i in range(1, len(client_list))]
    adj_matrix = gen_rand_adj_matrix(N_CLIENTS)
    create_clients_graph(client_list, adj_matrix, None)

    for i in range(100):
        [x.exchange_models() for x in client_list]
        [x.aggregate_SAB(500, 5, False) for x in client_list]
        print(f'Epoch {i}')
        acc_list.append(sum([x.validate_client(False)[-1] for x in client_list]) / len(client_list))
    
    # Create a new figure and a subplot
    fig, ax = plt.subplots()

    # Create line plot
    ax.plot(range(len(acc_list)), acc_list)

    # Set the title and labels
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')

    # Save the figure
    plt.savefig('test_accuracy.png')

    # Show the plot
    plt.show()


