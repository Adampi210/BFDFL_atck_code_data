import sys
sys.path.append("../plot_data/")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import random

import csv

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method # FGSM
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent # PGD
from cleverhans.torch.attacks.noise import noise # Basic uniform noise perturbation

# User defined
from split_data import *
from nn_FL import *

# Device configuration
# Always check first if GPU is avaialble
device_used = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# If CUDA is not avaialbe, print message that CPU will be used
if device_used != torch.device('cuda:1'):
    print(f'CUDA not available, have to use {device_used}')

# Set hyperparameters
seed = 0 # Seed for PRNGs 
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
# Training parameters
iid_type = 'iid'      # 'iid' or 'non_iid'
BATCH_SIZE = 100      # Batch size while training
N_LOCAL_EPOCHS  = 1   # Number of epochs for local training
N_GLOBAL_EPOCHS = 100 # Number of epochs for global training
N_SERVERS  = 0        # Number of servers
N_CLIENTS  = 10       # Number of clients
dataset_name = 'cifar10' # 'fmnist' or 'cifar10'

# Adversarial parameters
attacks = ('none', 'FGSM', 'PGD', 'noise')      # Available attacks
architectures = ('star', 'full_decentralized')  # Architecture used
attack_used = 0                                 # Which attack from the list was used
attack = attacks[0]                             # Always start with no attack (attack at some point)
adv_pow = 1                                     # Power of the attack
adv_percent = 0.1                               # Percentage of adversaries
adv_number = int(adv_percent * N_CLIENTS)       # Number of adversaries
adv_list = random.sample(list(range(N_CLIENTS)), adv_number) # Choose the adversaries at random
attack_time = 0                                 # Global epoch at which the attack activates
# PGD attack parameters
eps_iter = 0.1 # Learning rate for PGD attack
nb_iter = 15   # Number of epochs for PGD attack
# Filename for the saved data
filename = '../../data/full_decentralized/%s/' % (dataset_name) + 'atk_%s_advs_%d_adv_pow_%s_clients_%d_atk_time_%d_arch_%s_seed_%d_iid_type_%s' % (attacks[attack_used], adv_number, str(adv_pow), N_CLIENTS, attack_time, architectures[0], seed, iid_type)

print(f'Servers: {N_SERVERS}, Clients: {N_CLIENTS}, Adversaries: {adv_number}')
print(f'iid-type: {iid_type}, seed: {seed}')
print(f'Attack used: {attacks[attack_used]}, Architecture: {architectures[1]}')

# Next, create a class for the neural net that will be used
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

class CIFAR10_Classifier(nn.Module):
    def __init__(self):
        # Inherit __init__ from nn.Module
        super(CIFAR10_Classifier, self).__init__()

        # Define layers for CIFAR10_Classifier
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = (3, 3), stride = (1, 1), padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3, 3), stride = (1, 1), padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),
            nn.Dropout(p = 0.5)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),
            nn.Dropout(p = 0.5)
        )

        self.fcls = nn.Sequential(
            nn.Linear(in_features = 64 * 8 * 8, out_features = 256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 10)
        )

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[3], x.shape[1], x.shape[2]) 

        x = self.conv0(x)
        x = self.conv1(x)
        x = x.view(x.size(0), -1) # Flatten the tensor, except the batch dimension
        x = self.fcls(x)

        return x

# Split the data for the specified number of clients and servers
if iid_type == 'iid':
    train_dset_split, valid_dset_split = split_data_iid_excl_server(N_CLIENTS, dataset_name)
elif iid_type == 'non_iid':
    train_dset_split, valid_dset_split = split_data_non_iid_excl_server(N_CLIENTS, dataset_name)

# Set the model used
if dataset_name == 'fmnist':
    NetBasic = FashionMNIST_Classifier
elif dataset_name == 'cifar10':
    NetBasic = CIFAR10_Classifier

# Initialize list storing all the nodes
node_list = []

# Create nodes in the graph
for i in range(N_CLIENTS):
    temp_node = client_FL(client_id = i, if_adv_client = True if i in adv_list else False)
    temp_node.get_data(train_dset_split[i], valid_dset_split[i])
    temp_node.init_compiled_model(NetBasic())
    node_list.append(temp_node)

# Add neigbors, for now just do the circle 
for i in range(N_CLIENTS - 1):
    node_list[i].add_neighbor(node_list[i + 1])
node_list[N_CLIENTS - 1].add_neighbor(node_list[0])


# Train and test
if __name__ == "__main__":
    global_loss, global_acc = 0, 0
    with open(filename + '.csv', 'w', newline = '') as file_data:
        # Setup a writer
        writer = csv.writer(file_data)
        # Training
        for i in range(N_GLOBAL_EPOCHS):
            if i == attack_time:
                attack = attacks[attack_used]
                for node in node_list:
                    if node.if_adv_client:
                        node.attack = attack
                        node.adv_pow = adv_pow
            print(f'global epoch: {i}')
            for node in node_list:
                node.train_client(BATCH_SIZE, N_LOCAL_EPOCHS, show_progress = False, eps_iter = eps_iter, nb_iter = nb_iter)
            for node in node_list:
                node.aggregate_d_fed_avg()
            global_acc = 1.0
            for node in node_list:
                curr_loss, curr_acc = node.validate_client(True)
                if curr_acc < global_acc:
                    global_loss, global_acc = curr_loss, curr_acc

            writer.writerow([i, global_loss.item(), global_acc])

# How far the attack goes based on centrality - epidemic spread, epidemic modelling over networks, topology specific epidemic modelling
# Infocomm sigcomm, graph theory









