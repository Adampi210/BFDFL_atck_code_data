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
from plot_basic_data import plot_acc_loss_data

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method # FGSM
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent # PGD
from cleverhans.torch.attacks.noise import noise # Basic uniform noise perturbation

# Device configuration
# Always check first if GPU is avaialble
device_used = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# If CUDA is not avaialbe, print message that CPU will be used
if device_used != torch.device('cuda'):
    print(f'CUDA not available, have to use {device_used}')

# Set hyperparameters
# Training parameters
BATCH_SIZE = 500     # Batch size while training
N_LOCAL_EPOCHS  = 1  # Number of epochs for local training
N_GLOBAL_EPOCHS = 10 #
N_SERVERS  = 1       # Number of servers
N_CLIENTS  = 10      # Number of clients
# Remark: With 0 servers and 1 client I get the same scenarion as in mnist_basic_net

# Adversarial parameters
attacks = ('None', 'FGSM', 'PGD', 'Noise')
architectures = ('star', 'fully_decentralized')
attack_used = 1
attack = attacks[0]
adv_pow = 1
adv_number = 0
adv_list = random.sample(list(range(N_CLIENTS)), adv_number)
attack_time = 4
# PGD attack parameters

# Next, create a class for the neural net that will be used
filename = '../../data/star/' + 'attack_%s_adv_pow_%s_adv_number_%s_atk_time_%s_architecture_%s.csv' % (attacks[attack_used], str(adv_pow), str(adv_number), attack_time, architectures[0])

class NetBasic(nn.Module):
    def __init__(self):
        # Inherit the __init__() method from nn.Module (call __init__() from the parent class of NetBasic)
        super(NetBasic, self).__init__()

        # Define layers of the NetBasic
        # Add first convolutional layer (stride specifies to move kernel by 1 pixel horizontally and vertically, padding = 'same' specifies to pad the output with 0s to preserve image size (decreases in convolution))
        self.conv0 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3, 3), stride = (1, 1), padding = 'same')
        # Add Batch Normalization layer (Basically normalizes batch by subtracting mean and dividing by st.dev.)
        # num_features = input channels, eps = scalable epsilon value used in normalization formula, momentum adds moving average (takes 0.1 * prev input + 0.9 * new)
        # And affine allows to learn the gamma and beta parameters used in batch normalization
        self.bn0   = nn.BatchNorm2d(num_features = 16, momentum = 0.1, eps = 1e-5, affine = True)
        # Add relu activation layer
        self.relu0 = nn.ReLU()
        # Add max pooling layer
        self.pool0 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))

        # Next add another layers batch: convolutional, batch normalization, relu, max pooling
        self.conv1 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3, 3), stride = (1, 1), padding = 'same')
        self.bn1   = nn.BatchNorm2d(num_features = 32, momentum = 0.1, eps = 1e-5, affine = True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))

        # And another layers batch, similar to previous, no pooling this time
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = 'same')
        self.bn2   = nn.BatchNorm2d(num_features = 64, momentum = 0.1, eps = 1e-5, affine = True)
        self.relu2 = nn.ReLU()

        # Finally add last layers batch, that includes flattening the output, and 2 linear layers (only add linear here)
        self.lin0  = nn.Linear(in_features = 64 * 7 * 7, out_features = 128)
        self.lin1  = nn.Linear(in_features = 128, out_features = 10)

    # Push the input through the net
    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = torch.flatten(x, 1) # Flatten the tensor, except the batch dimension
        x = self.lin0(x)
        x = self.lin1(x)

        return x

        
# Split the data for the specified number of clients and servers
train_dset_split, valid_dset_split = split_data_uniform_excl_server(N_CLIENTS)

# Initialize the main server
main_server = Server_FL()
main_server.get_data(valid_dset_split[0])

# Add clients to the server
for i in range(N_CLIENTS):
    temp_client = client_FL(client_id = i, if_adv_client = True if i in adv_list else False)
    temp_client.get_data(train_dset_split[i], valid_dset_split[i])
    temp_client.init_compiled_model()
    main_server.add_client(temp_client)

for client in main_server.list_clients:
    print(client.client_id, client.if_adv_client)

# Check initial accuracy
main_server.aggregate_client_models_fed_avg()
main_server.validate_global_model()
main_server.distribute_global_model()

# Train and test
if __name__ == "__main__":
    global_loss, global_acc = 0, 0
    with open(filename, 'w', newline = '') as file_data:
        # Setup a writer
        writer = csv.writer(file_data)
        # Training
        for i in range(N_GLOBAL_EPOCHS):
            if i == attack_time:
                attack = attacks[attack_used]
            print(f'global epoch: {i}')
            for client in main_server.list_clients:
                client.train_client(BATCH_SIZE, N_LOCAL_EPOCHS)
                client.validate_client()

            main_server.aggregate_client_models_fed_avg()
            global_loss, global_acc = main_server.validate_global_model()
            main_server.distribute_global_model()
            writer.writerow([i, global_loss.data.item(), global_acc])
    plot_acc_loss_data(filename)

# Use FMNIST 
# Use 50 devices
# Decentralized + 1 adversarial + consider centrality + undirected + DFedAvg (assume matrix to be doubly stochastic)
# Add non-iid