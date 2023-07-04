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

# User defined
from split_data import *
from nn_FL_1 import *

# Device configuration
# Always check first if GPU is avaialble
device_used = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# If CUDA is not avaialbe, print message that CPU will be used
if device_used != torch.device('cuda'):
    print(f'CUDA not available, have to use {device_used}')

# Set hyperparameters
seed = 1 # Seed for PRNGs 
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
# Training parameters
iid_type = 'iid'      # 'iid' or 'non_iid'
BATCH_SIZE = 500     # Batch size while training
N_LOCAL_EPOCHS  = 5   # Number of epochs for local training
N_GLOBAL_EPOCHS = 100 # Number of epochs for global training
N_SERVERS  = 0        # Number of servers
N_CLIENTS  = 10       # Number of clients
dataset_name = 'fmnist' # 'fmnist' or 'cifar10'
aggreg_schemes = ('push_sum', 'sab', 'belief_secure_push_sum', 'test')
aggregation_mechanism = aggreg_schemes[1]
# 0: 
#   First all local train with Adam
#   Then all aggregate from neigbors (without changing model params)
#   Then all update model params
# Adversarial parameters
attacks = ('none', 'FGSM', 'PGD', 'noise')      # Available attacks
architectures = ('star', 'full_decentralized')  # Architecture used
attack_used = 0                                 # Which attack from the list was used
attack = attacks[0]                             # Always start with no attack (attack at some point)
adv_pow = 0                                     # Power of the attack
adv_percent = 0.0                               # Percentage of adversaries
adv_number = int(adv_percent * N_CLIENTS)       # Number of adversaries
# adv_list = list(range(adv_number))
# adv_list = random.sample(list(range(N_CLIENTS)), adv_number) # Choose the adversaries at random
attack_time = 25 if attack_used != 0 else 0      # Global epoch at which the attack activates
# PGD attack parameters
eps_iter = 0.1 # Learning rate for PGD attack
nb_iter = 15   # Number of epochs for PGD attack
# Filename for the saved data
dir_name = '../../data/full_decentralized/%s/' % (dataset_name) + \
    'atk_%s_advs_%d_adv_pow_%s_clients_%d_atk_time_%d_arch_%s_seed_%d_iid_type_%s_%s/' % (attacks[attack_used], adv_number, str(adv_pow), N_CLIENTS, attack_time, architectures[0], seed, iid_type, aggregation_mechanism)
os.makedirs(dir_name, exist_ok = True)

centralities = ('in_deg_centrality', 'out_deg_centrality', 'closeness_centrality', 'betweeness_centrality', 'eigenvector_centrality')

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

def run_and_save_simulation(train_split, valid_split, centrality_measure = None):
    # Initialize list storing all the nodes
    node_list = []
    acc_clients = [0] * N_CLIENTS
    loss_clients = [0] * N_CLIENTS
    
    # Create nodes in the graph
    node_list = [client_FL(i) for i in range(N_CLIENTS)]
    [node.get_data(train_split[i], valid_split[i]) for node, i in zip(node_list, range(N_CLIENTS))]
    [node.init_compiled_model(NetBasic()) for node in node_list]

    # Add neigbors, random graph
    random_plain_adj_list = gen_rand_adj_matrix(N_CLIENTS)
    adj_matrix, graph_representation = create_clients_graph(node_list, random_plain_adj_list, 0)
    
    # Save the adjacency matrix
    adj_matrix_figname = dir_name + 'adj_matrix_' + str(hash_np_arr(adj_matrix))
    np.savetxt(adj_matrix_figname, adj_matrix)
    centrality_data = calc_centralities(node_list, graph_representation)
    graph_plot = nx.draw_networkx(graph_representation, with_labels = True)
    plt.savefig(dir_name + 'graph_' + str(hash_np_arr(adj_matrix)) + '.png')

    # Save network centralities
    with open(dir_name + 'centrality_clients_' + str(hash_np_arr(adj_matrix)) + '.csv', 'w', newline = '') as centrality_data_file:
        writer = csv.writer(centrality_data_file)
        for node_id in centrality_data.keys():
            data_cent_node = [node_id]
            data_cent_node.extend(centrality_data[node_id])
            writer.writerow(data_cent_node)

    nodes_to_atk_centrality = sort_by_centrality(dir_name + 'centrality_clients_' + str(hash_np_arr(adj_matrix)) + '.csv')

    # Init accuracy and loss values and files
    curr_loss, curr_acc = 0, 0
    centrality_used = 'none' if centrality_measure == None else centralities[centrality_measure]
    file_acc_name = dir_name + 'acc_%s' % centrality_used + '_' + str(hash_np_arr(adj_matrix))
    file_loss_name = dir_name + 'loss_%s' % centrality_used + '_' + str(hash_np_arr(adj_matrix))
    with open(file_acc_name + '.csv', 'w', newline = '') as file_acc:
        with open(file_loss_name + '.csv', 'w', newline = '') as file_loss:
            # Setup writers
            writer_acc = csv.writer(file_acc)
            writer_loss = csv.writer(file_loss)
            # Training
            # Setup attack values
            attack = attacks[0]
            if centrality_used == 'none':
                adv_list = []
                writer_acc.writerow(['none', adv_list])
                writer_loss.writerow(['none', adv_list])
            else: 
                adv_list = nodes_to_atk_centrality[centrality_measure][0:adv_number]
                writer_acc.writerow([centralities[centrality_measure], adv_list])
                writer_loss.writerow([centralities[centrality_measure], adv_list])

            # Setup nodes
            for node in node_list:
                node.attack = attack
                node.adv_pow = adv_pow
                node.if_adv_client = False
                # Only for adversarial clients
                if node.client_id in adv_list:
                    node.if_adv_client = True
                    node.eps_iter = eps_iter
                    node.nb_iter = nb_iter


            # Run the training
            for i in range(N_GLOBAL_EPOCHS):
                # Update values if attacking
                if i == attack_time:
                    attack = attacks[attack_used]
                    for node in node_list:
                        if node.if_adv_client:
                            node.attack = attack
                            node.adv_pow = adv_pow
                            node.eps_iter = eps_iter
                            node.nb_iter = nb_iter
                # Train and aggregate
                print(f'global epoch: {i}')
                # Exchange models and aggregate, MAIN PART
                [node.exchange_models() for node in node_list]
                [node.aggregate_SAB(BATCH_SIZE, N_LOCAL_EPOCHS, False) for node in node_list]
                # Save accuracies
                for node in node_list:
                    curr_loss, curr_acc = node.validate_client()
                    acc_clients[node.client_id] = curr_acc
                    loss_clients[node.client_id] = curr_loss
                
                # Save data
                writer_acc.writerow([i, acc_clients])
                writer_loss.writerow([i, loss_clients])


if __name__ == '__main__':
    run_and_save_simulation(train_dset_split, valid_dset_split, None)


