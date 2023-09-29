import sys
import os
import re

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
import time
import csv

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method # FGSM
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent # PGD
from cleverhans.torch.attacks.noise import noise # Basic uniform noise perturbation

# User defined
from split_data import *
from nn_FL_de_cent import *
from neural_net_architectures import *
# Device configuration
# Always check first if GPU is avaialble
device_used = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# If CUDA is not avaialbe, print message that CPU will be used
if device_used != torch.device('cuda:1'):
    print(f'CUDA not available, have to use {device_used}')

start_time = time.time()
# Set hyperparameters
seed = 37 # Seed for PRNGs 
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Aggregation and datase parameters
dataset_name = 'fmnist' # 'fmnist' or 'cifar10'
dataset_size = int(6e4) if dataset_name in ['fmnist', 'mnist'] else int(5e4)
aggreg_schemes = ('push_sum', 'sab', 'belief_secure_push_sum', 'test')
aggregation_mechanism = aggreg_schemes[1]

# Topology used from a filename
# Create directory for the network data. Will include accuracy sub-directories
dir_networks = '../../data/full_decentralized/network_topologies'
dir_data = '../../data/full_decentralized/%s/' % dataset_name
graph_type = ('ER', 'dir_scale_free', 'dir_geom', 'k_out', 'pref_attach', 'SNAP_Cisco')
graph_type_used = graph_type[2]
# This is the source for network topology

# ADJUSTABLE #####
designated_clients = 20
# ER
if graph_type_used == 'ER':
    prob_conn = 5
    data_dir_name = dir_data + '%s_graph_c_%d_p_0%d/' % (graph_type_used, designated_clients, prob_conn)
    network_topology = '%s_graph_c_%d_p_0%d_seed_%d.txt' % (graph_type_used, designated_clients, prob_conn, seed)
# DIR GEOM
elif graph_type_used == 'dir_geom':
    geo_graph_configs = ('2d_very_close_nodes', '2d_close_nodes', '2d_far_nodes')
    config_used = 2
    data_dir_name = dir_data + '%s_graph_c_%d_type_%s/' % (graph_type_used, designated_clients, geo_graph_configs[config_used])
    network_topology = '%s_graph_c_%d_type_%s_seed_%d.txt' % (graph_type_used, designated_clients, geo_graph_configs[config_used], seed)
# K-OUT
elif graph_type_used == 'k_out':
    k_dec = 0.25
    k_used = int(designated_clients * k_dec)
    data_dir_name = dir_data + '%s_graph_c_%d_k_%d/' % (graph_type_used, designated_clients, k_used)
    network_topology = '%s_graph_c_%d_k_%d_seed_%d.txt' % (graph_type_used, designated_clients, k_used, seed)
# PREF_ATTACH
elif graph_type_used == 'pref_attach':
    pref_attach_configs = ('sparse', 'medium', 'dense')
    config_used = 0
    data_dir_name = dir_data + '%s_graph_c_%d_type_%s/' % (graph_type_used, designated_clients, pref_attach_configs[config_used])
    network_topology = '%s_graph_c_%d_type_%s_seed_%d.txt' % (graph_type_used, designated_clients, pref_attach_configs[config_used], seed)
# SNAP
elif graph_type_used == 'SNAP_Cisco':
    client_val_used = 0
    seed_graph = 0
    client_vals = []
    graph_types = {}
    # Iterate over all files 
    for filename in os.listdir(dir_networks):
        # Check if the filename starts with 'SNAP_Cisco' and ends with '_seed_x.txt'
        if filename.startswith('SNAP_Cisco') and filename.endswith('_seed_%d.txt' % seed_graph):
            # Use a regex to find the number after 'c_'
            match = re.search(r'c_(\d+)_type_(g\d+)_', filename)
            if match:
                # If a match was found, add the number to the list
                client_vals.append(int(match.group(1)))
                graph_types[int(match.group(1))] = match.group(2)
    client_vals = sorted(client_vals)
    data_dir_name = dir_data + '%s_c_%d_type_%s/' % (graph_type_used, client_vals[client_val_used], graph_types[client_vals[client_val_used]])
    network_topology = '%s_c_%d_type_%s_seed_%d.txt' % (graph_type_used, client_vals[client_val_used], graph_types[client_vals[client_val_used]], seed_graph)

##################
network_topology_filepath = os.path.join(dir_networks, network_topology)
adj_matrix = np.loadtxt(network_topology_filepath)
os.makedirs(data_dir_name, exist_ok = True)

# Save the adjacency matrix, the graph graphical representation, and the client centralities
# np.savetxt(data_dir_name + network_topology, adj_matrix)

# Create and save graph
graph_representation = create_graph(adj_matrix)
# graph_plot = nx.draw_networkx(graph_representation, with_labels = True)
# plt.savefig(data_dir_name + 'graph_picture' + '.png')
# Save network centralities
centrality_data = calc_centralities(len(adj_matrix[0]), graph_representation)

with open(data_dir_name + 'node_centrality'+ '.csv', 'w', newline = '') as centrality_data_file:
    writer = csv.writer(centrality_data_file)
    for node_id in centrality_data.keys():
        data_cent_node = [node_id]
        data_cent_node.extend(centrality_data[node_id])
        writer.writerow(data_cent_node)

# Training parameters
N_CLIENTS = len(adj_matrix[0]) # Number of clients
N_SERVERS  = 0        # Number of servers
iid_type = 'non_iid'      # 'iid' or 'non_iid'
N_LOCAL_EPOCHS  = 10  # Number of epochs for local training
N_GLOBAL_EPOCHS = 100 # Number of epochs for global training
BATCH_SIZE = 500 # Batch size while training

# Adversarial parameters
attacks = ('none', 'FGSM', 'PGD', 'noise')      # Available attacks
architectures = ('star', 'full_decentralized')  # Architecture used
attack_used = 1                                 # Which attack from the list was used
attack = attacks[0]                             # Always start with no attack (attack at some point)
adv_pow = 100                                     # Power of the attack
adv_percent = 0.2                               # Percentage of adversaries
adv_number = int(adv_percent * N_CLIENTS)       # Number of adversaries
# adv_list = list(range(adv_number))
# adv_list = random.sample(list(range(N_CLIENTS)), adv_number) # Choose the adversaries at random
attack_time = 25                                # Global epoch at which the attack activates
# PGD attack parameters
eps_iter = 0.0 # Learning rate for PGD attack
nb_iter = 15   # Number of epochs for PGD attack

# Define centrality measures and directories
centralities = ('none', 'in_deg_centrality', 'out_deg_centrality', 'closeness_centrality', 'betweeness_centrality', 'eigenvector_centrality')
cent_measure_used = 5

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

# Run simulations and save the data
def run_and_save_simulation(train_split, valid_split, adj_matrix, centrality_measure = 0):
    # Initialize list storing all the nodes
    node_list = []
    acc_clients = [0] * N_CLIENTS
    loss_clients = [0] * N_CLIENTS
    # Create nodes in the graph
    node_list = [client_FL(i, device = device_used) for i in range(N_CLIENTS)]
    [node.get_data(train_split[i], valid_split[i]) for node, i in zip(node_list, range(N_CLIENTS))]
    [node.init_compiled_model(NetBasic()) for node in node_list] # This takes 36s for 10 clients
    # Add neigbors, specified graph
    create_clients_graph(node_list, adj_matrix, 0)

    # Sort by centralities
    # nodes_to_atk_centrality = sort_by_centrality(centrality_data) # For normal operation
    # New framework #########################
    # prefix_name = 'score_cent_dist_manual_weight_0%d' % int(10 * score_cent_dist_weight) # For centrality-distance tradeoff
    # prefix_name = 'cluster_metis_alg' # For creating clusters based on the metis algorithm and choosing most central node for each cluster
    prefix_name = 'least_overlap_area' # For creating clusters based on the new least overlap area algorithm
    # prefix_name = 'random_nodes'
    if 'score_cent_dist_manual_weight_0' in prefix_name:
        score_cent_dist_weight = 0 # 1 is the same as original, only choose by centralities, 0 chooses most spread out nodes
        if centralities[centrality_measure] == 'none':
            nodes_to_atk_centrality = []
        else:
            nodes_to_atk_centrality = score_cent_dist_manual(score_cent_dist_weight, N_CLIENTS, adv_number, graph_representation, centrality_measure - 1)
    elif 'cluster_metis_alg' in prefix_name:
        if centralities[centrality_measure] == 'none':
            nodes_to_atk_centrality = []
        else:
            nodes_to_atk_centrality = cluster_metis_alg(N_CLIENTS, adv_number, graph_representation, centrality_measure - 1)
    elif 'random_nodes' in prefix_name:
        if centralities[centrality_measure] == 'none':
            nodes_to_atk_centrality = []
        else:
            nodes_to_atk_centrality = random_nodes(N_CLIENTS, adv_number)
    elif 'least_overlap_area' in prefix_name:
        if centralities[centrality_measure] == 'none':
            nodes_to_atk_centrality = []
        else:
            nodes_to_atk_centrality = least_overlap_area(N_CLIENTS, adv_number, graph_representation)
    # Init accuracy and loss values and files
    curr_loss, curr_acc = 0, 0
    centrality_used = centralities[centrality_measure]
    # atk_%s_advs_%d_adv_pow_%s_atk_time_%d_seed_%d_iid_type_%s/' % (attacks[attack_used], adv_number, str(adv_pow), attack_time, seed, iid_type)
    file_acc_name = data_dir_name + 'acc_%s_atk_%s_advs_%d_adv_pow_%s_atk_time_%d_seed_%d_iid_type_%s_cent_%s' % (prefix_name, attacks[attack_used], adv_number, str(adv_pow), attack_time, seed, iid_type, centrality_used)
    file_loss_name = data_dir_name + 'loss_%s_atk_%s_advs_%d_adv_pow_%s_atk_time_%d_seed_%d_iid_type_%s_cent_%s' % (prefix_name, attacks[attack_used], adv_number, str(adv_pow), attack_time, seed, iid_type, centrality_used)

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
                # adv_list = nodes_to_atk_centrality[centrality_measure - 1][0:adv_number] # Old framework
                adv_list = nodes_to_atk_centrality
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
    print(f'File network: {network_topology}')
    print(f'Seed: {seed}, Adv percent: {adv_percent}, Adv power: {adv_pow}')
    print(f'iid_type: {iid_type}')
    print(f'Centrality: {centralities[cent_measure_used]}')
    run_and_save_simulation(train_dset_split, valid_dset_split, adj_matrix, cent_measure_used)
    print('Total time %lfs' % (time.time() - start_time))
    


# Run noise attack, make sure the noise is significant in comparison to the information being aggregated
# Noise is chosen to be a function of the aggregated gradients, function chosen to be proportional to the gradients without randomness
# Check different types of graphs, random connection model, preferencial attachment model, geometric, scale-free graph
# 