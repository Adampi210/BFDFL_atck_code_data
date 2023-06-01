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
import numpy as np
# Set manual seeds for reproducibility 
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

# Split data between servers and clients IID
def split_data_iid_incl_server(num_servers, num_clients, dataset_name):
    # Data directory
    data_dir = '~/data/datasets' + dataset_name
    # Decide on number of total data containers (chunks of data to split into)
    total_data_containers = num_clients + num_servers
    # Get the data, currently 3 datasets available
    if dataset_name == 'mnist':
        train_data = torchvision.datasets.MNIST(root = data_dir, train = True, download = True, transform = transforms.ToTensor())
        validation_data  = torchvision.datasets.MNIST(root = data_dir, train = False, transform = transforms.ToTensor())
    elif dataset_name == 'fmnist':
        train_data = torchvision.datasets.FashionMNIST(root = data_dir, train = True, download = True, transform = transforms.ToTensor())
        validation_data  = torchvision.datasets.FashionMNIST(root = data_dir, train = False, transform = transforms.ToTensor())
    elif dataset_name == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root = data_dir, train = True, download = True, transform = transforms.ToTensor())
        validation_data  = torchvision.datasets.CIFAR10(root = data_dir, train = False, transform = transforms.ToTensor())
    else:
        print(f'Currently {dataset_name} is not supported')
        return -1

    # Get the fractions that determine the split (i.e. what fraction of the dataset each split will have)
    data_split_fractions = [1 / total_data_containers for i in range(total_data_containers)]
    data_split_fractions[total_data_containers - 1] = 1 - sum(data_split_fractions[0:total_data_containers - 1])
    # Get the shuffled indecies for each client from training and validation datasets
    train_data_split = data_utils.random_split(train_data, data_split_fractions, torch.Generator())
    valid_data_split = data_utils.random_split(validation_data, data_split_fractions, torch.Generator())
    # Get the datasets for each client
    # For mnist and fmnist I can just use the indices method
    if dataset_name == 'mnist' or dataset_name == 'fmnist':
        train_list_dsets = [data_utils.TensorDataset(subset.dataset.data[subset.indices], subset.dataset.targets[subset.indices]) for subset in train_data_split]
        valid_list_dsets = [data_utils.TensorDataset(subset.dataset.data[subset.indices], subset.dataset.targets[subset.indices]) for subset in valid_data_split]
    # For cifar, iterate through indices manually
    elif dataset_name == 'cifar10':
        train_list_dsets = [data_utils.TensorDataset(torch.tensor(np.array([subset.dataset.data[i] for i in subset.indices])),
                                                    torch.tensor(np.array([subset.dataset.targets[i] for i in subset.indices])))
                            for subset in train_data_split]

        valid_list_dsets = [data_utils.TensorDataset(torch.tensor(np.array([subset.dataset.data[i] for i in subset.indices])),
                                                    torch.tensor(np.array([subset.dataset.targets[i] for i in subset.indices])))
                            for subset in valid_data_split]
    return train_list_dsets, valid_list_dsets


# This function only splits the data between the clients, currently split is iid
def split_data_iid_excl_server(num_clients, dataset_name):
    # Data directory
    data_dir = '~/data/datasets' + dataset_name
    # Get the data, currently 3 datasets available
    if dataset_name == 'mnist':
        train_data = torchvision.datasets.MNIST(root = data_dir, train = True, download = True, transform = transforms.ToTensor())
        validation_data  = torchvision.datasets.MNIST(root = data_dir, train = False, transform = transforms.ToTensor())
    elif dataset_name == 'fmnist':
        train_data = torchvision.datasets.FashionMNIST(root = data_dir, train = True, download = True, transform = transforms.ToTensor())
        validation_data  = torchvision.datasets.FashionMNIST(root = data_dir, train = False, transform = transforms.ToTensor())
    elif dataset_name == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root = data_dir, train = True, download = True, transform = transforms.ToTensor())
        validation_data  = torchvision.datasets.CIFAR10(root = data_dir, train = False, transform = transforms.ToTensor())
    else:
        print(f'Currently {dataset_name} is not supported')
        return -1
    # Get the fractions that determine the split (i.e. what fraction of the dataset each split will have)
    data_split_fractions = [1 / num_clients for i in range(num_clients)]
    if sum(data_split_fractions) != 1:
        data_split_fractions[num_clients - 1] = 1 - sum(data_split_fractions[0:num_clients - 1])
    # Get the shuffled indecies for each client from training and validation datasets
    train_data_split = data_utils.random_split(train_data, data_split_fractions, torch.Generator())
    valid_data_split = data_utils.random_split(validation_data, data_split_fractions, torch.Generator())
    # Get the datasets for each client
    # For mnist and fmnist I can just use the indices method
    if dataset_name == 'mnist' or dataset_name == 'fmnist':
        train_list_dsets = [data_utils.TensorDataset(subset.dataset.data[subset.indices], subset.dataset.targets[subset.indices]) for subset in train_data_split]
        valid_list_dsets = [data_utils.TensorDataset(subset.dataset.data[subset.indices], subset.dataset.targets[subset.indices]) for subset in valid_data_split]
    # For cifar, iterate through indices manually
    elif dataset_name == 'cifar10':
        train_list_dsets = [data_utils.TensorDataset(torch.tensor(np.array([subset.dataset.data[i] for i in subset.indices])),
                                                    torch.tensor(np.array([subset.dataset.targets[i] for i in subset.indices])))
                            for subset in train_data_split]

        valid_list_dsets = [data_utils.TensorDataset(torch.tensor(np.array([subset.dataset.data[i] for i in subset.indices])),
                                                    torch.tensor(np.array([subset.dataset.targets[i] for i in subset.indices])))
                            for subset in valid_data_split]

    return train_list_dsets, valid_list_dsets


def split_data_non_iid_incl_server(clients):
    # TODO
    pass

def split_data_non_iid_excl_server(clients):
    # TODO
    pass

# Use FMNIST 
# Use 50 devices
# Decentralized + 1 adversarial + consider centrality + undirected + DFedAvg (assume matrix to be doubly stochastic)
# Add non-iid
servers, clients = 1, 10
dset_len = clients
a,b = split_data_iid_excl_server(clients, 'cifar10')
[print(a[i][0][1]) for i in range(dset_len)]
