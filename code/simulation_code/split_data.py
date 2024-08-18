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
import os
import json
import matplotlib.pyplot as plt
import re

# Set manual seeds for reproducibility 

# Number of classes each non-iid client should have
NUM_CLASSES_NONIID = 3


# Splits the dataset in a iid way, excluding server (servers should have all data)
def iid_split(data_set, num_containers, dataset_name = None):
    # Get the fractions that determine the split (i.e. what fraction of the dataset each split will have)
    data_split_fractions = [1 / num_containers for i in range(num_containers)]
    data_split_fractions[num_containers - 1] = 1 - sum(data_split_fractions[0:num_containers - 1])
    # Get the shuffled indecies for each client from training and validation datasets
    data_split = data_utils.random_split(data_set, data_split_fractions, torch.Generator())

    # Get the datasets for each client
    # For mnist and fmnist I can just use the indices method
    if dataset_name == 'mnist' or dataset_name == 'fmnist':
        devices_list_dsets = [data_utils.TensorDataset(subset.dataset.data[subset.indices], subset.dataset.targets[subset.indices]) for subset in data_split]
    # For cifar, iterate through indices manually
    elif dataset_name == 'cifar10':
        devices_list_dsets = [data_utils.TensorDataset(torch.tensor(np.array([subset.dataset.data[i] for i in subset.indices])),
                                                    torch.tensor(np.array([subset.dataset.targets[i] for i in subset.indices])))
                            for subset in data_split]
    return devices_list_dsets

# Split data between servers and clients iid
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

    train_list_dsets = iid_split(train_data, total_data_containers, dataset_name)
    valid_list_dsets = iid_split(validation_data, total_data_containers, dataset_name)

    return train_list_dsets, valid_list_dsets

# Split data betweeen clients iid
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
    
    train_list_dsets = iid_split(train_data, num_clients, dataset_name)
    valid_list_dsets = iid_split(validation_data, num_clients, dataset_name)

    return train_list_dsets, valid_list_dsets

# Splits the dataset in a non-iid way, excluding server (servers should have all data)
def non_iid_excl_split(data_set, num_clients, num_classes_non_iid = NUM_CLASSES_NONIID, client_class_split = None):
    print(f'Classes: {num_classes_non_iid}')
    # Calculate number of classes
    num_classes = len(data_set.classes)
    # Split the classes among clients, if not already
    if client_class_split == None:
        client_classes = [random.sample(list(range(num_classes)), num_classes_non_iid) for _ in range(num_clients)]
    else:
        client_classes = client_class_split
    # For each class, calculate how many clients use it
    class_usage = {i:0 for i in range(num_classes)}
    # Iterate through client classes to do that
    for client_class_list in client_classes:
        for client_class in client_class_list:
            class_usage[client_class] += 1
    # Next, separate the data for differen classes, and shuffle it
    data_np, labels_np = np.array(data_set.data[:]), np.array(data_set.targets[:])
    # Prevents the error where one class is never present
    for client_class in class_usage.keys():
        if class_usage[client_class] == 0:
            class_usage[client_class] += 1
    # Create a dict that holds the datapoints for every class
    # I.e. for every class it will hold n separate lists, where n is the number of times the class is used
    data_per_class_dict = {class_data: None for class_data in class_usage.keys()}
    # First, split the data per class
    for class_data in data_per_class_dict.keys():
        # For each class, separate only the values whose label is equal to the current class
        data_per_class_dict[class_data] = data_np[labels_np[:] == class_data]
        # Also random shuffle to make sure this is non-biased
        random.shuffle(data_per_class_dict[class_data])
    
    # Then iterate again, but for each class data, split it into specific number of arrays
    for class_data in data_per_class_dict.keys():
        # The number of arrays to split into is determined by how many clients use this class
        data_per_class_dict[class_data] = np.array_split(data_per_class_dict[class_data], class_usage[class_data])

    # Iterator will be used to keep track of how many times a given class was distributed
    iterator_per_class = {class_data: 0 for class_data in class_usage.keys()}
    # List of Tensor datasets for every client (i.e. the dataset for every client)
    list_dsets = []

    # For every client iterate through its classes
    for client_class_list in client_classes:
        client_data = []   # Will hold the data the client gets
        client_labels = [] # Will hold the labels the client gets
        # Then go through each class
        for client_class in client_class_list:
            # Extend the client data by the data corresponding to the class
            client_data.extend(data_per_class_dict[client_class][iterator_per_class[client_class]])
            # And extend the labels by the labels of that class
            client_labels.extend([client_class] * len(data_per_class_dict[client_class][iterator_per_class[client_class]]))
            # Increment iterator since a class was used up
            iterator_per_class[client_class] += 1
        # Convert both to numpy arrays
        client_data = np.array(client_data)
        client_labels = np.array(client_labels)
        # Add the client data to the list of datasets
        list_dsets.append(data_utils.TensorDataset(torch.tensor(client_data), torch.tensor(client_labels)))

    return list_dsets, client_classes

# Splits the dataset in a non-iid way, excluding server (servers should have all data)
def non_iid_incl_split(data_set, num_servers, num_clients, client_class_split = None):
    # Calculate number of classes
    num_classes = len(data_set.classes)
    # Split the classes among servers
    server_classes = [list(range(num_classes)) for _ in range(num_servers)]

    # Split the classes among clients, if not already
    if client_class_split == None:
        client_classes = [random.sample(list(range(num_classes)), NUM_CLASSES_NONIID) for _ in range(num_clients)]
    else:
        client_classes = client_class_split

    # For each class, calculate how many clients and servers use it
    client_class_usage = {i:0 for i in range(num_classes)}
    server_class_usage = {i:num_servers for i in range(num_classes)}
    # Iterate through client classes to do that
    for client_class_list in client_classes:
        for client_class in client_class_list:
            client_class_usage[client_class] += 1

    # Next, separate the data for different classes, and shuffle it
    data_np, labels_np = np.array(data_set.data[:]), np.array(data_set.targets[:])
    # Prevents the error where one class is never present
    for client_class in client_class_usage.keys():
        if client_class_usage[client_class] == 0:
            client_class_usage[client_class] += 1
    # Create a dict that holds the datapoints for every class
    # I.e. for every class it will hold n separate lists, where n is the number of times the class is used
    data_per_class_dict = {class_data: None for class_data in client_class_usage.keys()}
    server_data_per_class_dict = {class_data: None for class_data in server_class_usage.keys()}

    # First, split the data per class
    for class_data in data_per_class_dict.keys():
        # For each class, separate only the values whose label is equal to the current class
        data_per_class_dict[class_data] = data_np[labels_np[:] == class_data]
        # Also random shuffle to make sure this is non-biased
        random.shuffle(data_per_class_dict[class_data])
    # Then assign the data to the servers
    # Calculate where to split the data
    server_data_percentage = num_servers / (2 * num_clients)
    server_client_split_idx = int(len(data_per_class_dict[0]) - int(server_data_percentage * len(data_per_class_dict[0])))
    # Split the data between server part and client part
    for class_data in data_per_class_dict.keys():
        server_data_per_class_dict[class_data] = np.array_split(data_per_class_dict[class_data][server_client_split_idx:], num_servers)
        data_per_class_dict[class_data] = data_per_class_dict[class_data][:server_client_split_idx]
    # Then iterate again, but for each class data, split it into specific number of arrays
    for class_data in data_per_class_dict.keys():
        # The number of arrays to split into is determined by how many clients use this class
        data_per_class_dict[class_data] = np.array_split(data_per_class_dict[class_data], client_class_usage[class_data])

    # Iterator will be used to keep track of how many times a given class was distributed
    server_iterator_per_class = {class_data: 0 for class_data in server_class_usage.keys()}
    client_iterator_per_class = {class_data: 0 for class_data in client_class_usage.keys()}
    # List of Tensor datasets for every server and every client (i.e. the dataset for every server/client)
    list_dsets = []

    # For every client iterate through its classes
    for server_class_list in server_classes:
        server_data = []   # Will hold the data the server gets
        server_labels = [] # Will hold the labels the server gets
        # Then go through each class
        for server_class in server_class_list:
            # Extend the server data by the data corresponding to the class
            server_data.extend(server_data_per_class_dict[server_class][server_iterator_per_class[server_class]])
            # And extend the labels by the labels of that class
            server_labels.extend([server_class] * len(server_data_per_class_dict[server_class][server_iterator_per_class[server_class]]))
            # Increment iterator since a class was used up
            server_iterator_per_class[server_class] += 1
        # Convert both to numpy arrays
        server_data = np.array(server_data)
        server_labels = np.array(server_labels)
        # Add the server data to the list of datasets
        list_dsets.append(data_utils.TensorDataset(torch.tensor(server_data), torch.tensor(server_labels)))

    # For every client iterate through its classes
    for client_class_list in client_classes:
        client_data = []   # Will hold the data the client gets
        client_labels = [] # Will hold the labels the client gets
        # Then go through each class
        for client_class in client_class_list:
            # Extend the client data by the data corresponding to the class
            client_data.extend(data_per_class_dict[client_class][client_iterator_per_class[client_class]])
            # And extend the labels by the labels of that class
            client_labels.extend([client_class] * len(data_per_class_dict[client_class][client_iterator_per_class[client_class]]))
            # Increment iterator since a class was used up
            client_iterator_per_class[client_class] += 1
        # Convert both to numpy arrays
        client_data = np.array(client_data)
        client_labels = np.array(client_labels)
        # Add the client data to the list of datasets
        list_dsets.append(data_utils.TensorDataset(torch.tensor(client_data), torch.tensor(client_labels)))

    return list_dsets, client_classes

# Split data between servers and clients non-iid
def split_data_non_iid_incl_server(num_servers, num_clients, dataset_name):
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
    # Split the training and validation data
    train_list_dsets, client_class_split = non_iid_incl_split(train_data, num_servers, num_clients)
    valid_list_dsets = iid_split(validation_data, num_servers + num_clients, dataset_name)
    # Return the split datasets
    return train_list_dsets, valid_list_dsets

# Split data betweeen clients non-iid
def split_data_non_iid_excl_server(num_clients, dataset_name, num_classes_non_iid = NUM_CLASSES_NONIID):
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
    # Split the training and validation data
    train_list_dsets, client_class_split = non_iid_excl_split(train_data, num_clients, num_classes_non_iid)
    valid_list_dsets = iid_split(validation_data, num_clients, dataset_name)
    # Return the split datasets
    return train_list_dsets, valid_list_dsets

# Read the sigmf data file
def read_sigmf_data(file_path):
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.complex128)
    return data

# Read the dataset
def read_oracle_dataset(data_dir, window_size=128):
    all_data = []
    all_labels = []
    
    pattern = r'_IQ#(\w+)_'
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.sigmf-data'):
            data_file = os.path.join(data_dir, filename)
            
            # Load data
            data = read_sigmf_data(data_file)
            match = re.search(pattern, filename)
            if match:
                label = int(match.group(1))
            else:
                print(f"Couldn't extract IQ imbalance configuration from filename: {filename}")
                continue
            
            # Split data into windows
            num_windows = len(data) // window_size
            data_windows = np.array_split(data[:num_windows*window_size], num_windows)
            
            all_data.extend(data_windows)
            all_labels.extend([label] * len(data_windows))

    # Create a mapping of unique labels to integers
    unique_labels = sorted(set(all_labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    
    # Convert labels to integers
    int_labels = [label_to_int[label] for label in all_labels]

    return np.array(all_data), np.array(all_labels), np.array(int_labels), label_to_int


# Split the data in iid way
def split_data_oracle_iid(data, labels, labels_int, num_clients):
    # Shuffle the data
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    labels_int = labels_int[indices]
    
    # Split data equally among clients
    data_split = np.array_split(data, num_clients)
    labels_split = np.array_split(labels_int, num_clients)  # Use labels_int instead of labels
    
    return data_split, labels_split

def split_data_oracle_iid_excl_server(num_clients, dataset_dir):
    data, labels, labels_int, label_to_int = read_oracle_dataset(dataset_dir)
    data_split, labels_split = split_data_oracle_iid(data, labels, labels_int, num_clients)
    
    train_list_dsets = [data_utils.TensorDataset(
        torch.from_numpy(np.array(data)).to(torch.complex64),
        torch.tensor(labels).long()
    ) for data, labels in zip(data_split, labels_split)]
    
    # For simplicity, we'll use the same data for validation
    valid_list_dsets = train_list_dsets
    
    return train_list_dsets, valid_list_dsets, label_to_int

def plot_constellation(data, label, filename):
    plt.figure(figsize=(10, 10))
    plt.scatter(data.real, data.imag)
    plt.title(f"Constellation Plot for IQ Imbalance Configuration {label}")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    
def test_oracle_dataset():
    data_dir = '../../../../data/ORACLE_dataset_demodulated/KRI-16IQImbalances-DemodulatedData/'
    num_clients = 5
    
    print('Loading data')
    train_list_dsets, valid_list_dsets, label_to_int = split_data_oracle_iid_excl_server(num_clients, data_dir)
    
    print(f"Number of clients: {num_clients}")
    print(f"Total number of windows: {sum(len(dset) for dset in train_list_dsets)}")
    print(f"Label mapping: {label_to_int}")
    
    # Print some information about the first client's data
    print(f"\nFirst client data length: {len(train_list_dsets[0])}")
    print(f"First client labels: {train_list_dsets[0].tensors[1][:10]}...")  # Show first 10 labels
    
    # Plot constellation for the first window of the first client
    plot_constellation(train_list_dsets[0][0][0], train_list_dsets[0][0][1], "constellation_plot_sample.png")

    print(f"\nNumber of windows in first client's dataset: {len(train_list_dsets[0])}")
    print(f"Shape of first window in first client's dataset: {train_list_dsets[0][0][0].shape}")
    print(f"Label of first window in first client's dataset: {train_list_dsets[0][0][1]}")
    
    # Verify IID distribution
    all_labels = set()
    for dset in train_list_dsets:
        all_labels.update(dset.tensors[1].numpy())
    print(f"\nNumber of unique labels across all clients: {len(all_labels)}")
    print(f"Number of unique labels for first client: {len(set(train_list_dsets[0].tensors[1].numpy()))}")

if __name__ == "__main__":
    test_oracle_dataset()