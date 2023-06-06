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

# Number of classes each non-iid client should have
NUM_CLASSES_NONIID = 3

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

# Splits the dataset in a non-iid way, excluding server (servers should have all data)
def non_iid_excl_split(data_set, num_clients, client_class_split = None):
    # Calculate number of classes
    num_classes = len(data_set.classes)
    # Split the classes among clients, if not already
    if client_class_split == None:
        client_classes = [random.sample(list(range(num_classes)), NUM_CLASSES_NONIID) for _ in range(num_clients)]
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
    valid_list_dsets, _ = non_iid_incl_split(validation_data, num_servers, num_clients, client_class_split)
    # Return the split datasets
    return train_list_dsets, valid_list_dsets

# Split data betweeen clients non-iid
def split_data_non_iid_excl_server(num_clients, dataset_name):
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
    train_list_dsets, client_class_split = non_iid_excl_split(train_data, num_clients)
    valid_list_dsets, _ = non_iid_excl_split(validation_data, num_clients, client_class_split) # Cant do that
    # Return the split datasets
    return train_list_dsets, valid_list_dsets
