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

# Create a compiled model class (or learner class) that accepts a created model, optimizer, and loss function, and gives some training/testng functionality
class CompiledModel():
    def __init__(self, model, optimizer, loss_func):
        self.model = model          # Set the class model attribute to passed mode
        self.optimizer = optimizer  # Set optimizer attribute to passed optimizer
        self.loss_func = loss_func  # Set loss function attribute to passed loss function
    
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
        return [self.model.state_dict()[layer] for layer in self.model.state_dict()]

    # Set parameters of the network to some specified parameters of the same network type (i.e. all layers match shapes)
    def set_params(self, param_list):
        # First get the new state dictionary
        new_state_dict = {key: new_weights for key, new_weights in zip(self.model.state_dict(), param_list)}
        self.model.load_state_dict(new_state_dict)

# A class for every server object
class Server_FL():
    def __init__(self, init_model_compiled = None):
        self.list_clients = [] # Initialize list of clients to empty list (Clients of the given server)
        self.list_servers = [] # Initialize list of servers to empty list (Servers connected to the given server)
        self.global_server_model = init_model_compiled # Initialize global model for a given server to some initial compiled model
        self.client_scale_dictionary = dict()          # Initialzie empty dictionary of weights of each client (meaning normalized sizes of each client)
        self.total_clients_weights = 0                 # Total size of the sum of datasets of all clients
    
    # Server should only validate
    def get_data(self, valid_dataset):
        # Get the training and testing data
        x_valid, self.y_valid = valid_dataset.tensors 

        # Reshape the input data to have desired shape (specific for given data, change as needed, here 1 channel for MNIST)
        x_valid = x_valid.view(x_valid.shape[0], 1, x_valid.shape[1], x_valid.shape[2])

        # Normalize the datasets to be between 0 and 1
        self.x_valid = x_valid.float() / 255


    # Method to add a client to the client list for a given server
    def add_client(self, client_object):
        self.list_clients.append(client_object)
        self.client_scale_dictionary[client_object.client_id] = client_object.dset_size
        self.total_clients_weights += client_object.dset_size
    
    # Method to add a server to the server list for a given server
    def add_server(self, server_object):
        self.list_servers.append(server_object)

    # Initialize compiled model (if model not initialized) with specified architecture, optimizer, and loss function (adjustable)
    def init_compiled_model(self):
        # This will always run, but will only initialize the model if the clients model has not been specified
        if self.global_server_model == None:
            global_model_raw = NetBasic() # Note that the architecture can be changed as necessary (also can import model architecture from other file)
            global_loss_function = nn.CrossEntropyLoss() # Again, adjustable here for each client, or can pass a model
            global_optimizer = optim.Adam(global_model_raw.parameters()) # Adjustable here or pass in the compiled model
            # Initialize the compiled model
            self.global_server_model = CompiledModel(model = global_model_raw, optimizer = global_optimizer, loss_func = global_loss_function)
        else:
            pass
    
    # Method to send the global model over to the clients of the server
    def distribute_global_model(self):
        # If the model is not initialized, initialize it to some default model specified in this class
        if self.global_server_model == None:
            self.init_compiled_model()

        # Go through each client and set their models to be equal to the global model
        # To do that, get the current state dictionary, and then distribute the dictionary to all the clients in client list
        global_model_state_dict = self.global_server_model.get_params()
        for client in self.list_clients:
            client.client_model.set_params(global_model_state_dict)

    # Method to aggregate client models into a global model:
    def aggregate_client_models_fed_avg(self):
        scaled_client_weights = [[layer * self.client_scale_dictionary[client.client_id] / self.total_clients_weights \
                                    for layer in client.client_model.get_params()] for client in self.list_clients]

        new_model_parameters = [sum(layers) for layers in zip(*scaled_client_weights)]

        if self.global_server_model == None:
            self.init_compiled_model()
        self.global_server_model.set_params(new_model_parameters)

    # Test the accuracy and loss of a global model
    def validate_global_model(self):
        global_loss, global_accuracy = self.global_server_model.validate(valid_data = self.x_valid, valid_labels = self.y_valid)
        print(f'Global model parameters: Current loss: {global_loss}, Current accuracy: {global_accuracy}')
        return global_loss, global_accuracy

class client_FL():
    def __init__(self, client_id, init_model_client = None, if_adv_client = False):
        self.client_model = init_model_client # Set the model of the client to be 
        self.client_id = client_id            # Each client should have a distinct id (can be just a number, or address, etc.)
        self.if_adv_client = if_adv_client    # If a client is an adversarial client
    # Get client data from training and validation dataloaders
    # Client data should be local to the client
    def get_data(self, train_dataset, valid_dataset):
        # Get the training and testing data
        x_train, self.y_train = train_dataset.tensors 
        x_valid, self.y_valid = valid_dataset.tensors 

        # Reshape the input data to have desired shape (specific for given data, change as needed, here 1 channel for MNIST)
        x_train = x_train.view(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
        x_valid = x_valid.view(x_valid.shape[0], 1, x_valid.shape[1], x_valid.shape[2])

        # Normalize the datasets to be between 0 and 1
        self.x_train = x_train.float() / 255
        self.x_valid = x_valid.float() / 255

        # Also save the dataset size
        self.dset_size = len(self.x_train)

    # Initialize compiled model (if model not initialized) with specified architecture, optimizer, and loss function (adjustable)
    def init_compiled_model(self):
        # This will always run, but will only initialize the model if the clients model has not been specified
        if self.client_model == None:
            local_client_raw_model = NetBasic() # Note that the architecture can be changed as necessary (also can import model architecture from other file)
            local_client_loss_function = nn.CrossEntropyLoss() # Again, adjustable here for each client, or can pass a model
            local_client_optimizer = optim.Adam(local_client_raw_model.parameters()) # Adjustable here or pass in the compiled model
            # Initialize the compiled model
            self.client_model = CompiledModel(model = local_client_raw_model, optimizer = local_client_optimizer, loss_func = local_client_loss_function)
        else:
            pass

    # Train the client model for a specified number of epochs on local data
    def train_client(self, batch_s, n_epoch, show_progress = False):
        if self.if_adv_client == False or (self.if_adv_client and attack == 'None'):
            if self.client_model == None:
                self.init_compiled_model()
                print("Model was not initialized, called init_compiled_model() to initialize")
            # Train the compiled model for a specified number of epochs
            self.client_model.train(train_data = self.x_train, train_labels = self.y_train, train_batch_size = batch_s, n_epoch = n_epoch, show_progress = show_progress)
            return
        else:
            if attack == 'FGSM':
                print('Attacking dataset')
                if self.client_model == None:
                    self.init_compiled_model()
                    print("Model was not initialized, called init_compiled_model() to initialize")
                # Create posioned dataset
                new_adv_data = fast_gradient_method(model_fn = self.client_model.model, x = self.x_train, eps = adv_pow, norm = 2)
                # Train on the posioned dataset
                self.client_model.train(train_data = new_adv_data,  train_labels = self.y_train, train_batch_size = batch_s, n_epoch = n_epoch, show_progress = show_progress)
                return
            elif attack == 'PGD':
                pass
            elif attack == 'Noise':
                pass
    
    # Test the accuracy and loss for a client's model
    def validate_client(self):
        client_loss, client_accuracy = self.client_model.validate(valid_data = self.x_valid, valid_labels = self.y_valid)
        print(f'Client id: {self.client_id}, Current loss: {client_loss}, Current accuracy: {client_accuracy}')
        return client_loss, client_accuracy


# This function splits the data between the servers and clients, currently split is iid
def split_data_uniform_incl_server(num_servers, num_clients):
     # Import data, all the data is in the (N, C, H, W) format (N - data samles, C - channels, H - height, W - width)
    mnist_dir = '~/data/datasets/mnist' # Specify which directory to download MNIST to
    total_data_chunks = num_clients + num_servers
    # Get the data
    train_data = torchvision.datasets.MNIST(root = mnist_dir, train = True, download = True, transform = transforms.ToTensor())
    validation_data  = torchvision.datasets.MNIST(root = mnist_dir, train = False, transform = transforms.ToTensor())
    # Get the shuffled indecies for each client from training and validation datasets
    train_data_split = data_utils.random_split(train_data, [1 / total_data_chunks for i in range(total_data_chunks)], torch.Generator())
    valid_data_split = data_utils.random_split(validation_data, [1 / total_data_chunks for i in range(total_data_chunks)], torch.Generator())
    # Get the datasets for each client and server
    train_list_dsets = [data_utils.TensorDataset(subset.dataset.data[subset.indices], subset.dataset.targets[subset.indices]) for subset in train_data_split]
    valid_list_dsets = [data_utils.TensorDataset(subset.dataset.data[subset.indices], subset.dataset.targets[subset.indices]) for subset in valid_data_split]

    return train_list_dsets, valid_list_dsets

# This function only splits the data between the clients, currently split is iid
def split_data_uniform_excl_server(num_clients):
    # Import data, all the data is in the (N, C, H, W) format (N - data samles, C - channels, H - height, W - width)
    mnist_dir = '~/data/datasets/mnist' # Specify which directory to download MNIST to
    # Get the data
    train_data = torchvision.datasets.MNIST(root = mnist_dir, train = True, download = True, transform = transforms.ToTensor())
    validation_data  = torchvision.datasets.MNIST(root = mnist_dir, train = False, transform = transforms.ToTensor())
    # Get the shuffled indecies for each client from training and validation datasets
    train_data_split = data_utils.random_split(train_data, [1 / num_clients for i in range(num_clients)], torch.Generator())
    valid_data_split = data_utils.random_split(validation_data, [1 / num_clients for i in range(num_clients)], torch.Generator())
    # Get the datasets for each client 
    train_list_dsets = [data_utils.TensorDataset(subset.dataset.data[subset.indices], subset.dataset.targets[subset.indices]) for subset in train_data_split]
    valid_list_dsets = [data_utils.TensorDataset(subset.dataset.data[subset.indices], subset.dataset.targets[subset.indices]) for subset in valid_data_split]

    return train_list_dsets, valid_list_dsets


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