import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method # FGSM
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent # PGD
from cleverhans.torch.attacks.noise import noise # Basic uniform noise perturbation


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
    def init_compiled_model(self, net_arch_class):
        # This will always run, but will only initialize the model if the clients model has not been specified
        if self.global_server_model == None:
            global_model_raw = net_arch_class # Note that the architecture can be changed as necessary (also can import model architecture from other file)
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
    def init_compiled_model(self, net_arch_class):
        # This will always run, but will only initialize the model if the clients model has not been specified
        if self.client_model == None:
            local_client_raw_model = net_arch_class # Note that the architecture can be changed as necessary (also can import model architecture from other file)
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
                # TODO
                pass
            elif attack == 'Noise':
                # TODO
                pass
    
    # Test the accuracy and loss for a client's model
    def validate_client(self):
        client_loss, client_accuracy = self.client_model.validate(valid_data = self.x_valid, valid_labels = self.y_valid)
        print(f'Client id: {self.client_id}, Current loss: {client_loss}, Current accuracy: {client_accuracy}')
        return client_loss, client_accuracy
