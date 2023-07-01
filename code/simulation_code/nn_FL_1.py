import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import hashlib
import numpy as np
import networkx as nx
import csv
import copy

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method # FGSM
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent # PGD
from cleverhans.torch.attacks.noise import noise # Basic uniform noise perturbation


# Create a compiled model class (or learner class) that accepts a created model, optimizer, and loss function, and gives some training/testng functionality
class CompiledModel():
    def __init__(self, model, optimizer, loss_func):
        self.model = model          # Set the class model attribute to passed mode
        self.optimizer = optimizer  # Set optimizer attribute to passed optimizer
        self.loss_func = loss_func  # Set loss function attribute to passed loss function
        # self.local_z_model = copy.deepcopy(self.model)  # Create a copy of the model that will store the z model for push-sum training
        # self.local_z_model_loss = nn.CrossEntropyLoss() # Loss for local z model
        # self.local_z_model_optim = optim.Adam(self.local_z_model.parameters())
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

    # Calculate gradients
    def calc_grads(self, train_data, train_labels, train_batch_size, n_epoch, show_progress = False):
        self.model.train() # Put the model in training mode
        local_loss = nn.CrossEntropyLoss()
        local_optim = optim.Adam(self.model.parameters())
        # Create a dataloader for training
        data   = train_data.clone().detach()    # Represent data as a tensor
        labels = train_labels.clone().detach()  # Represent labels as a tensor
        train_dataset = data_utils.TensorDataset(data, labels) # Create the training dataset
        # Create the train dataloader
        train_dataloader = DataLoader(dataset = train_dataset, batch_size = train_batch_size, shuffle = True)
        # Get the batch
        x, y = next(iter(train_dataloader))
        local_optim.zero_grad() # Reset the gradients for optimizer
        self.model.zero_grad()     # Reset the gradients for model 
        output = self.model(x)     # Calculate the outputs for batch inputs
        loss = local_loss(output, y) # Calculate the loss
        loss.backward()                  # Calculate gradients
        # Get the gradients
        gradients = [param.grad.detach().clone() for param in self.model.parameters()]

        return gradients

    def set_grads_and_step(self, gradients, step_size):
        self.model.train() # Put the model in training mode
        local_optim = optim.SGD(self.model.parameters(), lr = step_size)
        # Reset the gradients for optimizer and model
        local_optim.zero_grad() 
        self.model.zero_grad()   
        # Set new gradients
        for model_param, grad_new in zip(self.model.parameters(), gradients):
            model_param.grad = grad_new.clone()
        # Update model
        local_optim.step()
        
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
        return list(self.model.parameters())

    # Set parameters of the network to some specified parameters of the same network type (i.e. all layers match shapes)
    def set_params(self, param_list):
        # Iterate over new parameters and copy into current ones
        with torch.no_grad():
            for param_curr, param_new in zip(self.model.parameters(), param_list):
                param_curr.data.copy_(param_new)
    
    '''
    # Training method in the push-sum algorithm (since it's different)
    def train_push_sum(self, train_data, train_labels, train_batch_size, n_epoch, show_progress = False, local_model_z_params = None):
        self.model.train() # Put the model in training mode
        # Set parameters of the local z model
        new_z_state_dict = {key: new_weights for key, new_weights in zip(self.local_z_model.state_dict(), local_model_z_params)}
        self.local_z_model.load_state_dict(new_z_state_dict)
        self.local_z_model.train() # Put the local z model in training mode
        
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
                self.optimizer.zero_grad()                # Reset the gradients
                self.local_z_model_optim.zero_grad()      # Reset the gradients of the local z model
                output = self.local_z_model(x)            # Calculate the outputs for batch inputs
                loss = self.local_z_model_loss(output, y) # Calculate the loss for local z model
                loss.backward()                           # Calculate gradients
                # Copy the gradients
                for model_param, z_param in zip(self.model.parameters(), self.local_z_model.parameters()):
                    if z_param.grad is not None:
                        model_param.grad = z_param.grad.clone()
                self.optimizer.step()            # Update the parameters of original model
                self.local_z_model_optim.step()  # Update local z model parameters
                # Print batch loss 
                if show_progress: 
                    print(f'Epoch: {epoch}, Batch loss: {loss}')
    '''
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
            global_optimizer = optim.Adam(global_model_raw.parameters())
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
    def __init__(self, client_id, init_model_client = None, if_adv_client = False, attack = 'none', adv_pow = 0):
        self.client_model = init_model_client # Set the model of the client to be 
        self.moving_model = None
        self.temp_model = None
        self.client_id = client_id            # Each client should have a distinct id (can be just a number, or address, etc.)
        self.if_adv_client = if_adv_client    # If a client is an adversarial client
        self.attack = attack
        self.adv_pow = adv_pow
        self.eps_iter = 0
        self.nb_iter = 0
        self.out_neighbors = {}
        self.in_neighbors = {}
        self.global_model_curr = None  # Corresp to x_i(k)
        self.grad_est_curr = None      # Corresp to y_i(k)
        self.global_model_next = None  # Corresp to x_i(k + 1)
        self.grad_est_next = None      # Corresp to y_i(k + 1)
    # Get client data from training and validation dataloaders
    # Client data should be local to the client
    def get_data(self, train_dataset, valid_dataset):
        # Get the training and testing data
        x_train, self.y_train = train_dataset.tensors 
        x_valid, self.y_valid = valid_dataset.tensors 

        # Normalize the datasets to be between 0 and 1
        self.x_train = x_train.float() / 255
        self.x_valid = x_valid.float() / 255

        # Also save the dataset size
        self.dset_size = len(self.x_train)

    # Initialize compiled model (if model not initialized) with specified architecture, optimizer, and loss function (adjustable)
    def init_compiled_model(self, net_arch_class):
        # This will always run, but will only initialize the model if the clients model has not been specified
        if self.client_model == None:
            local_client_raw_model_0 = copy.deepcopy(net_arch_class) # Note that the architecture can be changed as necessary (also can import model architecture from other file)
            local_client_raw_model_1 = copy.deepcopy(net_arch_class) # Note that the architecture can be changed as necessary
            # local_client_loss_function_0 = nn.CrossEntropyLoss() # Again, adjustable here for each client, or can pass a model
            # local_client_loss_function_1 = nn.CrossEntropyLoss()
            local_client_optimizer_0 = optim.Adam(local_client_raw_model_0.parameters())
            local_client_optimizer_1 = optim.Adam(local_client_raw_model_1.parameters())
            # Initialize the compiled model
            self.client_model = CompiledModel(model = local_client_raw_model_0, optimizer = local_client_optimizer_0, loss_func = nn.CrossEntropyLoss())
            self.moving_model = CompiledModel(model = local_client_raw_model_1, optimizer = local_client_optimizer_1, loss_func = nn.CrossEntropyLoss())

            self.moving_model.set_params(self.client_model.get_params())
            # Need the data for the exchange, so if none, initialize
            # Set current model parameters
            if self.global_model_curr is None:
                self.global_model_curr = self.client_model.get_params()
                self.client_model.set_params(self.global_model_curr)
            # Calculate init gradients if not present
            if self.grad_est_curr is None:
                self.grad_est_curr = self.calc_client_grads(1000, 1 , False, self.client_model)
              
        else:
            pass

    # Train the client model for a specified number of epochs on local data
    def train_client(self, batch_s, n_epoch, show_progress = False):
        if self.if_adv_client == False or (self.if_adv_client and self.attack == 'none'):
            if self.client_model == None:
                print("Model was not initialized!")
            # Train the compiled model for a specified number of epochs
            self.client_model.train(train_data = self.x_train, train_labels = self.y_train, train_batch_size = batch_s, n_epoch = n_epoch, show_progress = show_progress)
            return
        else:
            if self.attack == 'FGSM':
                print('FGSM - attacking dataset')
                if self.client_model == None:
                    print("Model was not initialized!")
                # Create posioned dataset
                new_adv_data = fast_gradient_method(model_fn = self.client_model.model, x = self.x_train, eps = self.adv_pow, norm = 2)
                # Train on the posioned dataset
                self.client_model.train(train_data = new_adv_data,  train_labels = self.y_train, train_batch_size = batch_s, n_epoch = n_epoch, show_progress = show_progress)
                return
            elif self.attack == 'PGD':
                print('PGD - attacking dataset')
                if self.client_model == None:
                    print("Model was not initialized!")
                # Create posioned dataset
                new_adv_data = projected_gradient_descent(model_fn = self.client_model.model, x = self.x_train, 
                    eps = self.adv_pow, eps_iter = self.eps_iter, nb_iter = self.nb_iter, norm = 2)
                # Train on the posioned dataset
                self.client_model.train(train_data = new_adv_data,  train_labels = self.y_train, train_batch_size = batch_s, n_epoch = n_epoch, show_progress = show_progress)
                return
            elif self.attack == 'noise':
                print('noise - attacking dataset')
                if self.client_model == None:
                    print("Model was not initialized!")
                # Create posioned dataset
                new_adv_data = noise(x = self.x_train, eps = self.adv_pow)
                # Train on the posioned dataset
                self.client_model.train(train_data = new_adv_data,  train_labels = self.y_train, train_batch_size = batch_s, n_epoch = n_epoch, show_progress = show_progress)
                return
    
    # Train the client model for a specified number of epochs on local data using the push sum algorithm
    def train_client_push_sum(self, batch_s, n_epoch, show_progress = False, local_model_z_params = None):
        if self.if_adv_client == False or (self.if_adv_client and self.attack == 'none'):
            if self.client_model == None:
                print("Model was not initialized!")
            # Train the compiled model for a specified number of epochs
            self.client_model.train_push_sum(train_data = self.x_train, train_labels = self.y_train, train_batch_size = batch_s, n_epoch = n_epoch, show_progress = show_progress, local_model_z_params = local_model_z_params)
            return
        else:
            if self.attack == 'FGSM':
                print('FGSM - attacking dataset')
                if self.client_model == None:
                    print("Model was not initialized!")
                # Create posioned dataset
                new_adv_data = fast_gradient_method(model_fn = self.client_model.model, x = self.x_train, eps = self.adv_pow, norm = 2)
                # Train on the posioned dataset
                self.client_model.train_push_sum(train_data = new_adv_data,  train_labels = self.y_train, train_batch_size = batch_s, n_epoch = n_epoch, show_progress = show_progress, local_model_z_params = local_model_z_params)
                return
            elif self.attack == 'PGD':
                print('PGD - attacking dataset')
                if self.client_model == None:
                    print("Model was not initialized!")
                # Create posioned dataset
                new_adv_data = projected_gradient_descent(model_fn = self.client_model.model, x = self.x_train, 
                    eps = self.adv_pow, eps_iter = self.eps_iter, nb_iter = self.nb_iter, norm = 2)
                # Train on the posioned dataset
                self.client_model.train_push_sum(train_data = new_adv_data,  train_labels = self.y_train, train_batch_size = batch_s, n_epoch = n_epoch, show_progress = show_progress, local_model_z_params = local_model_z_params)
                return
            elif self.attack == 'noise':
                print('noise - attacking dataset')
                if self.client_model == None:
                    print("Model was not initialized!")
                # Create posioned dataset
                new_adv_data = noise(x = self.x_train, eps = self.adv_pow)
                # Train on the posioned dataset
                self.client_model.train_push_sum(train_data = new_adv_data,  train_labels = self.y_train, train_batch_size = batch_s, n_epoch = n_epoch, show_progress = show_progress, local_model_z_params = local_model_z_params)
                return
    
    # Calculate client gradients
    def calc_client_grads(self, batch_s, n_epoch, show_progress = False, model_used = None):
        if self.if_adv_client == False or (self.if_adv_client and self.attack == 'none'):
            if model_used == None:
                print("Model was not initialized!")
            # Train the compiled model for a specified number of epochs
            gradients = model_used.calc_grads(train_data = self.x_train, train_labels = self.y_train, train_batch_size = batch_s, n_epoch = n_epoch, show_progress = show_progress)
            return gradients
        else:
            if self.attack == 'FGSM':
                print('FGSM - attacking dataset')
                if model_used == None:
                    print("Model was not initialized!")
                # Create posioned dataset
                new_adv_data = fast_gradient_method(model_fn = model_used.model, x = self.x_train, eps = self.adv_pow, norm = 2)
                # Train on the posioned dataset
                gradients = model_used.calc_grads(train_data = new_adv_data,  train_labels = self.y_train, train_batch_size = batch_s, n_epoch = n_epoch, show_progress = show_progress)
                return gradients
            elif self.attack == 'PGD':
                print('PGD - attacking dataset')
                if model_used == None:
                    print("Model was not initialized!")
                # Create posioned dataset
                new_adv_data = projected_gradient_descent(model_fn = model_used.model, x = self.x_train, 
                    eps = self.adv_pow, eps_iter = self.eps_iter, nb_iter = self.nb_iter, norm = 2)
                # Train on the posioned dataset
                gradients = model_used.calc_grads(train_data = new_adv_data,  train_labels = self.y_train, train_batch_size = batch_s, n_epoch = n_epoch, show_progress = show_progress)
                return gradients
            elif self.attack == 'noise':
                print('noise - attacking dataset')
                if model_used == None:
                    print("Model was not initialized!")
                # Create posioned dataset
                new_adv_data = noise(x = self.x_train, eps = self.adv_pow)
                # Train on the posioned dataset
                gradients = model_used.calc_grads(train_data = new_adv_data,  train_labels = self.y_train, train_batch_size = batch_s, n_epoch = n_epoch, show_progress = show_progress)
                return gradients
    
    # Test the accuracy and loss for a client's model
    def validate_client(self, show_progress = False):
        client_loss, client_accuracy = self.client_model.validate(valid_data = self.x_valid, valid_labels = self.y_valid)
        if show_progress:
            print(f'Client id: {self.client_id}, Current loss: {client_loss}, Current accuracy: {client_accuracy}')
        return client_loss, client_accuracy

    # Add out neigbor
    def add_out_neighbor(self, out_neighbor_object):
        self.out_neighbors[out_neighbor_object.client_id] = out_neighbor_object
        out_neighbor_object.in_neighbors[self.client_id] = self

    # Add in neighbor
    def add_in_neighbor(self, in_neighbor_object):
        self.in_neighbors[in_neighbor_object.client_id] = in_neighbor_object
        in_neighbor_object.out_neighbors[self.client_id] = self

    # Add neigbor two ways
    def add_neighbor(self, neighbor_object):
        self.out_neighbors[neighbor_object.client_id] = neighbor_object
        self.in_neighbors[neighbor_object.client_id] = neighbor_object
        neighbor_object.out_neighbors[self.client_id] = self
        neighbor_object.in_neighbors[self.client_id] = self
    
    # Get the models from in-neighbors
    def exchange_models(self):
        # Get current parameters
        old_model_parameters = self.client_model.get_params()
        # Calculate scaling weights a_ij, b_ij for the model aggregation
        # Since A is row-stochastic, the scaling is by the in-degree of the current node
        # Since B is column-stochastic, the scaling is by the out-degree of each in-neighbor
        # Also always include itself
        global_model_param_weights = [1 / (len(self.in_neighbors) + 1) for _ in range(len(self.in_neighbors) + 1)]
        gradient_estimate_weights = [1 / (len(neighbor.out_neighbors) + 1) for neighbor in self.in_neighbors.values()]
        gradient_estimate_weights.append(1 / (len(self.out_neighbors) + 1))
        # Create parameter lists for global model and gradient estimate
        global_model_list = [neighbor.client_model.get_params() for neighbor in self.in_neighbors.values()]
        global_model_list.append(old_model_parameters)
        gradient_estimate_list = [neighbor.grad_est_curr for neighbor in self.in_neighbors.values()]
        gradient_estimate_list.append(self.grad_est_curr)
        # Aggregate models
        # Get the parameter lists
        global_model_next_params = copy.deepcopy(global_model_list[0])
        gradient_estimate_next_params = copy.deepcopy(gradient_estimate_list[0])
        # Reset the parameters to 0 and aggregate the neighbor parameters
        for param_new in global_model_next_params:
            param_new.data *= 0
        # Aggregate the neighbors
        for neighbor_model, neighbor_weight in zip(global_model_list, global_model_param_weights):
            for param_new, param_neighbor in zip(global_model_next_params, neighbor_model):
                param_new.data += neighbor_weight * param_neighbor.data
        # Reset the gradients to 0
        for grad_new in gradient_estimate_next_params:
            grad_new.data *= 0
        # Aggregate the neighbors
        for neighbor_grad, neighbor_weight in zip(gradient_estimate_list, gradient_estimate_weights):
            for grad_new, grad_neighbor in zip(gradient_estimate_next_params, neighbor_grad):
                grad_new.data += neighbor_weight * grad_neighbor.data

        # Calculate the final value as the sum of scaled layers
        self.global_model_next = global_model_next_params
        self.grad_est_next = gradient_estimate_next_params
        
    def aggregate_SAB(self, batch_s, n_epoch, show_progress = False, lr = 0.01):        
        # Update next model parameters
        # Subtract the estimated gradients
        self.moving_model.set_params(self.global_model_next)
        self.moving_model.set_grads_and_step(self.grad_est_curr, step_size = lr) 
        #self.moving_model.set_grads_and_step(self.calc_client_grads(batch_s, n_epoch, show_progress, self.client_model), step_size = lr)
        # for params_weighted, grads_est in zip(self.global_model_next, self.grad_est_curr):
        # for params_weighted, grads_est in zip(self.global_model_next, self.calc_client_grads(batch_s, n_epoch, show_progress, self.client_model)):
        #    params_weighted.data.sub_(lr * grads_est)
        # self.global_model_next = [x - lr * y for x, y in zip(self.global_model_next, self.calc_client_grads(batch_s, n_epoch, show_progress, self.client_model))] # Testing
        # The moving model gets the next parameters
        # self.moving_model.set_params(self.global_model_next) # = x_i(k + 1)
        self.global_model_next = self.moving_model.get_params()
        # Calculate new estimated gradients as follows
        est_grad_diff = [next_grad - curr_grad for next_grad, curr_grad in zip(self.calc_client_grads(batch_s, n_epoch, show_progress, self.moving_model), self.calc_client_grads(batch_s, n_epoch, show_progress, self.client_model))]

        for grad_weighted, grad_next_model in zip(self.grad_est_next, est_grad_diff):
            grad_weighted.data.add_(grad_next_model)
        
        # Update current values
        self.global_model_curr = self.global_model_next
        self.grad_est_curr = self.grad_est_next
        # Set the parameters to current version
        self.client_model.set_params(self.global_model_curr)
    
    '''
    # I need 2 methods for that, first to exchange the parameters for consensus, then for training and setting parameters
    # Exchange values using a subgradient-push method from https://ieeexplore.ieee.org/abstract/document/6930814
    def exchange_values_push_sum(self):
        old_model_parameters = self.client_model.get_params()
        # Calculate w_i(t + 1), need to add 1 to account for itself
        # Calculate for current node
        scaled_curr_weights = [layer / (len(self.out_neighbors) + 1) for layer in old_model_parameters]
        # Calculate for in-neighbors
        scaled_neighbor_weights = [[layer / (len(neighbor.out_neighbors) + 1) for layer in neighbor.client_model.get_params()] \
            for neighbor in self.in_neighbors.values()]
        # Combine
        scaled_neighbor_weights.append(scaled_curr_weights)
        # Calculate the final value as the sum of scaled layers
        self.w_t_next = [sum(layers) for layers in zip(*scaled_neighbor_weights)]

        # Calculate y_i(t + 1) - the scaling value
        self.y_t_next = sum([neighbor.y_t / (len(neighbor.out_neighbors) + 1) for neighbor in self.in_neighbors.values()]) + self.y_t / (len(self.out_neighbors) + 1) 
        # Finally, z_i(t + 1) is the new model summed from the clients scaled by y_i(t + 1)
        self.z_t_next = [layer / self.y_t_next for layer in self.w_t_next]

    # Train and aggregate using the push sum method, where first parameters are set to w_i(t + 1), and then gradients from z added
    def train_and_aggregate_push_sum(self, batch_s, n_epoch, show_progress = False):
        # Set new parameters to w_i(t + 1)
        self.client_model.set_params(self.w_t_next)
        self.y_t = self.y_t_next
        # Train using push_sum algorithm (calculate gradients on the z model)
        self.train_client_push_sum(batch_s = batch_s, n_epoch = n_epoch, show_progress = show_progress, local_model_z_params = self.z_t_next)

    # Version above for explicit adversarial attack where adversaries don't use aggregated models (just train on themselves)
    def train_and_aggregate_push_sum_adv(self, batch_s, n_epoch, show_progress = False):
        if self.if_adv_client:
            self.train_client()
        else:
            self.train_and_aggregate_push_sum(batch_s = batch_s, n_epoch = n_epoch, show_progress = show_progress)

    # train for testing
    def train_test(self, batch_s, n_epoch, show_progress = False):
        self.client_model.set_params(self.w_t_next)
        self.train_client(batch_s = batch_s, n_epoch = n_epoch, show_progress = show_progress)
    '''
# Hash a numpy array
def hash_np_arr(np_arr):
    # Convert to a string of bytes
    byte_str = np_arr.tobytes()
    # Create the hash
    hash_arr = hashlib.sha1(byte_str)
    # Create hex representation
    hex_hash_arr = hash_arr.hexdigest()
    return hex_hash_arr

# Generate random adjacency matrix for a random graph
# TODO aggregation changes the weights, see how that affects centralities
def gen_rand_adj_matrix(n_clients):
    # The created graph must be fully connected
    is_strongly_connected = False
    while is_strongly_connected == False:
        # Choose random number of edges
        num_edges = random.randint(n_clients, n_clients ** 2)
        # Choose the random clients
        chosen_edges = random.choices(range(0, n_clients ** 2), k = num_edges)
        # Convert to matrix coords
        for i, edge in enumerate(chosen_edges):
            chosen_edges[i] = (edge // n_clients, edge % n_clients)

        plain_adj_matrix = np.zeros((n_clients, n_clients))
        # Create the matrix (binary)
        for coord in chosen_edges:
            i, j = coord[0], coord[1]
            if i != j:
                plain_adj_matrix[i][j] = 1
        # Create the graph, make sure it's strongly connected
        graph = nx.from_numpy_matrix(plain_adj_matrix, create_using = nx.DiGraph)
        is_strongly_connected = nx.is_strongly_connected(graph)
    # Only return fully connected graph
    return plain_adj_matrix

def create_clients_graph(node_list, plain_adj_matrix, aggregation_mechanism):
    # First add neighbors, create graph concurrently
    graph = nx.DiGraph()
    for device in node_list:
        i = device.client_id
        for j in range(len(node_list)):
            if plain_adj_matrix[i][j]:
                device.add_out_neighbor(node_list[j])
                graph.add_edge(i, j)
    # Then create the adjacency matrix
    adj_matrix = np.zeros((len(node_list), len(node_list)))
    # for device in node_list:
        # Connection goes from i to j, not 0 if that happens
        # This is actually specific to the aggregation mechanism used
        # This aggregation mechanism depends on the size of datasets of in neighbors
        #if aggregation_mechanism == 'base':
        #    i = device.client_id
        #    i_neigbor_sum = sum([neighbor.dset_size for neighbor in device.in_neighbors.values()])
        
        # for j in device.in_neighbors.keys():
        #    adj_matrix[j][i] = device.in_neighbors[j].dset_size / i_neigbor_sum
        
    # ALso make the graph
    return plain_adj_matrix, graph

def calc_centralities(node_list, graph_representation):
    # The format is a dict, where the key is client id, and value is a list of centralities in this order:
    # id:[in_deg_centrality, out_deg_centrality, closeness_centrality, betweeness_centrality, eigenvector_centrality]
    centralities_data = {node.client_id:[] for node in node_list}
    # Calculate centralities
    in_deg_centrality = nx.in_degree_centrality(graph_representation)
    out_deg_centrality = nx.out_degree_centrality(graph_representation)
    closeness_centrality = nx.closeness_centrality(graph_representation)
    betweeness_centrality = nx.betweenness_centrality(graph_representation)
    eigenvector_centrality = nx.eigenvector_centrality(graph_representation.reverse(), max_iter = 1000) # Reverse for out-edges eigenvector centrality
    # Assign centralities
    for node in centralities_data.keys():
        centralities_data[node].extend([in_deg_centrality[node], out_deg_centrality[node], closeness_centrality[node], betweeness_centrality[node], eigenvector_centrality[node]])
        
    return centralities_data

def sort_by_centrality(centrality_data_file):
    # Read centrality data
    node_centralities = []
    with open(centrality_data_file, 'r') as centrality_data:
        reader = csv.reader(centrality_data)
        for row in reader:
            row_float = [float(j) if i > 0 else int(j) for i, j in enumerate(row)]
            row_float[0] = int(row_float[0])
            node_centralities.append(row_float)
    node_centralities = np.array(node_centralities)
    # Sort centralities
    in_deg_sort = node_centralities[node_centralities[:, 1].argsort()[::-1]][:, 0]
    out_deg_sort = node_centralities[node_centralities[:, 2].argsort()[::-1]][:, 0]
    closeness_sort = node_centralities[node_centralities[:, 3].argsort()[::-1]][:, 0]
    betweeness_sort = node_centralities[node_centralities[:, 4].argsort()[::-1]][:, 0]
    eigenvector_sort = node_centralities[node_centralities[:, 5].argsort()[::-1]][:, 0]

    node_sorted_centrality = np.array([in_deg_sort ,  out_deg_sort, closeness_sort, betweeness_sort, eigenvector_sort])
    return node_sorted_centrality

if __name__ == "__main__":
    a = sort_by_centrality('../../data/full_decentralized/fmnist/atk_none_advs_1_adv_pow_1_clients_10_atk_time_0_arch_star_seed_0_iid_type_iid/centrality_clients_fb5507dc8f227c762865a6f14daa2358a0003fff.csv')
    print(a)

# Increase number of epochs 3-5 times the current one, check different learning rates
# Decrease step size significantly, increase the sizes of minibatchs
# Use strongly convex loss (different loss functions)
# For lower-bound the optimality gap: 
# Assume high t = 0 optimality gap, upperbound the difference between w_i(t+1) and w_i(t)
# Assume w_i(0) - w* > Beta, find E||w_i(t + 1) - w_i(t)|| <= some func. Then traig + Jensen inequality
# Relate the centralities of nodes to the attacks - try to relate centralities to the optimality gap and attackers
# Simulations 10, 30, 50 clients, 5-10/20% adversaries
# If time left: 2 cases if adversaries cooperate/dont cooperate
# Find the signal to noise ratio that guarantees the adversarial clients to not converge 