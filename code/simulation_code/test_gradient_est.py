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
import copy
import csv

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method # FGSM
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent # PGD
from cleverhans.torch.attacks.noise import noise # Basic uniform noise perturbation

from split_data import *
# Device configuration
# Always check first if GPU is avaialble
device_used = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# If CUDA is not avaialbe, print message that CPU will be used
if device_used != torch.device('cuda'):
    print(f'CUDA not available, have to use {device_used}')
# Set hyperparameters
seed = 0 # Seed for PRNGs 
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

iid_type = 'iid'      # 'iid' or 'non_iid'
BATCH_SIZE = 1000     # Batch size while training
N_LOCAL_EPOCHS  = 1   # Number of epochs for local training
N_GLOBAL_EPOCHS = 100 # Number of epochs for global training
N_SERVERS  = 0        # Number of servers
N_CLIENTS  = 10       # Number of clients
dataset_name = 'fmnist' # 'fmnist' or 'cifar10'

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

class CompiledModel():
    def __init__(self, model, optimizer, loss_func):
        self.model = model          # Set the class model attribute to passed mode
        self.optimizer = optimizer  # Set optimizer attribute to passed optimizer
        self.loss_func = loss_func  # Set loss function attribute to passed loss function

    # Calculate gradients
    def calc_grads(self, train_data, train_labels, train_batch_size, n_epoch, show_progress = False):
        self.model.train() # Put the model in training mode
        local_loss = nn.CrossEntropyLoss()
        cumulative_gradients = []
        # Create a dataloader for training
        data   = train_data.clone().detach()    # Represent data as a tensor
        labels = train_labels.clone().detach()  # Represent labels as a tensor
        train_dataset = data_utils.TensorDataset(data, labels) # Create the training dataset
        # Create the train dataloader
        train_dataloader = DataLoader(dataset = train_dataset, batch_size = train_batch_size, shuffle = True)
        dataloader_iterator = iter(train_dataloader)
        gradients_cumulative = []
        grad_test = []
        # Get the batch
        # Iterate for the specified number of epochs (i.e. this is how many diff batches will be used to evaluate the gradients)
        for i in range(n_epoch):
            x, y = next(dataloader_iterator) # Get next batch
            self.model.zero_grad()           # Reset the gradients for model 
            output = self.model(x)           # Calculate the outputs for batch inputs
            loss = local_loss(output, y) # Calculate the loss
            loss.backward()              # Calculate gradients

            # Append the calculated gradients to cumulative gradients array
            gradients_cumulative.append([param.grad.detach().clone() / n_epoch for param in self.model.parameters()])
        
        # Calculate final gradients as sum of cumulative gradients for each layer
        gradients = [sum(x) for x in zip(*gradients_cumulative)]
        return gradients

    # Set model gradients and step
    def set_grads_and_step(self, gradients, step_size = None):
        self.model.train() # Put the model in training mode
        if step_size == None:
            local_optim = optim.Adam(self.model.parameters())
        else:
            local_optim = optim.SGD(self.model.parameters(), step_size)
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


# Set the model used
if dataset_name == 'fmnist':
    NetBasic = FashionMNIST_Classifier
elif dataset_name == 'cifar10':
    NetBasic = CIFAR10_Classifier

# Split the data for the specified number of clients and servers
if iid_type == 'iid':
    train_dset_split, valid_dset_split = split_data_iid_excl_server(N_CLIENTS, dataset_name)
elif iid_type == 'non_iid':
    train_dset_split, valid_dset_split = split_data_non_iid_excl_server(N_CLIENTS, dataset_name)

 
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

        # Add out neigbor
    
    # Add out neighbor
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
                self.grad_est_curr = self.client_model.calc_grads(train_data = self.x_train, train_labels = self.y_train, train_batch_size = 1000, n_epoch = 2, show_progress = False)
              
        else:
            pass

        # Test the accuracy and loss for a client's model
    
    # Calculate validation loss and accuracy 
    def validate_client(self, show_progress = False):
        client_loss, client_accuracy = self.client_model.validate(valid_data = self.x_valid, valid_labels = self.y_valid)
        if show_progress:
            print(f'Client id: {self.client_id}, Current loss: {client_loss}, Current accuracy: {client_accuracy}')
        return client_loss, client_accuracy

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
        
    def aggregate_SAB(self, batch_s, n_epoch, show_progress = False, lr = None):        
        # Update next model parameters
        # Subtract the estimated gradients
        self.moving_model.set_params(self.global_model_next)
        self.moving_model.set_grads_and_step(self.grad_est_curr, lr) 
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
    
    def test_train(self, batch_s, n_epoch, show_progress = False):
        # Calculate current gradients
        grad_est_prev = self.grad_est_curr
        self.grad_est_curr = self.calc_client_grads(batch_s = batch_s, n_epoch = n_epoch, show_progress = False, model_used = self.moving_model)
        # Update the moving model to next model
        self.moving_model.set_grads_and_step(self.grad_est_curr)
        # Calculate new gradients
        self.grad_est_next = self.calc_client_grads(batch_s = batch_s, n_epoch = n_epoch, show_progress = False, model_used = self.moving_model)
        # Calculate update model
        new_grad = [x - y + z for x, y, z in zip(grad_est_prev, self.grad_est_curr, self.grad_est_next)]
        self.client_model.set_grads_and_step(new_grad)
        # This is x(t) now, since client_model has current model
        self.moving_model.set_params(self.client_model.get_params())


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


if __name__ == '__main__':
    acc_list = []
    client_list = [client_FL(i) for i in range(N_CLIENTS)]
    [x.get_data(train_dset_split[i], valid_dset_split[i]) for x, i in zip(client_list, range(N_CLIENTS))]
    [x.init_compiled_model(NetBasic()) for x in client_list]
    [client_list[i].add_neighbor(client_list[i - 1]) for i in range(1, len(client_list))]
    adj_matrix = gen_rand_adj_matrix(N_CLIENTS)
    create_clients_graph(client_list, adj_matrix, None)

    for i in range(100):
        [x.exchange_models() for x in client_list]
        [x.aggregate_SAB(500, 5, False) for x in client_list]
        print(f'Epoch {i}')
        acc_list.append(sum([x.validate_client(False)[-1] for x in client_list]) / len(client_list))
    
    # Create a new figure and a subplot
    fig, ax = plt.subplots()

    # Create line plot
    ax.plot(range(len(acc_list)), acc_list)

    # Set the title and labels
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')

    # Save the figure
    plt.savefig('test_accuracy.png')

    # Show the plot
    plt.show()


