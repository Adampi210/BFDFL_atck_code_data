import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.linalg import expm
import random
import hashlib
import numpy as np
import networkx as nx
import csv
import copy
import time 
from collections import deque
from split_data import *
# from sklearn.cluster import KMeans, SpectralClustering
from neural_net_architectures import *

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method # FGSM
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent # PGD
from cleverhans.torch.attacks.noise import noise # Basic uniform noise perturbation


# Create a compiled model class (or learner class) that accepts a created model, optimizer, and loss function, and gives some training/testng functionality
class CompiledModel():
    def __init__(self, model, optimizer, loss_func, device):
        self.model = model          # Set the class model attribute to passed mode
        self.optimizer = optimizer  # Set optimizer attribute to passed optimizer
        self.loss_func = loss_func  # Set loss function attribute to passed loss function
        self.loss_grad = nn.CrossEntropyLoss()  # Set the loss for the gradient calculation
        self.device = device                    # Set the device to be GPU
        self.model = self.model.to(self.device) # Move the model to the device (GPU)

    # Calculate gradients
    def calc_grads(self, train_data, train_labels, train_batch_size, n_epoch, show_progress = False):
        # Setup
        self.model.train()        # Put the model in training mode
        cumulative_gradients = [] # Will hold different averaged gradients (gradient lists)

        # Create a training dataset, move data to GPU
        train_dataset = torch.utils.data.TensorDataset(train_data.to(self.device), train_labels.to(self.device))

        # Create the train dataloader and dataloader iterator
        train_dataloader = DataLoader(dataset = train_dataset, batch_size = train_batch_size, shuffle = True, pin_memory = False)
        dataloader_iterator = iter(train_dataloader)
        
        # Get the batch
        # Iterate for the specified number of epochs (i.e. this is how many diff batches will be used to evaluate the gradients)
        for i in range(n_epoch):
            # Since the number of batches might go over dataset size, I might have to recreate the dataloader with different shuffles
            try:
                x, y = next(dataloader_iterator) # Get next batch
            # If couldn't find next batch, recreate the dataloder
            except StopIteration:
                train_dataloader = DataLoader(dataset = train_dataset, batch_size = train_batch_size, shuffle = True, pin_memory = False)
                dataloader_iterator = iter(train_dataloader)
                x, y = next(dataloader_iterator)
            
            # Calculate the gradients
            self.model.zero_grad()           # Reset the gradients for model 
            output = self.model(x)           # Calculate the outputs for batch inputs
            loss = self.loss_grad(output, y) # Calculate the loss
            loss.backward()                  # Calculate gradients

            # Append the calculated gradients to cumulative gradients array
            cumulative_gradients.append([param.grad.detach().clone() / n_epoch for param in self.model.parameters()])
        
        # Calculate final gradients as sum of cumulative gradients for each layer
        gradients = [sum(x) for x in zip(*cumulative_gradients)]
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

        # Create validation dataset
        valid_dataset = torch.utils.data.TensorDataset(valid_data.to(self.device), valid_labels.to(self.device))

        # Create validation dataloader, pin_memory = True to speed up the transfer to GPU
        valid_dataloader = DataLoader(dataset = valid_dataset, batch_size = int(len(valid_data)), shuffle = True, pin_memory = False)

        # Initialize validation parameters
        total_loss    = 0 # Total loss measured
        total_correct = 0 # Total correct predictions
        total_pts     = len(valid_dataloader.dataset) # Total number of all points

        # Validate by checking batches
        with torch.no_grad():
            for x, y in valid_dataloader:
                output = self.model(x)                                 # Predict the output
                total_loss += self.loss_func(output, y)                # Calculate loss for given datapoint and increment total loss
                total_correct += (torch.argmax(output, dim = 1) == y).sum().item() # Calculate total correct
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

# Class for every client (individual device) object
class client_FL():
    def __init__(self, client_id, init_model_client = None, if_adv_client = False, attack = 'none', adv_pow = 0, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
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
        self.device = device           # Set the device to be GPU
        # TODO pass the device from the full_dec, split among cuda diff
    # Get client data from training and validation dataloaders
    # Client data should be local to the client
    def get_data(self, train_dataset, valid_dataset):
        # Get the training and testing data
        x_train, self.y_train = train_dataset.tensors 
        x_valid, self.y_valid = valid_dataset.tensors 
        self.y_train = self.y_train.to(self.device)
        self.y_valid = self.y_valid.to(self.device)

        # Normalize the datasets to be between 0 and 1
        self.x_train = x_train.float() / 255
        self.x_train = self.x_train.to(self.device)
        self.x_valid = x_valid.float() / 255
        self.x_valid = self.x_valid.to(self.device)

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
            self.client_model = CompiledModel(model = local_client_raw_model_0, optimizer = local_client_optimizer_0, loss_func = nn.CrossEntropyLoss(), device = self.device)
            self.moving_model = CompiledModel(model = local_client_raw_model_1, optimizer = local_client_optimizer_1, loss_func = nn.CrossEntropyLoss(), device = self.device)

            self.moving_model.set_params(self.client_model.get_params())
            # Need the data for the exchange, so if none, initialize
            # Set current model parameters
            if self.global_model_curr is None:
                self.global_model_curr = self.client_model.get_params()
                self.client_model.set_params(self.global_model_curr)
            # Calculate init gradients if not present
            if self.grad_est_curr is None:
                pass
            self.grad_est_curr = self.client_model.calc_grads(train_data = self.x_train, train_labels = self.y_train, train_batch_size = len(self.x_train), n_epoch = 2, show_progress = False)
              
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
                if show_progress:
                    print('Client %d: FGSM - attacking dataset' % self.client_id)
                if model_used == None:
                    print("Model was not initialized!")
                # Create posioned dataset
                new_adv_data = fast_gradient_method(model_fn = model_used.model, x = self.x_train, eps = self.adv_pow, norm = 2)
                # Train on the posioned dataset
                gradients = model_used.calc_grads(train_data = new_adv_data,  train_labels = self.y_train, train_batch_size = batch_s, n_epoch = n_epoch, show_progress = show_progress)
                return gradients
            elif self.attack == 'PGD':
                if show_progress:
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
                if show_progress:
                    print('noise - attacking dataset')
                if model_used == None:
                    print("Model was not initialized!")
                # Create posioned dataset
                new_adv_data = noise(x = self.x_train, eps = self.adv_pow)
                # Train on the posioned dataset
                gradients = model_used.calc_grads(train_data = new_adv_data,  train_labels = self.y_train, train_batch_size = batch_s, n_epoch = n_epoch, show_progress = show_progress)
                return gradients
    
    # Get the models from in-neighbors (adversary never does this)
    def exchange_models(self):
        if self.if_adv_client == False or (self.if_adv_client and self.attack == 'none'):
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
        else:
            return

    # Update rule for SAB alg 
    def aggregate_SAB(self, batch_s, n_epoch, show_progress = False, lr = None): 
        if self.if_adv_client == False or (self.if_adv_client and self.attack == 'none'):
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
        else:
            # Save current params
            curr_model_adv_gradients = self.calc_client_grads(batch_s, n_epoch, show_progress, self.client_model)
            self.client_model.set_grads_and_step(curr_model_adv_gradients, lr) 


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
            if np.array(plain_adj_matrix)[i][j]:
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

def create_graph(adj_matrix):
    # First add neighbors, create graph concurrently
    graph = nx.DiGraph()
    n_clients = len(adj_matrix[0])
    for i in range(n_clients):
        for j in range(n_clients):
            if adj_matrix[i][j]:
                graph.add_edge(i, j)

    # Make the graph
    return graph

def calc_centralities(n_clients, graph_representation):
    # The format is a dict, where the key is client id, and value is a list of centralities in this order:
    # id:[in_deg_centrality, out_deg_centrality, closeness_centrality, betweeness_centrality, eigenvector_centrality]
    centralities_data = {i:[] for i in range(n_clients)}
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

def sort_by_centrality(centrality_data):
    # Read centrality data
    node_centralities = [[int(i)] + x for i, x in centrality_data.items()]
    node_centralities = np.array(node_centralities)

    # Sort centralities
    in_deg_sort = [int(_) for _ in node_centralities[node_centralities[:, 1].argsort()[::-1]][:, 0]]
    out_deg_sort = [int(_) for _ in node_centralities[node_centralities[:, 2].argsort()[::-1]][:, 0]]
    closeness_sort = [int(_) for _ in node_centralities[node_centralities[:, 3].argsort()[::-1]][:, 0]]
    betweeness_sort = [int(_) for _ in node_centralities[node_centralities[:, 4].argsort()[::-1]][:, 0]]
    eigenvector_sort = [int(_) for _ in node_centralities[node_centralities[:, 5].argsort()[::-1]][:, 0]]
    node_sorted_centrality = np.array([in_deg_sort,  out_deg_sort, closeness_sort, betweeness_sort, eigenvector_sort])

    return node_sorted_centrality

def score_cent_dist_manual(cent_weight, n_clients, n_advs, graph_representation, cent_used = -1):
    cent_clients = calc_centralities(n_clients, graph_representation)
    # Scale centralities
    max_cent, min_cent = max(np.array(list(cent_clients.values()))[:, cent_used]), min(np.array(list(cent_clients.values()))[:, cent_used])
    nodes_sorted_by_cent = sort_by_centrality(cent_clients)
    adv_chosen = 0
    adv_nodes = []
    dist_to_advs = {client_id : 0 for client_id in cent_clients.keys()}
    client_scores = {client_id : cent_weight * cent_clients[client_id][cent_used] + (1 - cent_weight) * dist_to_advs[client_id] for client_id in cent_clients.keys()}
    clients_not_chosen = [_ for _ in client_scores.keys()]
    adv_nodes = nodes_sorted_by_cent[cent_used, 0: n_advs]

    return adv_nodes

# Selects the nodes at random
def random_nodes(n_clients, n_advs):
    return list(np.random.choice(np.arange(n_clients), n_advs))

def bfs_cluster(graph, start_node, area):
    visited = set()
    queue = deque([start_node])
    cluster = set()
    
    while queue and len(cluster) < area:
        current_node = queue.popleft()
        if current_node not in visited:
            visited.add(current_node)
            cluster.add(current_node)
            queue.extend(neighbor for neighbor in graph.neighbors(current_node) if neighbor not in visited)
    
    return cluster

def least_overlap_area(n_clients, n_advs, graph_representation):
    # Calculate the area each BFS should cover
    area = n_clients // n_advs
    # Run BFS for each node and store the resulting cluster
    node_clusters = {}
    for node in range(n_clients):
        node_clusters[node] = bfs_cluster(graph_representation, node, area)
    
    # Initialize adversarial nodes and available nodes
    adv_nodes = []
    available_nodes = set(range(n_clients))
    
    while len(adv_nodes) < n_advs:
        min_overlap = float('inf')
        best_node = None
        
        for node in available_nodes:
            # Calculate overlap with already selected adversarial nodes
            overlap = sum(len(node_clusters[node].intersection(node_clusters[adv])) for adv in adv_nodes)
            
            if overlap < min_overlap:
                min_overlap = overlap
                best_node = node

        # Add the best node to adversarial nodes and remove from available nodes
        adv_nodes.append(best_node)
        available_nodes.remove(best_node)
    
    return adv_nodes

# Extension of least overlap area with centrality hopping
# TODO Add modification to make hop distance depend on the graph structure
def MaxSpANFL_w_centrality_hopping(n_clients, n_advs, graph_representation, hop_distance, cent_used = -1):
    # Get initial adversaries
    adv_nodes = least_overlap_area(n_clients, n_advs, graph_representation)
    cent_clients = calc_centralities(n_clients, graph_representation)
    cent_clients = {client: cent[cent_used] for client, cent in cent_clients.items()}
    adv_nodes_hop = []
    # Run centrality-based hopping for each adversarial node
    for i in range(len(adv_nodes)):
        current_node = adv_nodes[i]
        for _ in range(hop_distance):
            # Check for variance (i.e. if not much difference stay)
            neighbors = [i for i in graph_representation.neighbors(current_node)]
            neighbors.append(current_node) 
            # Don't choose duplicate nodes     
            for node in adv_nodes_hop:
                if node in neighbors:
                    neighbors.remove(node)
            # Get most central current node
            current_node = max(neighbors, key = lambda x: cent_clients[x])
        adv_nodes_hop.append(current_node)
    return adv_nodes_hop
    
# Same as above but hop to random neighbor instead most central
def MaxSpANFL_w_random_hopping(n_clients, n_advs, graph_representation, hop_distance, cent_used = -1):
    # Get initial adversaries
    adv_nodes = least_overlap_area(n_clients, n_advs, graph_representation)
    cent_clients = calc_centralities(n_clients, graph_representation)
    cent_clients = {client: cent[cent_used] for client, cent in cent_clients.items()}
    adv_nodes_hop = []
    # Run centrality-based hopping for each adversarial node
    for i in range(len(adv_nodes)):
        current_node = adv_nodes[i]
        for _ in range(hop_distance):
            # Check for variance (i.e. if not much difference stay)
            neighbors = [i for i in graph_representation.neighbors(current_node)]
            neighbors.append(current_node) 
            # Don't choose duplicate nodes     
            for node in adv_nodes_hop:
                if node in neighbors:
                    neighbors.remove(node)
            # Get most central current node
            current_node = np.random.choice(neighbors)
        adv_nodes_hop.append(current_node)
    return adv_nodes_hop

# Calculate average distance between adversarial nodes
def average_distance_between_advs(G, adv_list):
    total_distance = 0
    count = 0
    for i in range(len(adv_list)):
        for j in range(i + 1, len(adv_list)):
            try:
                distance = nx.shortest_path_length(G, source = adv_list[i], target = adv_list[j])
                total_distance += distance
                count += 1
            except nx.NetworkXNoPath:
                continue
    if count == 0:
        return 0
    avg_distance = total_distance / count
    return avg_distance

def add_edges_to_make_strongly_connected(adjacency_matrix):
    graph = nx.DiGraph(adjacency_matrix)
    
    # Identify strongly connected components (SCCs)
    sccs = list(nx.strongly_connected_components(graph))
    if len(sccs) == 1:
        print("The graph is already strongly connected.")
        return adjacency_matrix  # The graph is already strongly connected
    
    # Create a condensed graph of SCCs
    condensed_graph = nx.condensation(graph, sccs)
    
    # Find nodes in the condensed graph with no incoming or outgoing edges
    nodes_with_no_incoming = [node for node in condensed_graph.nodes() if condensed_graph.in_degree(node) == 0]
    nodes_with_no_outgoing = [node for node in condensed_graph.nodes() if condensed_graph.out_degree(node) == 0]
    
    # To make the graph strongly connected, connect the SCCs in a cycle. For simplicity, connect end nodes to start nodes
    for start_node in nodes_with_no_incoming:
        for end_node in nodes_with_no_outgoing:
            # Find representative nodes from the original graph
            start_rep = next(iter(sccs[start_node]))
            end_rep = next(iter(sccs[end_node]))
            
            # Add edge to the original adjacency matrix
            adjacency_matrix[end_rep][start_rep] = 1
    
    return adjacency_matrix

def make_graph_strongly_connected_and_update_matrix(graph_name):
    # Load the adjacency matrix from a file
    dir_graphs = '../../data/full_decentralized/network_topologies/'

    adjacency_matrix = np.loadtxt(dir_graphs + graph_name)  # Assuming CSV format for simplicity

    # Update the adjacency matrix to make the graph strongly connected
    updated_matrix = add_edges_to_make_strongly_connected(adjacency_matrix)
    
    # Optionally, save the updated matrix back to a file or return it
    # np.savetxt("updated_matrix.csv", updated_matrix, delimiter=',')
    return updated_matrix

def extract_strongly_connected_subgraph(adj_matrix, target_nodes):
    # Create a directed graph from the adjacency matrix
    adj_matrix = add_edges_to_make_strongly_connected(adj_matrix)
    G = nx.from_numpy_matrix(adj_matrix, create_using = nx.DiGraph)
    
    # If the graph already matches the target size, return its adjacency matrix
    if G.number_of_nodes() == target_nodes:
        return nx.adjacency_matrix(G)
    
    # While the graph has more nodes than desired, remove nodes with minimal impact
    while len(G.nodes()) > target_nodes:
        # Compute betweenness centrality for all nodes
        centrality = nx.betweenness_centrality(G)
        
        # Find the node with the lowest centrality score
        min_centrality_node = min(centrality, key=centrality.get)
        
        # Remove this node from the graph
        G.remove_node(min_centrality_node)
        new_adj_matrix = nx.adjacency_matrix(G)
        G = nx.from_numpy_matrix(new_adj_matrix, create_using = nx.DiGraph)
        # After removal, the graph may not be strongly connected
        # Find the largest strongly connected component
            
    # After reducing the graph to the target size, return its adjacency matrix
    return nx.adjacency_matrix(G)

if __name__ == "__main__":
    pass

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


# Find real-world graph types for wireless and use them
# For theoretical analysis good idea to start general, then maybe look at specific types of graphs
# Small world graphs
# Here is what to do: 
#   - First, implement that covariance calculation (for choosing k nodes, how different measures choose different nodes)
#   - Then, look at each centrality measure variance
#   - Then, implement the weighting algorithm for the calculation efficiency of cent measure vs the attack potency
#   - Those above should be instanteneous. Next see some potential stuff like time of launching the attack vs its effect (maybe?)


# TODO:
# Implement different choosing strategies, then run the tests
# To implement:


# Compare physical distance of the nodes in both cases (based on centrality distance + based on distance distance)

# Check the clustering algorithm on 25 client case, see if its doing what we think
# Check if separate at r = 0.05 0.1 0.15 for 25 client case (geom)