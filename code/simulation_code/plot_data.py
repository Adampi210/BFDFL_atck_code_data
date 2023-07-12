import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import ast
from nn_FL_1 import *
import networkx as nx
import re

# Read parameters
seed = 0 # Seed for PRNGs 

def plot_acc_dec_data(csv_file_name):
    iterator = -1
    epochs, acc_data = [[], [], [], [], []], [[], [], [], [], []]
    adv_list = []
    with open(csv_file_name, 'r') as file_data:
        reader = csv.reader(file_data)
        for row in reader:
            if any([i in row for i in ('in_deg_centrality', 'out_deg_centrality', 'closeness_centrality', 'betweeness_centrality', 'eigenvector_centrality')]):
                iterator += 1
                # adv_list.append(ast.literal_eval(row[1]))
            else:
                acc_data_clients = ast.literal_eval(row[1])
                epochs[iterator].append(int(row[0]) + 1)
                acc_data[iterator].append(acc_data_clients)
    # Create figure and axis
    for j, centrality in enumerate(('in_deg_centrality', 'out_deg_centrality', 'closeness_centrality', 'betweeness_centrality', 'eigenvector_centrality')):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        client_values = list(range(len(acc_data[0][0])))
        # Plot the data
        for i in range(len(acc_data[0][0])):
            ax.plot3D(epochs[j], np.array(acc_data[j])[:, i], client_values[i])
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_zlabel('Client')
        ax.set_title('Global Accuracy after each aggregation', fontsize = 20)
        ax.axvline(x = 25, color = 'r', label = 'Attack Begins', lw=2.5)
        #ax.view_init(elev=60, azim=290)  # Adjust the angles to rotate the plot
        ax.grid(True)
        plt.savefig(csv_file_name[:-4] + '_' + centrality + '.png')

def plot_average_clients(csv_file_name):
    iterator = -1
    epochs, acc_data = [[], [], [], [], []], [[], [], [], [], []]
    adv_list = []
    with open(csv_file_name, 'r') as file_data:
        reader = csv.reader(file_data)
        for row in reader:
            if any([i in row for i in ('in_deg_centrality', 'out_deg_centrality', 'closeness_centrality', 'betweeness_centrality', 'eigenvector_centrality')]):
                iterator += 1
                # adv_list.append(ast.literal_eval(row[1]))
            else:
                acc_data_clients = ast.literal_eval(row[1])
                epochs[iterator].append(int(row[0]) + 1)
                acc_data[iterator].append(sum(acc_data_clients) / len(acc_data_clients))
    # Create figure and axis
    for j, centrality in enumerate(('in_deg_centrality', 'out_deg_centrality', 'closeness_centrality', 'betweeness_centrality', 'eigenvector_centrality')):
        fig, ax = plt.subplots()

        # Plot the data
        for i in range(len(acc_data[0])):
            ax.plot(epochs[j], np.array(acc_data[j]))

        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Global Accuracy after each aggregation', fontsize = 20)
        ax.axvline(x = 25, color = 'r', label = 'Attack Begins', lw=2.5)
        #ax.view_init(elev=60, azim=290)  # Adjust the angles to rotate the plot
        ax.grid(True)
        plt.savefig(csv_file_name[:-4] + '_averaged_' + centrality + '.png')

def calc_diff_attack(csv_original_name, csv_attacked_name):
    total_diff = 0
    with open(csv_original_name, 'r') as original_data:
        reader_orig = csv.reader(original_data)
        with open(csv_attacked_name, 'r') as attacked_data:
            reader_attck = csv.reader(attacked_data)
            i = 0
            for row_orig, row_attck in zip(reader_orig, reader_attck):
                if i == 0:
                    attacked_nodes = np.fromstring(row_attck[1])
                    attacked_nodes = [int(_) for _ in attacked_nodes]
                elif i >= 25:
                    acc_orig = ast.literal_eval(row_orig[1])
                    acc_orig = sum(acc_orig) / len(acc_orig)
                    acc_attck = ast.literal_eval(row_attck[1])
                    acc_honest = [_ for i, _ in enumerate(acc_attck) if i not in attacked_nodes]
                    acc_honest = sum(acc_honest) / len(acc_honest)
                    total_diff += acc_orig - acc_honest
                i += 1

        return total_diff

def plot_acc_diff(dataset_name = 'fmnist'):
    dir_networks = '/root/programming/Purdue-Research-Programs-Notes/data/full_decentralized/network_topologies'
    dir_data = '/root/programming/Purdue-Research-Programs-Notes/data/full_decentralized/%s/' % dataset_name
    centralities = ('none', 'in_deg_centrality', 'out_deg_centrality', 'closeness_centrality', 'betweeness_centrality', 'eigenvector_centrality')
    network_topology = 'random_graph_c_10_p_05_seed_0.txt'
     
    dist_filenames = set()
    file_network_toplogy = os.path.join(dir_networks, network_topology)
    adj_matrix = np.loadtxt(file_network_toplogy)
    hash_adj_matrix = hash_np_arr(adj_matrix)
    data_dir_name = dir_data + str(hash_adj_matrix) + '/' 
    # First get distinct plot types
    for root, dirs, files in os.walk(data_dir_name):
        for filename in files:
            if "acc_" in filename:
                dist_filenames.add(filename)
    for filename_data in dist_filenames:
        acc_data = {cent:None for cent in centralities}
        for root, dirs, files in os.walk(data_dir_name):
            if not any([cent in root for cent in centralities]):
                continue
            for filename in files:
                if filename == filename_data:
                    acc_data[centralities[[cent in root for cent in centralities].index(True)]] = []
                    with open(root + '/' + filename, 'r') as acc_data_file:
                        reader = csv.reader(acc_data_file)
                        i = 0
                        for row in reader:
                            if i == 0:
                                attacked_nodes = np.fromstring(row[1].strip("[]"), sep = ' ')
                                attacked_nodes = [int(_) for _ in attacked_nodes]
                            else:
                                acc = ast.literal_eval(row[1])
                                acc_honest = [_ for i, _ in enumerate(acc) if i not in attacked_nodes]
                                acc_honest = sum(acc_honest) / len(acc_honest)
                                acc_data[centralities[[cent in root for cent in centralities].index(True)]].append(acc_honest)
                            i += 1

        # Plot the accuracies
        plt.figure(figsize=(10, 6))
        if any([x == None for x in acc_data.values()]):
            continue
        for cent, acc_aver in acc_data.items():
            plt.plot(range(len(acc_aver)), acc_aver, label = cent)

        plt.xlabel('Epoch') 
        plt.ylabel('Accuracy') 

        plt.grid(True)
        plt.legend()
        plt.savefig(data_dir_name + filename_data[:-4] + '.png')

# Plot accuracy averaged over the specified data
def plot_acc_aver(graph_type_used = '', dataset_name = 'fmnist'):
    # Setup
    dir_data = '/root/programming/Purdue-Research-Programs-Notes/data/full_decentralized/%s/' % dataset_name
    dir_data += graph_type_used + '/'
    centralities = ('none', 'in_deg_centrality', 'out_deg_centrality', 'closeness_centrality', 'betweeness_centrality', 'eigenvector_centrality')
    cent_data = {cent:[] for cent in centralities}
    aver_cent_data = {cent:[] for cent in centralities}
    seed_range = 3
    acc_data = []
    root_dir = ''
    # Get distinct settings
    acc_diff_fnames = set()
    for root, dirs, files in os.walk(dir_data):
        for fname in files:
            if fname.startswith('acc_'):
                fname_parts = re.split('_seed', fname)
                acc_diff_fnames.add(fname_parts[0])
        root_dir = root        
    
    # Create averaged dictionary data and plot
    for acc_fname in acc_diff_fnames:
        for iid_type in ('iid', 'non_iid'):
            for cent in cent_data.keys():
                for seed in range(seed_range):
                    acc_data_fname = acc_fname + '_seed_%d_iid_type_%s_cent_%s.csv' % (seed, iid_type, cent)
                    acc_data = []
                    with open(root_dir + acc_data_fname, 'r') as acc_data_file:
                        reader = csv.reader(acc_data_file)
                        i = 0
                        for row in reader:
                            if i == 0:
                                attacked_nodes = np.fromstring(row[1].strip("[]"), sep = ' ')
                                attacked_nodes = [int(_) for _ in attacked_nodes]
                            else:
                                acc = ast.literal_eval(row[1])
                                acc_honest = [_ for i, _ in enumerate(acc) if i not in attacked_nodes]
                                acc_honest = sum(acc_honest) / len(acc_honest)
                                acc_data.append(acc_honest)
                            i += 1
                    cent_data[cent].append(acc_data)
            # Calc averaged accuracies over different seeds
            for cent in centralities:
                aver_acc = []
                for cent_diff_seeds in zip(*cent_data[cent]):
                    aver_acc.append(sum(cent_diff_seeds) / len(cent_diff_seeds))
                aver_cent_data[cent] = aver_acc

            # Plot the accuracies
            plt.figure(figsize=(10, 6))
            if any([x == None for x in aver_cent_data.values()]):
                continue
            for cent, acc_aver in aver_cent_data.items():
                plt.plot(range(len(acc_aver)), acc_aver, label = cent)

            plt.xlabel('Epoch') 
            plt.ylabel('Accuracy') 

            plt.grid(True)
            plt.legend()
            plt.savefig(dir_data + acc_fname + 'iid_type_%s.png' % iid_type)

            # Reset values
            cent_data = {cent:[] for cent in cent_data.keys()}
            aver_cent_data = {cent:[] for cent in cent_data.keys()}

                    

# Used to generate ER graphs
def gen_ER_graph(n_clients, prob_conn = 0.5, graph_name = '', seed = 0):
    dir_networks = '/root/programming/Purdue-Research-Programs-Notes/data/full_decentralized/network_topologies/'
    graph = None
    is_strongly_connected = False
    while not is_strongly_connected:
        if prob_conn <= 0.3:
            graph = nx.fast_gnp_random_graph(n = n_clients, p = prob_conn, seed = seed, directed = True) 
        else:
            graph = nx.gnp_random_graph(n = n_clients, p = prob_conn, seed = seed, directed = True) 
        is_strongly_connected = nx.is_strongly_connected(graph)
        seed += 100
    adj_matrix = nx.adjacency_matrix(graph)
    np.savetxt(dir_networks + graph_name, adj_matrix.todense(), fmt='%d')
    
    return graph, adj_matrix

# Generate scale free graphs, cannot find a way to make strongly connected
def gen_dir_scale_free_graph(n_clients, type_graph = 'default', graph_name = '', seed = 0):
    dir_networks = '/root/programming/Purdue-Research-Programs-Notes/data/full_decentralized/network_topologies/'
    scale_free_configs = {
        "default": [0.41, 0.54, 0.05, 0.2, 0],
        "in_deg_focus": [0.6, 0.2, 0.2, 0.3, 0],
        "out_deg_focus": [0.2, 0.2, 0.6, 0, 0.3],
        "inter_conn_focus": [0.2, 0.6, 0.2, 0.1, 0.1],
        "equal_pref": [0.33, 0.33, 0.34, 0.1, 0.1]
    }
    graph = None
    is_strongly_connected = False
    alpha, beta, gamma, delta_in, delta_out = scale_free_configs[type_graph]
    graph = nx.scale_free_graph(n = n_clients, alpha = alpha, beta = beta, gamma = gamma, delta_in = delta_in, delta_out = delta_out, seed = seed) 

    # TODO fix this
    adj_matrix = nx.adjacency_matrix(graph)
    np.savetxt(dir_networks + graph_name, adj_matrix.todense(), fmt='%d')
    
    return graph, adj_matrix

# Used to generate geometric graphs
def gen_dir_geom_graph(n_clients, graph_type = '2d_close_nodes', graph_name = '', seed = 0):
    dir_networks = '/root/programming/Purdue-Research-Programs-Notes/data/full_decentralized/network_topologies/'
    geo_graph_configs = {
        '2d_close_nodes': [2, 0.5],
        '2d_far_nodes': [2, 0.2],
        '3d_close_nodes': [3, 0.6],
        '3d_far_nodes': [3, 0.3]
    }
    dim, radius = geo_graph_configs[graph_type]
    graph = None    
    is_strongly_connected = False
    while not is_strongly_connected:
        graph = nx.random_geometric_graph(n = n_clients, radius = radius, dim = dim, seed = seed) 
        graph = nx.DiGraph(graph)
        is_strongly_connected = nx.is_strongly_connected(graph)
        seed += 100
    adj_matrix = nx.adjacency_matrix(graph)
    np.savetxt(dir_networks + graph_name, adj_matrix.todense(), fmt='%d')
    
    return graph, adj_matrix

# Used to generate k-out graphs
def gen_k_out_graph(n_clients, k_val_percentage = 0.25, graph_name = '', seed = 0):
    dir_networks = '/root/programming/Purdue-Research-Programs-Notes/data/full_decentralized/network_topologies/'
    k_out_val = int(k_val_percentage * n_clients)
    print(k_out_val)
    graph = None    
    is_strongly_connected = False
    while not is_strongly_connected:
        graph = nx.random_k_out_graph(n = n_clients, k = k_out_val, alpha = 1, self_loops = False, seed = seed)
        seed += 100
        is_strongly_connected = nx.is_strongly_connected(graph)
    adj_matrix = nx.adjacency_matrix(graph)
    np.savetxt(dir_networks + graph_name, adj_matrix.todense(), fmt='%d')

    return graph, adj_matrix

# Used to generate preferencial attachment graph
def gen_pref_attach_graph(n_clients, graph_type = 'sparse', graph_name = '', seed = 0):
    dir_networks = '/root/programming/Purdue-Research-Programs-Notes/data/full_decentralized/network_topologies/'
    pref_attach_configs = {
        'sparse': 1,
        'medium': 2,
        'dense': 3,
    }
    graph = None    
    is_strongly_connected = False
    while not is_strongly_connected:
        graph = nx.barabasi_albert_graph(n = n_clients, m = pref_attach_configs[graph_type], seed = seed)
        seed += 100
        graph = nx.DiGraph(graph)
        is_strongly_connected = nx.is_strongly_connected(graph)
    adj_matrix = nx.adjacency_matrix(graph)
    np.savetxt(dir_networks + graph_name, adj_matrix.todense(), fmt='%d')

    return graph, adj_matrix

def make_graphs():
    dir_networks = '/root/programming/Purdue-Research-Programs-Notes/data/full_decentralized/network_topologies/'
    graph_type = ('ER', 'dir_scale_free', 'dir_geom', 'k_out', 'pref_attach')
    for graph in graph_type:
        if graph == 'ER':
            continue
        elif graph == 'dir_scale_free':
            continue
        elif graph == 'dir_geom':
            for geo_graph_config in ('2d_close_nodes', '2d_far_nodes', '3d_close_nodes', '3d_far_nodes'):
                for seed in range(20):
                    graph_name = 'dir_geom_graph_c_20_type_%s_seed_%d.txt' % (geo_graph_config, seed)
                    gen_dir_geom_graph(20, graph_type = geo_graph_config, graph_name = graph_name, seed = seed)
        elif graph == 'k_out':
            for k_dec in (0.25, 0.50, 0.75):
                for seed in range(20):
                    graph_name = 'k_out_graph_c_20_k_%d_seed_%d.txt' % (int(20 * k_dec), seed)
                    print(graph_name)
                    gen_k_out_graph(20, k_val_percentage = k_dec, graph_name = graph_name, seed = seed)    
        elif graph == 'pref_attach':
            for graph_type in ('sparse', 'medium', 'dense'):
                for seed in range(20):
                    graph_name = 'pref_attach_graph_c_20_type_%s_seed_%d.txt' % (graph_type, seed)
                    gen_pref_attach_graph(20, graph_type = graph_type, graph_name = graph_name, seed = seed)

if __name__ == '__main__':
    plot_acc_aver('pref_attach_graph_c_20_type_sparse', 'fmnist')

