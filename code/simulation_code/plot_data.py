import os
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import ast
from nn_FL_de_cent import *
import networkx as nx
import re
import pandas as pd
import glob

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
def plot_acc_aver(graph_type_used = '', dataset_name = 'fmnist', seed_range = 50):
    # Setup
    dir_graphs = '../../data/plots/'
    dir_data = '../../data/full_decentralized/%s/' % dataset_name
    dir_data += graph_type_used + '/'
    centralities = ('none', 'in_deg_centrality', 'out_deg_centrality', 'closeness_centrality', 'betweeness_centrality', 'eigenvector_centrality')
    cent_data = {cent:[] for cent in centralities}
    aver_cent_data = {cent:[] for cent in centralities}
    acc_data = []
    root_dir = ''
    cent_name_dir = {'none':'No Attack', 'in_deg_centrality': 'In-Degree Centrality Based Attack', 'out_deg_centrality': 'Out-Degree Centrality Based Attack', 'closeness_centrality' :'Closeness Centrality Based Attack', 'betweeness_centrality' :'Betweenness Centrality Based Attack', 'eigenvector_centrality': 'Eigenvector Centrality Based Attack'}

    # Get distinct settings
    acc_diff_fnames = set()
    for root, dirs, files in os.walk(dir_data):
        for fname in files:
            if fname.startswith('acc_'):
                fname_parts = re.split('_seed', fname)
                if '300' not in fname_parts[0]:
                    if '.png' not in fname_parts[0]:
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

            if 'ER' in graph_type_used:
                graph_type_name = 'ER'
            elif 'pref_attach' in graph_type_used:
                graph_type_name = 'Preferential Attachment'
            elif 'dir_geom' in graph_type_used:
                graph_type_name = 'Directed Geometric'
            elif 'out' in graph_type_used:
                graph_type_name = 'K-Out'
            elif 'SNAP' in graph_type_used:
                graph_type_name = 'SNAP Dataset'

            # Plot the accuracies
            plt.figure(figsize=(10, 6))
            if any([x == None for x in aver_cent_data.values()]):
                continue
            for cent, acc_aver in aver_cent_data.items():
                plt.plot(range(25, len(acc_aver)), acc_aver[25:], label = cent_name_dir[cent])  # start plotting from 25th epoch
            plt.title('Model Accuracy over Epochs under Different Attacks \n for %s Graph' % (graph_type_name), fontsize=16)
            plt.xlabel('Epoch') 
            plt.ylabel('Accuracy') 
            plt.minorticks_on()
            plt.grid(True)
            plt.ylim(plt.ylim()[0], plt.ylim()[-1])
            plt.xlim(25, plt.xlim()[-1])  # start x-axis from 25th epoch
            plt.vlines(x = 25, ymin = 0, ymax = plt.ylim()[1], colors = 'black', linestyles = 'dashed', label = 'Attack starts')
            # plt.legend()
            plt.savefig(dir_graphs + graph_type_used + '_' + acc_fname + '_iid_type_%s.png' % iid_type)

            # Reset values
            cent_data = {cent:[] for cent in cent_data.keys()}
            aver_cent_data = {cent:[] for cent in cent_data.keys()}

# Plot accuracy averaged over the specified data
def plot_acc_aver_snap(graph_type_used = '', dataset_name = 'fmnist'):
    # Setup
    if 'SNAP' not in graph_type_used:
        print('This can only be used with SNAP dataset graphs!')
        return
    dir_graphs = '../../data/plots/'
    dir_data = '../../data/full_decentralized/%s/' % dataset_name
    dir_data += graph_type_used + '/'
    centralities = ('none', 'in_deg_centrality', 'out_deg_centrality', 'closeness_centrality', 'betweeness_centrality', 'eigenvector_centrality')
    cent_data = {cent:[] for cent in centralities}
    aver_cent_data = {cent:[] for cent in centralities}
    seed_range = 5
    acc_data = []
    root_dir = ''
    cent_name_dir = {'none':'No Attack', 'in_deg_centrality': 'In-Degree Centrality Based Attack', 'out_deg_centrality': 'Out-Degree Centrality Based Attack', 'closeness_centrality' :'Closeness Centrality Based Attack', 'betweeness_centrality' :'Betweenness Centrality Based Attack', 'eigenvector_centrality': 'Eigenvector Centrality Based Attack'}

    # Get distinct settings
    acc_diff_fnames = set()
    for root, dirs, files in os.walk(dir_data):
        for fname in files:
            if fname.startswith('acc_'):
                fname_parts = re.split('_seed', fname)
                if '300' not in fname_parts[0]:
                    if '.png' not in fname_parts[0]:
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

            if 'ER' in graph_type_used:
                graph_type_name = 'ER'
            elif 'pref_attach' in graph_type_used:
                graph_type_name = 'Preferential Attachment'
            elif 'dir_geom' in graph_type_used:
                graph_type_name = 'Directed Geometric'
            elif 'out' in graph_type_used:
                graph_type_name = 'K-Out'
            elif 'SNAP' in graph_type_used:
                graph_type_name = 'SNAP Dataset'

            # Plot the accuracies
            plt.figure(figsize=(10, 6))
            if any([x == None for x in aver_cent_data.values()]):
                continue
            for cent, acc_aver in aver_cent_data.items():
                plt.plot(range(len(acc_aver)), acc_aver, label = cent_name_dir[cent])
            plt.title('Model Accuracy over Epochs under Different Attacks \n for %s Graph' % (graph_type_name), fontsize=16)
            plt.xlabel('Epoch') 
            plt.ylabel('Accuracy') 
            plt.minorticks_on()
            plt.grid(True)
            plt.ylim(0.1, plt.ylim()[-1])
            plt.xlim(0, plt.xlim()[-1])
            plt.legend()
            plt.vlines(x=25, ymin=0, ymax=plt.ylim()[1], colors='black', linestyles='dashed', label='Attack starts')
            plt.savefig(dir_graphs + graph_type_used + '_' + acc_fname + '_iid_type_%s.png' % iid_type)

            # Reset values
            cent_data = {cent:[] for cent in cent_data.keys()}
            aver_cent_data = {cent:[] for cent in cent_data.keys()}
        
# Used to generate ER graphs
def gen_ER_graph(n_clients, prob_conn = 0.5, graph_name = '', seed = 0):
    dir_networks = '../../data/full_decentralized/network_topologies/'
    graph = None
    is_strongly_connected = False
    while not is_strongly_connected:
        if prob_conn <= 0.1:
            graph = nx.gnp_random_graph(n = n_clients, p = prob_conn, seed = seed, directed = False)
            graph = nx.DiGraph(graph)
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
    dir_networks = '../../data/full_decentralized/network_topologies/'
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
    dir_networks = '../../data/full_decentralized/network_topologies/'
    geo_graph_configs = {
        '2d_very_close_nodes': [2, 0.5],
        '2d_close_nodes': [2, 0.3],
        '2d_far_nodes': [2, 0.2],
        '2d_r_02' : [2, 0.2],
        '2d_r_04' : [2, 0.4],
        '2d_r_06' : [2, 0.6]
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
    dir_networks = '../../data/full_decentralized/network_topologies/'
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
    dir_networks = '../../data/full_decentralized/network_topologies/'
    pref_attach_configs = {
        'sparse': 1,
        'medium': 2,
        'dense': 3,
        'dense_2': 5,
        'dense_3': 6,
        'dense_4': 7
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
    dir_networks = '../../data/full_decentralized/network_topologies/'
    graph_type = ('ER', 'dir_scale_free', 'dir_geom', 'k_out', 'pref_attach')
    # graph_type = ('k_out',)
    n_clients = 500
    seed_range = 5
    for graph in graph_type:
        print(graph)
        if graph == 'ER':
            for p in (0.1, 0.3, 0.5, 0.7, 0.9):
                for seed in range(seed_range):
                    graph_name = 'ER_graph_c_%d_p_0%d_seed_%d.txt' % (n_clients, int(p * 10), seed)
                    gen_ER_graph(n_clients, p, graph_name, seed)
        elif graph == 'dir_scale_free':
            continue
        elif graph == 'dir_geom':
            for geo_graph_config in ('2d_very_close_nodes', '2d_close_nodes', '2d_far_nodes'):
                print(geo_graph_config)
                for seed in range(seed_range):
                    print(seed)
                    graph_name = 'dir_geom_graph_c_%d_type_%s_seed_%d.txt' % (n_clients, geo_graph_config, seed)
                    gen_dir_geom_graph(n_clients, graph_type = geo_graph_config, graph_name = graph_name, seed = seed)
        elif graph == 'k_out':
            for k_dec in (0.25, 0.50, 0.75):
                for seed in range(seed_range):
                    graph_name = 'k_out_graph_c_%d_k_%d_seed_%d.txt' % (n_clients, int(n_clients * k_dec), seed)
                    print(graph_name)
                    gen_k_out_graph(n_clients, k_val_percentage = k_dec, graph_name = graph_name, seed = seed)    
        elif graph == 'pref_attach':
            for graph_type in ('dense', 'dense_2', 'dense_4'):
                for seed in range(seed_range):
                    graph_name = 'pref_attach_graph_c_%d_type_%s_seed_%d.txt' % (n_clients, graph_type, seed)
                    gen_pref_attach_graph(n_clients, graph_type = graph_type, graph_name = graph_name, seed = seed)

def calc_2_set_similarity(list_1, list_2):
    set_1 = set(list_1)
    set_2 = set(list_2)
    if len(set_1) == 0 or len(set_2) == 0:
        return 0

    return 1 - (len(set_1 - set_2) + len(set_2 - set_1)) / (2 * len(set_1))

def calc_inter_set_similarity(list_cents):
    sim_score_dict = {i / 10 : 0 for i in range(11)}
    for i, cent_1_list in enumerate(list_cents):
        temp_cent_sim_1_2 = []
        for j, cent_2_list in enumerate(list_cents):
            if i != j:
                temp_cent_sim_1_2.append(calc_2_set_similarity(cent_1_list, cent_2_list))
        aver_cent_1_sim = np.mean(temp_cent_sim_1_2) # TODO fix this to place more emphasis on similar nodes
        sim_score_dict[float(int(10 * aver_cent_1_sim) / 10)] += 1


    weighted_sim = 0
    for weight, sim_num in sim_score_dict.items():
        weighted_sim += weight * sim_num
    return weighted_sim / len(list_cents)
    # return similarity / norm_param

def score_graph_centralities_similarity(graph_name, num_attackers = 0):
    adv_cent_similarity_arr = []
    dir_graphs = '../../data/full_decentralized/network_topologies/'
    for root, dirs, files in os.walk(dir_graphs):
        for fname in files:
            if fname.startswith(graph_name) and '-checkpoint' not in fname:
                adj_matrix = np.loadtxt(root + fname)
                centrality_data = sort_by_centrality(calc_centralities(len(adj_matrix[0]), create_graph(adj_matrix)))
                adv_centralities = [centrality_data[i][0:num_attackers] for i in range(len(centrality_data))]
                adv_cent_similarity_arr.append(calc_inter_set_similarity(adv_centralities))
    return np.mean(adv_cent_similarity_arr), np.var(adv_cent_similarity_arr)

def score_graph_types_centralities_similarity(dataset_name, adv_percentage = 0.2):
    data_dir = '../../data/full_decentralized/%s/' % dataset_name

    # Get all graph types
    graph_types = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    # Get all similarity results
    sim_data = []
    for graph_type in graph_types:
        # Get the client number
        n_clients = re.search('_c_(\d+)_', graph_type)
        if n_clients is not None:
            n_clients = float(n_clients.group(1))
            n_advs = int(n_clients * adv_percentage)

            # Run the function and get the two floats
            mean_similarity, var_similarity = score_graph_centralities_similarity(graph_type, n_advs)

            # Save the results
            sim_data.append([graph_type, mean_similarity, var_similarity])

    # Write the results to a .txt file
    with open('../../data/full_decentralized/network_cent_sim/cent_sim_adv_0%d_dset_%s.txt' % (int(adv_percentage * 10), dataset_name), 'w', newline = '') as cent_similarity_file:
        writer = csv.writer(cent_similarity_file, delimiter = ';')
        writer.writerows(sim_data)

def make_similarity_graphs(dataset_name):
    data_dir = '../../data/full_decentralized/network_cent_sim/'
    result_dir = '../../data/full_decentralized/network_cent_sim_plots/'
    graph_types = ('ER', 'dir_scale_free', 'dir_geom', 'k_out', 'pref_attach', 'SNAP_Cisco')

    cent_files = [_ for _ in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, _))]
    cent_files = sorted([_ for _ in cent_files if dataset_name in _])
    last_cent_file = cent_files.pop(1)
    cent_files.append(last_cent_file)

    cent_sim_data = {}
    cent_files.pop(0)
    snap_data = {}
    # Calculate averaged data
    for cent_file in cent_files:
        with open(os.path.join(data_dir, cent_file), 'r') as cent_data_file:
            for graph_cent_sim_data in cent_data_file:
                graph_type, mean_similarity, var_similarity = graph_cent_sim_data.strip().split(';')
                mean_similarity = float(mean_similarity)
                var_similarity = float(var_similarity)

                if 'SNAP' in graph_type:
                    if graph_type not in snap_data:
                        snap_data[graph_type] = {'mean_sim': [], 'var_sim': []}
                    snap_data[graph_type]['mean_sim'].append(mean_similarity)
                    snap_data[graph_type]['var_sim'].append(var_similarity)

                elif 'ER' in graph_type:
                    if 'ER' not in cent_sim_data:
                        cent_sim_data['ER'] = {'mean_sim': {x:[] for x in (0.1, 0.3, 0.5, 0.7, 0.9)}, 'var_sim': {x:[] for x in (0.1, 0.3, 0.5, 0.7, 0.9)}}
                    cent_sim_data['ER']['mean_sim'][float(graph_type[-1]) / 10].append(mean_similarity)
                    cent_sim_data['ER']['var_sim'][float(graph_type[-1]) / 10].append(var_similarity)
                elif 'dir_geom' in graph_type:
                    if 'dir_geom' not in cent_sim_data:
                        cent_sim_data['dir_geom'] = {'mean_sim': {x:[] for x in ('r = 0.2', 'r = 0.3', 'r = 0.5')}, 'var_sim': {x:[] for x in ('r = 0.2', 'r = 0.3', 'r = 0.5')}}
                    match = re.search(r'type_(2d.*nodes)', graph_type)
                    if match.group(1) == '2d_very_close_nodes':
                        radius_val = 'r = 0.5'
                    elif match.group(1) == '2d_close_nodes':
                        radius_val = 'r = 0.3'
                    elif match.group(1) == '2d_far_nodes':
                        radius_val = 'r = 0.2'
                    cent_sim_data['dir_geom']['mean_sim'][radius_val].append(mean_similarity)
                    cent_sim_data['dir_geom']['var_sim'][radius_val].append(var_similarity)
                elif 'pref_attach' in graph_type:
                    if 'pref_attach' not in cent_sim_data:
                        cent_sim_data['pref_attach'] = {'mean_sim': {x:[] for x in ('Dense Type 0', 'Dense Type 2', 'Dense Type 4')}, 'var_sim': {x:[] for x in ('Dense Type 0', 'Dense Type 2', 'Dense Type 4')}}
                    match = re.search(r'type_(dense(?:_2|_4)?)', graph_type)
                    if match is not None:
                        if match.group(1) in ('dense', 'dense_2', 'dense_4'):
                            if match.group(1) == 'dense':
                                type_pref_attach = 'Dense Type 0'
                            elif match.group(1) == 'dense_2':
                                type_pref_attach = 'Dense Type 2'
                            elif match.group(1) == 'dense_4':
                                type_pref_attach = 'Dense Type 4'
                            
                            cent_sim_data['pref_attach']['mean_sim'][type_pref_attach].append(mean_similarity)
                            cent_sim_data['pref_attach']['var_sim'][type_pref_attach].append(var_similarity)
                elif 'k_out' in graph_type:
                    if 'k_out' not in cent_sim_data:
                        cent_sim_data['k_out'] = {'mean_sim': {x:[] for x in (2, 5, 10, 15)}, 'var_sim': {x:[] for x in (2, 5, 10, 15)}}
                    match = re.search(r'_k_(\d+)', graph_type)
                    cent_sim_data['k_out']['mean_sim'][int(match.group(1))].append(mean_similarity)
                    cent_sim_data['k_out']['var_sim'][int(match.group(1))].append(var_similarity)

    # Average SNAP data and add
    cent_sim_data['SNAP'] = {'mean_sim': [], 'var_sim': []}
    temp_combined_snap = []
    for snap_graph in snap_data.keys():
        temp_combined_snap.append(snap_data[snap_graph]['mean_sim'])
    for adv_perc_sim_data_snap in zip(*temp_combined_snap):
        snap_mean, snap_var = np.mean(adv_perc_sim_data_snap), np.var(adv_perc_sim_data_snap)
        cent_sim_data['SNAP']['mean_sim'].append(snap_mean)
        cent_sim_data['SNAP']['var_sim'].append(snap_var)

    # Plot
    for graph_type, graph_sim_data in cent_sim_data.items():
        if 'SNAP' in graph_type:
            plt.figure()
            plt.errorbar([float(i) / 10 for i in range(1, len(graph_sim_data['mean_sim']) + 1)], graph_sim_data['mean_sim'], yerr = graph_sim_data['var_sim'], fmt = '-o')

            plt.title('Centralities Similarity Score for \n SNAP Dataset Graph', fontsize = 16, fontweight = 'bold')
            plt.xlabel('Adversarial Fraction', fontsize = 14)
            plt.ylabel('Normalized Centralities Similarity Score', fontsize = 14)
            plt.ylim(0, 1)
            plt.grid(True)  # Add a grid to the plot

            # Increase the size and weight of the axis tick labels
            plt.xticks(fontsize = 12, fontweight = 'bold')
            plt.yticks(fontsize = 12, fontweight = 'bold')

            plt.tight_layout()  # Ensure that all elements of the plot fit within the figure area
            plt.savefig(result_dir + '%s_%s.png' % (graph_type, dataset_name), dpi = 300)  # Save the figure with a high resolution
            plt.close()
            plt.close()
        elif 'ER' in graph_type:
            plt.figure()

            # Iterate over the keys and values in the 'mean_sim' dictionary
            for setting, mean_sim_values in graph_sim_data['mean_sim'].items():
                # Get the corresponding variance values
                var_sim_values = graph_sim_data['var_sim'][setting]
                
                # Plot the mean values with the variance as error bars
                plt.errorbar([float(i) / 10 for i in range(1, len(mean_sim_values) + 1)], mean_sim_values, yerr = var_sim_values, fmt = '-o', label = f'p = {setting}')

            plt.title('Centralities Similarity Score for ER Graphs', fontsize = 16, fontweight = 'bold')
            plt.xlabel('Adversarial Fraction', fontsize = 14)
            plt.ylabel('Normalized Centralities Similarity Score', fontsize = 14)
            plt.ylim(0, 1)
            plt.grid(True)  # Add a grid to the plot
            plt.legend(fontsize = 12)  # Add a legend to the plot

            # Increase the size and weight of the axis tick labels
            plt.xticks(fontsize = 12, fontweight = 'bold')
            plt.yticks(fontsize = 12, fontweight = 'bold')

            plt.tight_layout()  # Ensure that all elements of the plot fit within the figure area
            plt.savefig(result_dir + '%s_%s.png' % (graph_type, dataset_name), dpi = 300)  # Save the figure with a high resolution
            plt.close()
        elif 'pref_attach' in graph_type:
            plt.figure()

            # Iterate over the keys and values in the 'mean_sim' dictionary
            for setting, mean_sim_values in graph_sim_data['mean_sim'].items():
                # Get the corresponding variance values
                var_sim_values = graph_sim_data['var_sim'][setting]
                # Plot the mean values with the variance as error bars
                plt.errorbar([float(i) / 10 for i in range(1, len(mean_sim_values) + 1)], mean_sim_values, yerr = var_sim_values, fmt = '-o', label = f'Type: {setting}')

            plt.title('Centralities Similarity Score for \n Preferential Attachment Graphs', fontsize = 16, fontweight = 'bold')
            plt.xlabel('Adversarial Fraction', fontsize = 14)
            plt.ylabel('Normalized Centralities Similarity Score', fontsize = 14)
            plt.ylim(0, 1)
            plt.grid(True)  # Add a grid to the plot
            plt.legend(fontsize = 12)  # Add a legend to the plot

            # Increase the size and weight of the axis tick labels
            plt.xticks(fontsize = 12, fontweight = 'bold')
            plt.yticks(fontsize = 12, fontweight = 'bold')

            plt.tight_layout()  # Ensure that all elements of the plot fit within the figure area
            plt.savefig(result_dir + '%s_%s.png' % (graph_type, dataset_name), dpi = 300)  # Save the figure with a high resolution
            plt.close()
        elif 'dir_geom' in graph_type:
            plt.figure()

            # Iterate over the keys and values in the 'mean_sim' dictionary
            for setting, mean_sim_values in graph_sim_data['mean_sim'].items():
                # Get the corresponding variance values
                var_sim_values = graph_sim_data['var_sim'][setting]
                # Plot the mean values with the variance as error bars
                plt.errorbar([float(i) / 10 for i in range(1, len(mean_sim_values) + 1)], mean_sim_values, yerr = var_sim_values, fmt = '-o', label = f'radius: {setting}')

            plt.title('Centralities Similarity Score for \n Directed Geometric Graphs', fontsize = 16, fontweight = 'bold')
            plt.xlabel('Adversarial Fraction', fontsize = 14)
            plt.ylabel('Normalized Centralities Similarity Score', fontsize = 14)
            plt.ylim(0, 1)
            plt.grid(True)  # Add a grid to the plot
            plt.legend(fontsize = 12)  # Add a legend to the plot

            # Increase the size and weight of the axis tick labels
            plt.xticks(fontsize = 12, fontweight = 'bold')
            plt.yticks(fontsize = 12, fontweight = 'bold')

            plt.tight_layout()  # Ensure that all elements of the plot fit within the figure area
            plt.savefig(result_dir + '%s_%s.png' % (graph_type, dataset_name), dpi = 300)  # Save the figure with a high resolution
            plt.close()
        elif 'k_out' in graph_type:
            plt.figure()

            # Iterate over the keys and values in the 'mean_sim' dictionary
            for setting, mean_sim_values in graph_sim_data['mean_sim'].items():
                # Get the corresponding variance values
                var_sim_values = graph_sim_data['var_sim'][setting]                
                # Plot the mean values with the variance as error bars
                plt.errorbar([float(i) / 10 for i in range(1, len(mean_sim_values) + 1)], mean_sim_values, yerr = var_sim_values, fmt = '-o', label = f'k = {setting}')

            plt.title('Centralities Similarity Score for \n K-Out Graphs', fontsize = 16, fontweight = 'bold')
            plt.xlabel('Adversarial Fraction', fontsize = 14)
            plt.ylabel('Normalized Centralities Similarity Score', fontsize = 14)
            plt.ylim(0, 1)
            plt.grid(True)  # Add a grid to the plot
            plt.legend(fontsize = 12)  # Add a legend to the plot

            # Increase the size and weight of the axis tick labels
            plt.xticks(fontsize = 12, fontweight = 'bold')
            plt.yticks(fontsize = 12, fontweight = 'bold')

            plt.tight_layout()  # Ensure that all elements of the plot fit within the figure area
            plt.savefig(result_dir + '%s_%s.png' % (graph_type, dataset_name), dpi = 300)  # Save the figure with a high resolution
            plt.close()

def calc_centrality_measure_aver_variance(graph_name):
    # in_deg_centrality[node], out_deg_centrality[node], closeness_centrality[node], betweeness_centrality[node], eigenvector_centrality[node]
    cent_measure_var_array = []
    cent_variance = {_: [] for _ in ('in_deg', 'out_deg', 'closeness', 'betweeness', 'eigenvector')}
    dir_graphs = '../../data/full_decentralized/network_topologies/'
    for root, dirs, files in os.walk(dir_graphs):
        for fname in files:
            if fname.startswith(graph_name) and '-checkpoint' not in fname:
                adj_matrix = np.loadtxt(root + fname)
                centrality_data = calc_centralities(len(adj_matrix[0]), create_graph(adj_matrix))
                centrality_data = np.array(list(centrality_data.values()))
                # centrality_data = centrality_data / centrality_data.sum(axis = 0)  # normalize each column
                for i, cent_measure in enumerate(cent_variance.keys()):
                    cent_variance[cent_measure].append(np.var(centrality_data[:, i]))
    for cent_measure in cent_variance.keys():
        cent_variance[cent_measure] = np.mean(cent_variance[cent_measure])
    
    return cent_variance

# TODO normalize all measures to sum to 1
def make_variance_histograms(dataset_name):
    data_dir = '../../data/full_decentralized/%s/' % dataset_name
    result_dir = '../../data/full_decentralized/network_cent_variance_plots/'
    # Get all graph types
    graph_types = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    # Get all variance results
    var_data_graphs = {'ER':{20:{}, 100:{}, 500:{}}, 'SNAP':[], 'k_out':{20:{}, 100:{}, 500:{}}, 'pref_attach':{20:{}, 100:{}, 500:{}}, 'dir_geom':{20:{}, 100:{}, 500:{}}}

    # Get variance data
    for graph_type in graph_types:
        cent_vars = np.array(list(calc_centrality_measure_aver_variance(graph_type).values()))
        n_clients = re.search('_c_(\d+)_', graph_type)
        if n_clients is not None:
            n_clients = int(n_clients.group(1))
        if 'ER' in graph_type:
            prob_conn = re.search('_p_0(\d+)', graph_type)
            if prob_conn is not None:
                prob_conn = float(prob_conn.group(1)) / 10
                var_data_graphs['ER'][n_clients][prob_conn] = cent_vars
        elif 'SNAP' in graph_type:
            var_data_graphs['SNAP'].append(cent_vars)
        elif 'k_out' in graph_type:
            k_val = re.search('_k_(\d+)', graph_type)
            if k_val is not None:
                k_val = int(k_val.group(1))
                var_data_graphs['k_out'][n_clients][k_val] = cent_vars
        elif 'pref_attach' in graph_type:
            type_pref_attach = re.search(r'type_(dense(?:_2|_4)?)', graph_type)
            if type_pref_attach is not None:
                var_data_graphs['pref_attach'][n_clients][type_pref_attach.group(1)] = cent_vars
        elif 'dir_geom' in graph_type:
            type_dir_geom = re.search(r'type_(2d.*nodes)', graph_type)
            if type_dir_geom is not None:
                if type_dir_geom.group(1) == '2d_very_close_nodes':
                    radius_val = 'r = 0.5'
                elif type_dir_geom.group(1) == '2d_close_nodes':
                    radius_val = 'r = 0.3'
                elif type_dir_geom.group(1) == '2d_far_nodes':
                    radius_val = 'r = 0.2'
                var_data_graphs['dir_geom'][n_clients][radius_val] = cent_vars
    
    # Plot the variacne plots
    label_arr = ['In-Degree Centrality', 'Out-Degree Centrality', 'Closeness Centrality', 'Betweenness Centrality', 'Eigenvector Centrality']  # replace with your actual labels
    # Define your color map
    cmap = plt.get_cmap('tab10')

    # Generate colors from the color map
    colors = [cmap(i) for i in range(len(label_arr))]
    width_bar = 0.15  # width of the bars

    for graph_type, graph_data in var_data_graphs.items():
        if graph_type == 'SNAP':
            plt.figure(figsize = (18, 13))  # Set the figure size
            avg_data_snap_var = np.mean(graph_data, axis = 0)
            plt.bar(label_arr, avg_data_snap_var, color = colors, alpha = 0.7, label = 'SNAP')
            lower_font_size = 25
            higher_font_size = 35
            graph_var_data = {k: graph_var_data[k] for k in sorted(graph_var_data)}
            plt.subplots_adjust(bottom = 0.25)
            ax = plt.gca()  # get current axes
            ax.tick_params(axis = 'y', labelsize = lower_font_size)
            ax.tick_params(axis = 'x', labelsize = lower_font_size)

            plt.xticks(rotation = 30)  # Rotate x-axis labels
            ax.set_xlabel('Centrality Measure', fontsize = lower_font_size)
            ax.set_ylabel('Averaged Variance of Node Centrality', fontsize = lower_font_size)
            ax.set_title('Averaged Node Centrality Variances for Different Centrality Measures \n for Different Snap Graphs', fontsize = higher_font_size)
            ax.grid(True, zorder = 0)
            plt.savefig(result_dir + '%s_graphs_variance_histograms.png' % graph_type)
            plt.close()
        else:
            for n_clients, graph_var_data in graph_data.items():
                if graph_var_data:  # check if the dictionary is not empty
                    if graph_type == 'ER':
                        lower_font_size = 25
                        higher_font_size = 35
                        fig, ax = plt.subplots(figsize = (18, 13))
                        x = np.arange(len(graph_var_data))  # the label locations
                        for i, (key, values) in enumerate(graph_var_data.items()):
                            area_dist_data = [x[i] -3 * width_bar / 2 + j * width_bar for j in range(len(values))]
                            for j, value in enumerate(values):
                                ax.bar(area_dist_data[j], value, width_bar, label=f'{label_arr[j]}', color=colors[j])  # use color
                        # Create a custom legend
                        custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(len(label_arr))]
                        plt.subplots_adjust(bottom = 0.2)
                        ax.legend(custom_lines, label_arr, loc = 'lower center', bbox_to_anchor = (0.5, -0.27), fancybox = True, shadow = True, ncol = 3, fontsize = lower_font_size)

                        ax.set_xticks(x)
                        ax.set_xticklabels(graph_var_data.keys(), fontsize = lower_font_size)
                        ax = plt.gca()  # get current axes
                        ax.tick_params(axis = 'y', labelsize = lower_font_size)
                        ax.set_xlabel('Probability of Connection', fontsize = lower_font_size)
                        ax.set_ylabel('Variance of Node Centrality', fontsize = lower_font_size)
                        ax.set_title('Node Centrality Variances for Different Centrality Measures \n for %s Graphs with %d nodes' % (graph_type, n_clients), fontsize = higher_font_size)
                        ax.grid(True, zorder = 0)
                        plt.savefig(result_dir + '%s_graphs_c_%d_variance_histograms.png' % (graph_type, n_clients))
                        plt.close()
                    elif graph_type == 'pref_attach':
                        lower_font_size = 25
                        higher_font_size = 35
                        fig, ax = plt.subplots(figsize=(18, 13))
                        x = np.arange(len(graph_var_data))  # the label locations
                        for i, (key, values) in enumerate(graph_var_data.items()):
                            area_dist_data = [x[i] -3 * width_bar / 2 + j * width_bar for j in range(len(values))]
                            for j, value in enumerate(values):
                                ax.bar(area_dist_data[j], value, width_bar, label=f'{label_arr[j]}', color=colors[j])  # use color
                        # Create a custom legend
                        custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(len(label_arr))]
                        plt.subplots_adjust(bottom = 0.2)
                        ax.legend(custom_lines, label_arr, loc = 'lower center', bbox_to_anchor = (0.5, -0.27), fancybox = True, shadow = True, ncol = 3, fontsize = lower_font_size)

                        ax.set_xticks(x)
                        x_tick_labels = ['Dense Type 0', 'Dense Type 2', 'Dense Type 4']
                        ax.set_xticklabels(x_tick_labels, fontsize = lower_font_size)
                        ax = plt.gca()  # get current axes
                        ax.tick_params(axis = 'y', labelsize = lower_font_size)
                        ax.set_xlabel('Preferential Attachment Graph Type', fontsize = lower_font_size)
                        ax.set_ylabel('Variance of Node Centrality', fontsize = lower_font_size)
                        ax.set_title('Node Centrality Variances for Different Centrality Measures \n for Preferential Attachment Graphs with %d nodes' % n_clients, fontsize = higher_font_size)
                        ax.grid(True, zorder = 0)
                        plt.savefig(result_dir + '%s_graphs_c_%d_variance_histograms.png' % (graph_type, n_clients))
                        plt.close()
                    elif graph_type == 'dir_geom':
                        lower_font_size = 25
                        higher_font_size = 35
                        graph_var_data = {k: graph_var_data[k] for k in sorted(graph_var_data)}
                        fig, ax = plt.subplots(figsize=(18, 13))
                        x = np.arange(len(graph_var_data))  # the label locations
                        for i, (key, values) in enumerate(graph_var_data.items()):
                            area_dist_data = [x[i] -3 * width_bar / 2 + j * width_bar for j in range(len(values))]
                            for j, value in enumerate(values):
                                ax.bar(area_dist_data[j], value, width_bar, label=f'{label_arr[j]}', color=colors[j])  # use color
                        # Create a custom legend
                        custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(len(label_arr))]
                        plt.subplots_adjust(bottom = 0.2)
                        ax.legend(custom_lines, label_arr, loc = 'lower center', bbox_to_anchor = (0.5, -0.27), fancybox = True, shadow = True, ncol = 3, fontsize = lower_font_size)

                        ax.set_xticks(x)
                        # x_tick_labels = ['Dense Type 0', 'Dense Type 2', 'Dense Type 4']
                        ax.set_xticklabels(graph_var_data.keys(), fontsize = lower_font_size)
                        ax = plt.gca()  # get current axes
                        ax.tick_params(axis = 'y', labelsize = lower_font_size)
                        ax.set_xlabel('Directed Geometric Graph Type', fontsize = lower_font_size)
                        ax.set_ylabel('Variance of Node Centrality', fontsize = lower_font_size)
                        ax.set_title('Node Centrality Variances for Different Centrality Measures \n for Directed Geometric Graphs with %d nodes' % n_clients, fontsize = higher_font_size)
                        ax.grid(True, zorder = 0)
                        plt.savefig(result_dir + '%s_graphs_c_%d_variance_histograms.png' % (graph_type, n_clients))
                        plt.close()
                    elif graph_type == 'k_out':
                        lower_font_size = 25
                        higher_font_size = 35
                        graph_var_data = {k: graph_var_data[k] for k in sorted(graph_var_data)}
                        fig, ax = plt.subplots(figsize=(18, 13))
                        x = np.arange(len(graph_var_data))  # the label locations
                        for i, (key, values) in enumerate(graph_var_data.items()):
                            area_dist_data = [x[i] -3 * width_bar / 2 + j * width_bar for j in range(len(values))]
                            for j, value in enumerate(values):
                                ax.bar(area_dist_data[j], value, width_bar, label=f'{label_arr[j]}', color=colors[j])  # use color
                        # Create a custom legend
                        custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(len(label_arr))]
                        plt.subplots_adjust(bottom = 0.2)
                        ax.legend(custom_lines, label_arr, loc = 'lower center', bbox_to_anchor = (0.5, -0.27), fancybox = True, shadow = True, ncol = 3, fontsize = lower_font_size)

                        ax.set_xticks(x)
                        # x_tick_labels = ['Dense Type 0', 'Dense Type 2', 'Dense Type 4']
                        ax.set_xticklabels(graph_var_data.keys(), fontsize = lower_font_size)
                        ax = plt.gca()  # get current axes
                        ax.tick_params(axis = 'y', labelsize = lower_font_size)
                        ax.set_xlabel('Value of K in a K-Out Graph', fontsize = lower_font_size)
                        ax.set_ylabel('Variance of Node Centrality', fontsize = lower_font_size)
                        ax.set_title('Node Centrality Variances for Different Centrality Measures \n for K-Out Graphs with %d nodes' % n_clients, fontsize = higher_font_size)
                        ax.grid(True, zorder = 0)
                        plt.savefig(result_dir + '%s_graphs_c_%d_variance_histograms.png' % (graph_type, n_clients))
                        plt.close()

def plot_scored_tradeoff_time_centrality(graph_type_used = '', dataset_name = 'fmnist', seed_range = 50):
    # Setup
    dir_graphs = '../../data/full_decentralized/network_optimality_score_graphs/'
    dir_data = '../../data/full_decentralized/%s/' % dataset_name
    dir_data += graph_type_used + '/'
    centralities = ('none', 'in_deg_centrality', 'out_deg_centrality', 'closeness_centrality', 'betweeness_centrality', 'eigenvector_centrality')
    cent_data = {cent:[] for cent in centralities}
    aver_cent_data = {cent:[] for cent in centralities}
    acc_data = []
    root_dir = ''
    cent_name_dir = {'none':'No Attack', 'in_deg_centrality': 'In-Degree Centrality Based Attack', 'out_deg_centrality': 'Out-Degree Centrality Based Attack', 'closeness_centrality' :'Closeness Centrality Based Attack', 'betweeness_centrality' :'Betweenness Centrality Based Attack', 'eigenvector_centrality': 'Eigenvector Centrality Based Attack'}
    weight_prop_constant = 500
    # Get distinct settings
    acc_diff_fnames = set()
    for root, dirs, files in os.walk(dir_data):
        for fname in files:
            if fname.startswith('acc_'):
                fname_parts = re.split('_seed', fname)
                if '300' not in fname_parts[0]:
                    if '.png' not in fname_parts[0]:
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

            if 'ER' in graph_type_used:
                graph_type_name = 'ER'
            elif 'pref_attach' in graph_type_used:
                graph_type_name = 'Preferential Attachment'
            elif 'dir_geom' in graph_type_used:
                graph_type_name = 'Directed Geometric'
            elif 'out' in graph_type_used:
                graph_type_name = 'K-Out'
            elif 'SNAP' in graph_type_used:
                graph_type_name = 'SNAP Dataset'

            combined_acc = {cent:sum(acc_data) for cent, acc_data in aver_cent_data.items()}
            cent_time_eff = {cent:1 for cent in aver_cent_data.keys()}
            n_clients = re.search('_c_(\d+)_', graph_type_used)
            if n_clients is not None:
                n_clients = int(n_clients.group(1))
            cent_time_eff['in_deg_centrality'] = n_clients ** 2
            cent_time_eff['out_deg_centrality'] = n_clients ** 2
            cent_time_eff['eigenvector_centrality'] = n_clients ** 3
            if 'ER' in graph_type_used:
                prob_conn = re.search('_p_0(\d+)', graph_type_used)
                if prob_conn is not None:
                    prob_conn = float(prob_conn.group(1)) / 10
                cent_time_eff['betweeness_centrality'] = prob_conn * n_clients * (n_clients - 1) / 2 + n_clients ** 2
                cent_time_eff['closeness_centrality'] = prob_conn * n_clients * (n_clients - 1) / 2 + n_clients ** 2

            weights_cents = [float(x) / 100 for x in range(100)]
            cent_weighted_data = {cent:[combined_acc[cent] * weight + (cent_time_eff[cent] / weight_prop_constant) * (1 - weight) for weight in weights_cents] for cent in cent_time_eff.keys()}
            cent_weighted_data['none'] = [0 for x in cent_weighted_data['none']]
            print(cent_weighted_data['out_deg_centrality'])
            # Plot the accuracies
            plt.figure(figsize=(10, 6))
            if any([x == None for x in cent_weighted_data.values()]):
                continue
            for cent, tradeoff_score in cent_weighted_data.items():
                plt.plot(weights_cents, tradeoff_score, label = cent_name_dir[cent])
            plt.title('Weighted \n for %s Graph' % (graph_type_name), fontsize=16)
            plt.xlabel('Weight of accuracy sum') 
            plt.ylabel('Weighted score') 
            plt.minorticks_on()
            plt.grid(True)
            plt.ylim(0.1, plt.ylim()[-1])
            plt.xlim(0, plt.xlim()[-1])
            plt.vlines(x = 25, ymin = 0, ymax = plt.ylim()[1], colors = 'black', linestyles = 'dashed', label = 'Attack starts')
            plt.legend()
            plt.savefig(dir_graphs + graph_type_used + '_' + acc_fname + '_iid_type_%s.png' % iid_type)

            # Reset values
            cent_data = {cent:[] for cent in cent_data.keys()}
            aver_cent_data = {cent:[] for cent in cent_data.keys()}

# Function to plot averaged accuracy from multiple files
def plot_averaged_accuracy(plot_name, filenames):
    plt.figure(figsize = (10, 6))
    plt.title(plot_name)
    plt.xlabel('Epoch')
    plt.ylabel('Averaged Accuracy')

    for filename in filenames:
        epoch_accuracies = []
        with open(filename, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip the header
            for row in csv_reader:
                accuracies = eval(row[1])  # Convert string to list
                avg_accuracy = np.mean(accuracies)  # Calculate average
                epoch_accuracies.append(avg_accuracy)

        plt.plot(range(1, len(epoch_accuracies) + 1), epoch_accuracies, label=filename[55:100])

    plt.legend()
    plt.grid(True)
    plt.savefig(plot_name)

def plot_score_cent_dist_manual(dir_acc_data):
    # Initialize lists to store data for iid and non_iid
    iid_data = {'none': [], '00': [], '05': [], '010': []}
    non_iid_data = {'none': [], '00': [], '05': [], '010': []}
    # Loop through each file in the directory
    for filepath in glob.glob(os.path.join(dir_acc_data, 'acc_score_cent_dist_manual_weight_*.csv')):
        filename = os.path.basename(filepath)
        
        # Extract relevant parts from the filename
        weight, advs, adv_pow, seed, iid_type, cent = filename.split('_')[5], filename.split('_')[9], filename.split('_')[12], filename.split('_')[16], filename.split('_')[18], filename.split('_')[20].split('.')[0]
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filepath)
        
        # Drop the initial list from each epoch's data and calculate the mean
        df['mean'] = df.apply(lambda row: np.mean([float(x) for x in row[1][1:-1].split(", ") if float(x) not in eval(row[0])]), axis=1)
        
        # Add the mean data to the appropriate list based on iid_type and cent
        if iid_type == 'iid':
            iid_data[weight if cent == 'eigenvector_centrality' else 'none'].append(df['mean'].values)
        else:
            non_iid_data[weight if cent == 'eigenvector_centrality' else 'none'].append(df['mean'].values)

    plot_dir = ''
    for data, iid_type in zip(iid_data, ('iid', 'non_iid')):
        plt.figure(figsize = (10, 6))
        for weight, values in data.items():
            # Average across all seeds
            avg_values = np.mean(values, axis=0)
            plt.plot(avg_values, label=f'Weight: {weight}')
        
        plt.title(iid_type)
        plt.xlabel('Epoch')
        plt.ylabel('Average Accuracy')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f"{dir_acc_data.split('/')[-1]}_{iid_type}.png"))
        plt.close()

def plot_new_scheme(dir_acc_data):
    dir_plot_data = '../../data/full_decentralized/new_schemes_plots/'
    iid_types = ('iid', 'non_iid')
    prefix_names = ('score_cent_dist_manual_weight_00', 'score_cent_dist_manual_weight_05', 'score_cent_dist_manual_weight_010', 'cluster_metis_alg', 'random_nodes', 'least_overlap_area')
    seed_range = 50
    num_clients = 20
    num_advs = 4
    for iid_type in iid_types:
        plt.figure()
        
        for prefix_name in prefix_names:
            for cent in ['eigenvector_centrality', 'none']:
                all_seeds_data = []
                
                for seed in range(seed_range):
                    filename = f'acc_{prefix_name}_atk_FGSM_advs_{num_advs}_adv_pow_100_atk_time_25_seed_{seed}_iid_type_{iid_type}_cent_{cent}.csv'
                    filepath = os.path.join(dir_acc_data, filename)
                    
                    if not os.path.exists(filepath):
                        continue
                    
                    with open(filepath, 'r') as f:
                        reader = csv.reader(f)
                        adv_nodes = next(reader)[1]
                        if adv_nodes != '[]':
                            adv_nodes = list(map(int, adv_nodes.strip('[]').split(',')))
                        else:
                            adv_nodes = []
                        epoch_data = []
                        for row in reader:
                            accs = list(map(float, row[1].strip('[]').split(',')))
                            non_adv_accs = [accs[i] for i in range(len(accs)) if i not in adv_nodes]
                            epoch_data.append(np.mean(non_adv_accs))
                        
                        all_seeds_data.append(epoch_data)
                
                if all_seeds_data:
                    avg_data = np.mean(all_seeds_data, axis=0)
                    plt.plot(range(len(avg_data)), avg_data, label=f'{prefix_name}_{cent}')
        
        plt.xlabel('Epoch')
        plt.ylabel('Average Accuracy')
        plt.legend()
        plt.title(f"Average Accuracy vs Epoch for {iid_type} for {dir_acc_data.split('/')[-2]} graph")
        
        plot_name = f"{dir_acc_data.split('/')[-2]}_{iid_type}.png"
        plt.savefig(os.path.join(dir_plot_data, plot_name))

    
if __name__ == '__main__':
    for n_clients in [10, 25, 50, 75, 100]:
        for seed in range(50):
            for type_geom in ['2d_r_02', '2d_r_04', '2d_r_06']:
                graph_name = 'dir_geom_graph_c_%d_type_%s_seed_%d.txt' % (n_clients, type_geom, seed)
                gen_dir_geom_graph(n_clients, type_geom, graph_name, seed)
    # plot_new_scheme('../../data/full_decentralized/fmnist/ER_graph_c_20_p_05/')
    # make_graphs()    
    #for i in range(0, 11):
    #    score_graph_types_centralities_similarity('fmnist', float(i) / 10)
    # make_similarity_graphs('fmnist')
    # make_variance_histograms('fmnist')
    # x = calc_centrality_measure_aver_variance('ER_graph_c_20_p_01')
    # print(x)
    # plot_acc_aver('k_out_graph_c_20_k_15', 'fmnist')
    # file_dir = '../../data/full_decentralized/fmnist/ER_graph_c_20_p_05/' 
    # files = [file_dir + x + '.csv' for x in ('acc_score_cent_dist_manual_weight_00_atk_FGSM_advs_4_adv_pow_100_atk_time_25_seed_0_iid_type_iid_cent_eigenvector_centrality', 'acc_score_cent_dist_manual_weight_05_atk_FGSM_advs_4_adv_pow_100_atk_time_25_seed_0_iid_type_iid_cent_eigenvector_centrality', 'acc_score_cent_dist_manual_weight_010_atk_FGSM_advs_4_adv_pow_100_atk_time_25_seed_0_iid_type_iid_cent_eigenvector_centrality')]
    # plot_averaged_accuracy('New_Method_Plot_Acc_05.png', files)
    # plot_acc_aver_snap('SNAP_Cisco_c_28_type_g20', 'fmnist')
    # plot_scored_tradeoff_time_centrality('ER_graph_c_20_p_09', 'fmnist', 50)
    # calc_centrality_measure_aver_variance('dir_geom_graph_c_20_type_2d_close_nodes_seed_0.txt')


# Plot variables
# Different types of networks (ER, Geom, Pref-Attach, SNAP?) - iid and non-iid FMNIST and CIFAR10
# Different connectivity (ER 0.1 0.3 0.5 0.7, Geom r = 0.2 0.4 0.6, pref_attach -> look at parameters to change)
# Different newtork sizes (10 25 50 75 100)
# Different adversarial percentages (5, 10, 20%)



