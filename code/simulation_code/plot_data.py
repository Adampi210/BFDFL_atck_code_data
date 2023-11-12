import os
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np
import ast
from nn_FL_de_cent import *
import networkx as nx
import re
import pandas as pd
import glob

#plt.rcParams["font.family"] = "Times New Roman"
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
        '2d_r_06' : [2, 0.6],
        '2d_r_015' : [2, 0.15],
        '2d_r_01' : [2, 0.1],
        '2d_r_005' : [2, 0.05]
    }
    dim, radius = geo_graph_configs[graph_type]
    print(dim, radius)
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

def measure_avg_dist_diff_schemes(network_type):
    dir_networks = '../../data/full_decentralized/network_topologies/'
    adv_schemes = {'score_cent_dist_manual_weight_010': [], 'least_overlap_area': [], 'random_nodes': []}
    n_clients = int((re.search('_c_(\d+)', network_type)).group(1))
    adv_number = int(float(n_clients) * 0.2)
    for adv_scheme in adv_schemes.keys():
        for seed in range(50):
            graph_topology = network_type + '_seed_%d.txt' % seed
            network_topology_filepath = os.path.join(dir_networks, graph_topology)
            adj_matrix = np.loadtxt(network_topology_filepath) 
            graph_representation = create_graph(adj_matrix)
            if adv_scheme == 'score_cent_dist_manual_weight_010':
                adv_nodes = score_cent_dist_manual(1, n_clients, adv_number, graph_representation, -1)
            elif adv_scheme == 'least_overlap_area':
                adv_nodes = least_overlap_area(n_clients, adv_number, graph_representation)
            elif adv_scheme == 'random_nodes':
                adv_nodes = random_nodes(n_clients, adv_number)
            
            adv_schemes[adv_scheme].append(average_distance_between_advs(graph_representation, adv_nodes))

    for adv_scheme in adv_schemes.keys():
        adv_schemes[adv_scheme] = np.mean(adv_schemes[adv_scheme])
    print(adv_schemes)

def plot_new_schemes(network_type, iid_type, label_in_plot = 0):
    plot_labels = ('1) ', '2) ', '3) ', '4) ')
    dataset_name = 'fmnist' # Remember to change for CIFAR10
    dir_data = '../../data/full_decentralized/%s/%s/' % (dataset_name, network_type)
    dir_plots = '../../data/full_decentralized/finalized_plots/'
    adv_schemes = {'score_cent_dist_manual_weight_010': [], 'least_overlap_area': [], 'random_nodes': [], 'none': []} # All
    # adv_schemes = {'least_overlap_area': [], 'random_nodes': [], 'none': []} # Missing eigenv 
    # adv_schemes = {'score_cent_dist_manual_weight_010': [], 'least_overlap_area': [], 'none': []} # Missibg random
    n_clients = int((re.search('_c_(\d+)', network_type)).group(1))
    adv_frac = 0.2
    adv_number = int(float(n_clients) * adv_frac)
    if 'non_iid' in iid_type:
        seed_range = 20
    else:
        seed_range = 50
    pwr = 100
    # First get the data for none
    none_avail = True
    if none_avail:
        for seed in range(seed_range):
            file_name = 'acc_score_cent_dist_manual_weight_010_atk_FGSM_advs_0_adv_pow_0_atk_time_25_seed_%d_iid_type_%s_cent_none.csv' % (seed, iid_type)
            file_data_path = os.path.join(dir_data, file_name)
            with open(file_data_path, 'r') as acc_data_file:
                reader = csv.reader(acc_data_file)
                curr_seed_run = []
                for i, row in enumerate(reader):
                    if i != 0:
                        acc = ast.literal_eval(row[1])
                        acc = sum(acc) / len(acc)
                        curr_seed_run.append(acc)
                adv_schemes['none'].append(curr_seed_run)
        adv_schemes['none'] = np.mean(adv_schemes['none'], axis = 0)

    # Next do all attacks and compare
    for scheme in adv_schemes.keys():
        if scheme == 'none':
            continue
        for seed in range(seed_range):
            file_name = 'acc_%s_atk_FGSM_advs_%d_adv_pow_%d_atk_time_25_seed_%d_iid_type_%s_cent_eigenvector_centrality.csv' % (scheme, adv_number, pwr, seed, iid_type)
            file_data_path = os.path.join(dir_data, file_name)
            with open(file_data_path, 'r') as acc_data_file:
                reader = csv.reader(acc_data_file)
                curr_seed_run = []
                for i, row in enumerate(reader):
                    if i == 0:
                        attacked_nodes = ast.literal_eval(row[1])
                        attacked_nodes = [int(_) for _ in attacked_nodes]
                    else:
                        acc = ast.literal_eval(row[1])
                        acc_honest = [_ for i, _ in enumerate(acc) if i not in attacked_nodes]
                        acc_honest = sum(acc_honest) / len(acc_honest)
                        curr_seed_run.append(acc_honest)
            adv_schemes[scheme].append(curr_seed_run)
        adv_schemes[scheme] = np.mean(adv_schemes[scheme], axis = 0)

    # Finally make legend and plot
    legend = {'score_cent_dist_manual_weight_010': 'Eigenvector-Centrality Based Attack', 'least_overlap_area': 'BFDFL Attack', 'random_nodes': 'Random Choice Based Attack', 'none': 'No attack'}
    # Create the figure and axis
    fig, ax = plt.subplots(figsize = (16, 9))

    # Plot each data series
    #print(adv_schemes.items())
    for key, values in adv_schemes.items():
        ax.plot(range(len(values)), values, label = legend[key])

    # Add vertical line
    ax.axvline(x = 25, color = 'r', label = 'Attack begins')

    # Customize the plot
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    # Customize the plot to match IEEE format
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Average Accuracy for Honest Nodes', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(True)
    network_name = ''
    if 'dir_geom_graph' in network_type and n_clients != 20:
        radius = int((re.search('_r_0(\d+)', network_type)).group(1))
        network_name = 'Directed Geometric Graph'
    elif 'ER_graph' in network_type:
        probability = int((re.search('_p_0(\d+)', network_type)).group(1))
        network_name = 'ER graph'
    if 'non_iid' in iid_type:
        iid_type_title = 'Non-IID'
    else:
        iid_type_title = 'IID'
    title = '%s %s %s data' % (plot_labels[label_in_plot], network_name, iid_type_title)
    ax.set_title(title)
    # ax.legend()
    ax.grid()
    # plt.savefig(dir_plots + 'plot' + '_' + dataset_name + '_' + network_type + '_' + iid_type + '_FGSM_advs_%d_adv_pow_%d_atk_time_25_seed' % (adv_number, pwr) +'.png')
    print(title, [(i, adv_schemes[i][-1]) for i in adv_schemes.keys()])
    return adv_schemes, fig, ax, title

def create_composite_figure(graph_iid_tuples):    
    # Define the color scheme for the lines
    color_scheme = {
        'No attack': 'blue',
        'Eigenvector-Centrality Based Attack': 'green',
        'BFDFL Attack': 'black',  # Assuming this is the BLDFL attack
        'Random Choice Based Attack': 'purple',
        'Attack begins': 'red'  # Assuming this is the vertical line you mentioned
    }
    
    # Create a figure with subplots and a shared legend
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))  # Adjust the size as needed
    
    # To collect legend handles
    legend_handles = []  
    custom_legend_handles = []
    # Iterate over each graph_iid_tuple and plot in the respective subplot
    for i, (network_type, iid_type) in enumerate(graph_iid_tuples):
        # Generate each subplot
        _, fig_single, ax_single, title_plot = plot_new_schemes(network_type, iid_type, i)
        
        # Get the lines from the single plot and get the legend handles
        lines, labels = ax_single.get_legend_handles_labels()
        if i == 0:
            axs[i].set_ylabel('Average Accuracy for\nHonest Nodes %', fontsize=12)
            legend_handles.extend(lines)
            
        # Now, let's plot the same lines on the subplot axis with the specified colors
        for line in lines:
            if i == 0:
                handle = Line2D([], [], color=color_scheme[line.get_label()], label=line.get_label())
                custom_legend_handles.append(handle)
            line_color = color_scheme.get(line.get_label(), color_scheme[line.get_label()])  # Default to black if not specified
            axs[i].plot(line.get_xdata(), [100 * x for x in line.get_ydata()], label=line.get_label(), color=line_color)

        # Set the same x and y labels and title
        axs[i].set_xlabel('Global Epoch', fontsize=12)
        axs[i].set_title(title_plot, fontsize=12)
        axs[i].set_xticks([0, 20, 40, 60, 80, 100])
        # Set the same grid
        axs[i].grid(True)
        
        # Set the same limits on all subplots, if needed
        if 'Non' in title_plot:
            axs[i].set_yticks([10, 20, 30, 40, 50])
            axs[i].set_ylim([7, 41])
        else:
            axs[i].set_yticks([10, 20, 30, 40, 50])
            axs[i].set_ylim([10, 55])

    # Create a shared legend above the subplots
    order_legend = [3, 0, 2, 1, 4]
    ordered_custom_legend_handles = [custom_legend_handles[i] for i in order_legend]
    ordered_legend_handles = [legend_handles[i] for i in order_legend]
    fig.legend(ordered_custom_legend_handles, [h.get_label() for h in ordered_legend_handles], loc='upper center', bbox_to_anchor=(0.5, 1), ncol=5, fontsize=12)

    # Adjust the layout to prevent overlap, leaving space for the legend at the top
    fig.subplots_adjust(top=0.8)

    # Save the composite figure
    plt.savefig('composite_figure.png', dpi=300, bbox_inches='tight')
    plt.show()  # If you want to display the figure as well

def calculate_attack_gain_connectivity(dir_names):
    iid_types = ('iid', 'non_iid')
    attacks = ('score_cent_dist_manual_weight_010', 'least_overlap_area', 'random_nodes', 'none')
    n_bars = len(iid_types) * len(attacks)
    seed_range = 20
    dataset_name = 'fmnist' # Remember to change for CIFAR10
    dir_plots = '../../data/full_decentralized/finalized_plots/'
    pwr = 100
    colors = {'score_cent_dist_manual_weight_010': 'green', 'least_overlap_area': 'black', 'random_nodes': 'purple'}
    lighter_colors = {'score_cent_dist_manual_weight_010': 'lime', 'least_overlap_area': 'grey', 'random_nodes': 'violet'}
    attack_order = ['score_cent_dist_manual_weight_010', 'random_nodes', 'least_overlap_area']
    # Assuming dir_names contain full paths to the directories with the data
    # attacks is a list of attack names
    # iid_types is a list containing 'iid' and 'non_iid'
    # n_bars is the number of bars per group (should be len(attacks) * len(iid_types))

    # Initialize the data structure to hold the gains
    accuracies = {dir_name: {attack: {iid_type: [] for iid_type in iid_types} for attack in attacks} for dir_name in dir_names}
    gains = {dir_name: {attack: {iid_type: 0 for iid_type in iid_types} for attack in attacks} for dir_name in dir_names}

    for network in dir_names:
        dir_data = '../../data/full_decentralized/%s/%s/' % (dataset_name, network)
        for iid in iid_types:
            for seed in range(seed_range):
                file_name = 'acc_score_cent_dist_manual_weight_010_atk_FGSM_advs_0_adv_pow_0_atk_time_25_seed_%d_iid_type_%s_cent_none.csv' % (seed, iid)
                file_data_path = os.path.join(dir_data, file_name)
                with open(file_data_path, 'r') as acc_data_file:
                    reader = csv.reader(acc_data_file)
                    curr_seed_run = []
                    for i, row in enumerate(reader):
                        if i != 0:
                            acc = ast.literal_eval(row[1])
                            acc = sum(acc) / len(acc)
                            curr_seed_run.append(acc)
                    accuracies[network]['none'][iid].append(curr_seed_run)
            accuracies[network]['none'][iid] = np.mean(accuracies[network]['none'][iid], axis = 0)
    
    for network in dir_names:
        dir_data = '../../data/full_decentralized/%s/%s/' % (dataset_name, network)
        for attack in attacks:
            if attack == 'none':
                continue
            for iid in iid_types:
                for seed in range(seed_range):
                    n_clients = int((re.search('_c_(\d+)', network)).group(1))
                    adv_number = int(0.2 * n_clients)
                    file_name = 'acc_%s_atk_FGSM_advs_%d_adv_pow_%d_atk_time_25_seed_%d_iid_type_%s_cent_eigenvector_centrality.csv' % (attack, adv_number, pwr, seed, iid)
                    file_data_path = os.path.join(dir_data, file_name)
                    with open(file_data_path, 'r') as acc_data_file:
                        reader = csv.reader(acc_data_file)
                        curr_seed_run = []
                        for i, row in enumerate(reader):
                            if i == 0:
                                attacked_nodes = ast.literal_eval(row[1])
                                attacked_nodes = [int(_) for _ in attacked_nodes]
                            else:
                                acc = ast.literal_eval(row[1])
                                acc_honest = [_ for i, _ in enumerate(acc) if i not in attacked_nodes]
                                acc_honest = sum(acc_honest) / len(acc_honest)
                                curr_seed_run.append(acc_honest)
                        accuracies[network][attack][iid].append(curr_seed_run)
                accuracies[network][attack][iid] = np.mean(accuracies[network][attack][iid], axis = 0)
    for network in dir_names:
        for attack in attacks:
            for iid in iid_types:
                if attack == 'none':
                    gains[network][attack][iid] = np.sum(accuracies[network]['none'][iid][25:])
                    continue
                gains[network][attack][iid] = (np.sum(accuracies[network]['none'][iid][25:]) - np.sum(accuracies[network][attack][iid][25:])) * 100 / 75

    
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 1 / (n_bars + 1)
    index = np.arange(len(dir_names))

    # Define the legend labels
    legend_labels = [
        'Eigenvector-Centrality Based Attack IID', 
        'Eigenvector-Centrality Based Attack Non-IID', 
        'Random Choice Attack IID', 
        'Random Choice Attack Non-IID', 
        'BLDFL attack IID',
        'BLDFL attack Non-IID'
    ]

    # Create a list to hold the legend handles
    handles = []

    for i, network in enumerate(dir_names):
        for j, attack in enumerate(attack_order):
            if attack == 'none':
                continue
            for k, iid in enumerate(iid_types):
                gain = gains[network][attack][iid]
                pos = i + (j - len(attacks) / 2) * bar_width + k * bar_width * len(attacks)
                color = colors[attack] if iid == 'iid' else lighter_colors[attack]
                label = legend_labels[2 * j + k]  # Only label the first set
                bar = ax.bar(pos, gain, bar_width, label=label, color=color)
                if i == 0:  # Only add to handles on the first iteration
                    handles.append(bar)

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Averaged Attack Accuracy Loss %')

    ax.set_xticks(index + bar_width / 2)
    if 'ER_graph_' in dir_names[0]:
        ax.set_title('2) ER Graph 25 clients', fontsize=12)
        ax.set_xticklabels(['p = 0.1', 'p = 0.3', 'p = 0.5'], fontsize=12)
        ax.set_xlabel('Connection Probability', fontsize=12)
    else:
        ax.set_title('1) Directed Geometric Graph 25 clients', fontsize=12)
        ax.set_xticklabels(['r = 0.2', 'r = 0.4', 'r = 0.6'], fontsize=12)
        ax.set_xlabel('Radius', fontsize=12)
        # Create the legend
        ax.legend(handles=[h[0] for h in handles], labels=legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize='small')
    plt.tight_layout()  # Ensure that all elements of the plot fit within the figure area
    ax.grid(True)

    plt.show()
    print(gains)
    
def calculate_attack_gain_size_and_adv_percent(dir_names):
    iid_types = ('iid', 'non_iid')
    attacks = ('score_cent_dist_manual_weight_010', 'random_nodes', 'least_overlap_area', 'none')
    dev_nums = (10, 25, 50, 100)
    adv_percentages = {
        10: [int(0.1 * 10), int(0.2 * 10)],
        25: [int(0.1 * 25), int(0.2 * 25)],
        50: [int(0.06 * 50), int(0.1 * 50), int(0.2 * 50)],
        100: [int(0.05 * 100), int(0.1 * 100), int(0.2 * 100)]
    }

    n_bars = len(iid_types) * len(attacks)
    seed_range = 20
    dataset_name = 'fmnist' # Remember to change for CIFAR10
    dir_plots = '../../data/full_decentralized/finalized_plots/'
    pwr = 100
    # Define your colors
    colors = {
        'score_cent_dist_manual_weight_010': 'green', 
        'least_overlap_area': 'black', 
        'random_nodes': 'purple'
    }

    # Custom legend labels
    legend_labels = [
        'Eigenvector-Centrality Based Attack', 
        'Random Choice Attack', 
        'BLDFL Attack'
    ]

    # Create a list of Patch objects for the legend
    handles = [mpatches.Patch(color=colors[attack], label=label) for attack, label in zip(colors.keys(), legend_labels)]

    
    # Assuming dir_names contain full paths to the directories with the data
    # attacks is a list of attack names
    # iid_types is a list containing 'iid' and 'non_iid'
    # n_bars is the number of bars per group (should be len(attacks) * len(iid_types))
    accuracies = {
        dev_num: {
            attack: {
                iid: {
                    (0 if attack == 'none' else adv_percent): []
                    for adv_percent in (adv_percentages[dev_num] if attack != 'none' else [0])
                } for iid in iid_types
            } for attack in attacks
        } for dev_num in dev_nums
    }
    gains = {
        dev_num: {
            attack: {
                iid: {
                    (0 if attack == 'none' else adv_percent): 0
                    for adv_percent in (adv_percentages[dev_num] if attack != 'none' else [0])
                } for iid in iid_types
            } for attack in attacks
        } for dev_num in dev_nums
    }

    
    # Initialize the data structure to hold the gains
    
    for network in dir_names:
        dir_data = '../../data/full_decentralized/%s/%s/' % (dataset_name, network)
        n_clients_network = int((re.search('_c_(\d+)', network)).group(1))
        for iid in iid_types:
            for seed in range(seed_range):
                file_name = 'acc_score_cent_dist_manual_weight_010_atk_FGSM_advs_0_adv_pow_0_atk_time_25_seed_%d_iid_type_%s_cent_none.csv' % (seed, iid)
                file_data_path = os.path.join(dir_data, file_name)
                with open(file_data_path, 'r') as acc_data_file:
                    reader = csv.reader(acc_data_file)
                    curr_seed_run = []
                    for i, row in enumerate(reader):
                        if i != 0:
                            acc = ast.literal_eval(row[1])
                            acc = sum(acc) / len(acc)
                            curr_seed_run.append(acc)
                    accuracies[n_clients_network]['none'][iid][0].append(curr_seed_run)
            accuracies[n_clients_network]['none'][iid][0] = np.mean(accuracies[n_clients_network]['none'][iid][0], axis = 0)

    for network in dir_names:
        n_clients_network = int((re.search('_c_(\d+)', network)).group(1))
        dir_data = '../../data/full_decentralized/%s/%s/' % (dataset_name, network)
        for attack in attacks:
            if attack == 'none':
                continue
            for iid in iid_types:
                for adv_num in adv_percentages[n_clients_network]:
                    for seed in range(seed_range):
                        file_name = 'acc_%s_atk_FGSM_advs_%d_adv_pow_%d_atk_time_25_seed_%d_iid_type_%s_cent_eigenvector_centrality.csv' % (attack, adv_num, pwr, seed, iid)
                        file_data_path = os.path.join(dir_data, file_name)
                        with open(file_data_path, 'r') as acc_data_file:
                            reader = csv.reader(acc_data_file)
                            curr_seed_run = []
                            for i, row in enumerate(reader):
                                if i == 0:
                                    attacked_nodes = ast.literal_eval(row[1])
                                    attacked_nodes = [int(_) for _ in attacked_nodes]
                                else:
                                    acc = ast.literal_eval(row[1])
                                    acc_honest = [_ for i, _ in enumerate(acc) if i not in attacked_nodes]
                                    acc_honest = sum(acc_honest) / len(acc_honest)
                                    curr_seed_run.append(acc_honest)
                            accuracies[n_clients_network][attack][iid][adv_num].append(curr_seed_run)
                    accuracies[n_clients_network][attack][iid][adv_num] = np.mean(accuracies[n_clients_network][attack][iid][adv_num], axis = 0)
    
    for network in dir_names:
        dev_num = int((re.search('_c_(\d+)', network)).group(1))
        for attack in attacks:
            for iid in iid_types:
                if attack == 'none':
                    gains[dev_num]['none'][iid][0] = np.sum(accuracies[dev_num]['none'][iid][0][25:])
                    continue
                for adv_perc in adv_percentages[dev_num]:
                    gains[dev_num][attack][iid][adv_perc] = (np.sum(accuracies[dev_num]['none'][iid][0][25:]) - np.sum(accuracies[dev_num][attack][iid][adv_perc][25:])) * 100 / 75
    for i in adv_percentages.keys():
        for advs in adv_percentages[i]:
            print(f'{i}, {advs} ', end='')
            if gains[i]['random_nodes']['iid'][advs] > gains[i]['score_cent_dist_manual_weight_010']['iid'][advs]:
                x = gains[i]['random_nodes']['iid'][advs]
            else:
                x = gains[i]['score_cent_dist_manual_weight_010']['iid'][advs]
            print((gains[i]['least_overlap_area']['iid'][advs] - x) / (x) * 100)

    
    # Plot
    iid = 'iid'  # Focusing only on 'iid'
    attacks = ['score_cent_dist_manual_weight_010', 'random_nodes', 'least_overlap_area']
    bar_width = 0.2  # Width of the bars
    opacity = 0.8

    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # Adjust the size as needed
    axs = axs.flatten()  # Flatten the 2x2 array for easy indexing

    for idx, dev_num in enumerate([10, 25, 50, 100]):  # Order of dev_nums
        adv_percentages = gains[dev_num][attacks[0]][iid].keys()

        # Number of groups
        n_groups = len(adv_percentages)

        index = np.arange(n_groups)
        for i, attack in enumerate(attacks):
            gains_values = [gains[dev_num][attack][iid][adv_perc] for adv_perc in adv_percentages]
            axs[idx].bar(index + i * bar_width, gains_values, bar_width, alpha=opacity, color=colors[attack], label=attack if idx == 0 else "")
        if idx >= 2:
            axs[idx].set_xlabel('Number of Adversarial Devices', fontsize=14)
        if idx == 0 or idx == 2:
            axs[idx].set_ylabel('Averaged Attack Accuracy Loss %', fontsize=14)
        axs[idx].set_title(f'{idx + 1}) Network Size: {dev_num} Devices', fontsize=14)
               
        axs[idx].set_xticks(index + bar_width / 2)
        axs[idx].tick_params(axis='y', labelsize=12)
        axs[idx].set_xticklabels(adv_percentages, fontsize=12)

    # Create a single legend for the entire figure
    # Custom legend labels
    legend_labels = [
        'Eigenvector-Centrality Based Attack', 
        'Random Choice Attack', 
        'BLDFL Attack'
    ]

    # Create a single legend for the entire figure with custom labels
    # Adjust the legend placement
    handles = [handles[0], handles[2], handles[1]]
    fig.legend(handles, legend_labels, loc='upper center', ncol=3, fontsize='large')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make space for the legend
    plt.show()

def plot_timing_attack(dir_name):
    iid_types = ('iid', 'non_iid')
    attacks = ('score_cent_dist_manual_weight_010', 'least_overlap_area', 'random_nodes', 'none')
    n_bars = len(iid_types) * len(attacks)
    attack_times = (25, 50, 75)
    accuracies = {attack: {iid: {attack_time:[] for attack_time in attack_times} if attack != 'none' else 
                           []  for iid in iid_types} for attack in attacks}
    seed_range = 20
    dataset_name = 'fmnist' # Remember to change for CIFAR10
    pwr = 100
    dir_data = '../../data/full_decentralized/%s/%s/' % (dataset_name, dir_name)
            
    for iid in iid_types:
        for seed in range(seed_range):
            file_name = 'acc_score_cent_dist_manual_weight_010_atk_FGSM_advs_0_adv_pow_0_atk_time_25_seed_%d_iid_type_%s_cent_none.csv' % (seed, iid)
            file_data_path = os.path.join(dir_data, file_name)
            with open(file_data_path, 'r') as acc_data_file:
                reader = csv.reader(acc_data_file)
                curr_seed_run = []
                for i, row in enumerate(reader):
                    if i != 0:
                        acc = ast.literal_eval(row[1])
                        acc = sum(acc) / len(acc)
                        curr_seed_run.append(acc)
                accuracies['none'][iid].append(curr_seed_run)
        accuracies['none'][iid] = np.mean(accuracies['none'][iid], axis = 0)
        accuracies['none'][iid] = np.multiply(100, accuracies['none'][iid])

    for attack in attacks:
        if attack == 'none':
            continue
        for iid in iid_types:
            for attack_time in attack_times:
                n_clients = int((re.search('_c_(\d+)', dir_name)).group(1))
                adv_number = int(0.2 * n_clients)
                for seed in range(seed_range):
                    file_name = 'acc_%s_atk_FGSM_advs_%d_adv_pow_%d_atk_time_%d_seed_%d_iid_type_%s_cent_eigenvector_centrality.csv' % (attack, adv_number, pwr, attack_time, seed, iid)
                    file_data_path = os.path.join(dir_data, file_name)
                    with open(file_data_path, 'r') as acc_data_file:
                        reader = csv.reader(acc_data_file)
                        curr_seed_run = []
                        for i, row in enumerate(reader):
                            if i == 0:
                                attacked_nodes = ast.literal_eval(row[1])
                                attacked_nodes = [int(_) for _ in attacked_nodes]
                            else:
                                acc = ast.literal_eval(row[1])
                                acc_honest = [_ for i, _ in enumerate(acc) if i not in attacked_nodes]
                                acc_honest = sum(acc_honest) / len(acc_honest)
                                curr_seed_run.append(acc_honest)
                    accuracies[attack][iid][attack_time].append(curr_seed_run)
                accuracies[attack][iid][attack_time] = np.mean(accuracies[attack][iid][attack_time], axis = 0)
                accuracies[attack][iid][attack_time] = np.multiply(100, accuracies[attack][iid][attack_time])

    # Plot
    # Define colors for each attack
    colors = {'score_cent_dist_manual_weight_010': 'green', 'least_overlap_area': 'black', 'random_nodes': 'purple', 'none': 'blue'}
    legend = {'score_cent_dist_manual_weight_010': 'Eigenvector-Centrality Based Attack', 'least_overlap_area': 'BFDFL Attack', 'random_nodes': 'Random Choice Based Attack', 'none': 'No attack'}

    fig, axs = plt.subplots(3, 2, figsize=(10, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, attack_time in enumerate(attack_times):
        for j, iid in enumerate(iid_types):
            ax = axs[i, j]
            for attack in attacks:
                label = legend[attack]
                if attack == 'none':
                    ax.plot(accuracies[attack][iid], label=label, color=colors[attack])
                else:
                    ax.plot(accuracies[attack][iid][attack_time], label=label, color=colors[attack])
            if iid == 'iid':
                IID_name = ', IID'
            else:
                IID_name = ', Non-IID'
            # Add vertical line
            ax.axvline(x = attack_time, color = 'r', label = 'Attack begins')

            # Customize the plot
            ax.set_xticks([0, 20, 40, 60, 80, 100])
            ax.set_title('%d) Attack Time = %d%s' % ((i * 2 + j + 1), attack_time, IID_name))
            ax.grid()
            print((i * 2 + j + 1), iid)
            if iid == 'iid':
                ax.set_yticks([10, 20, 30, 40, 50])
                ax.set_ylim([10, 55])
            else:
                ax.set_yticks([10, 20, 30, 40, 50])
                ax.set_ylim([10, 41])
            if i == 2:
                ax.set_xlabel('Epoch')
            else:
                ax.set_xlabel('')
            if j == 0:
                ax.set_ylabel('Average Accuracy for\nHonest Nodes %')
            else:
                ax.set_ylabel('')

    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[0], handles[3], handles[2], handles[4], handles[1]]
    labels = [labels[0], labels[3], labels[2], labels[4], labels[1]]
    print(handles, labels)
    # First row of legend
    fig.legend(handles, labels, loc='upper center', ncol=3)
    plt.show()

if __name__ == '__main__':
    #plot_timing_attack('dir_geom_graph_c_25_type_2d_r_02')
    calculate_attack_gain_size_and_adv_percent(['dir_geom_graph_c_10_type_2d_r_02',
                           'dir_geom_graph_c_25_type_2d_r_02',
                           'dir_geom_graph_c_50_type_2d_r_02',
                           'dir_geom_graph_c_100_type_2d_r_02'])
    #calculate_attack_gain_connectivity(['dir_geom_graph_c_25_type_2d_r_02',
    #                                    'dir_geom_graph_c_25_type_2d_r_04',
    #                                    'dir_geom_graph_c_25_type_2d_r_06'])
    
    #calculate_attack_gain_connectivity(['ER_graph_c_25_p_01',
    #                                    'ER_graph_c_25_p_03',
    #                                    'ER_graph_c_25_p_05'])
    #plot_new_schemes('ER_graph_c_25_p_05', 'iid')
    #plots_baseline = [('dir_geom_graph_c_25_type_2d_r_02', 'iid'), 
    #                  ('dir_geom_graph_c_25_type_2d_r_02', 'non_iid'),
    #                  ('ER_graph_c_25_p_05', 'iid'),
    #                  ('ER_graph_c_25_p_05', 'non_iid')]
    #create_composite_figure(plots_baseline)
    #measure_avg_dist_diff_schemes('ER_graph_c_25_p_01')
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

    # Take 25 -> reduce power see if change (separation) (first)
    # Take 100 see case for 5 advs
    # 

# Plot variables
# Different types of networks (ER, Geom, Pref-Attach, SNAP?) - iid and non-iid FMNIST and CIFAR10
# Different connectivity (ER 0.1 0.3 0.5 0.7, Geom r = 0.2 0.4 0.6, pref_attach -> look at parameters to change)
# Different newtork sizes (10 25 50 75 100)
# Different adversarial percentages (5, 10, 20%)



