import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import ast
from nn_FL_1 import *

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
                    attacked_nodes = ast.literal_eval(row_attck[1])
                    attacked_nodes = [int(_) for _ in attacked_nodes]
                elif i > 25:
                    acc_orig = ast.literal_eval(row_orig[1])
                    acc_orig = sum(acc_orig) / len(acc_orig)
                    acc_attck = ast.literal_eval(row_attck[1])
                    acc_honest = [_ for i, _ in enumerate(acc_attck) if i not in attacked_nodes]
                    acc_honest = sum(acc_honest) / len(acc_honest)
                    total_diff += acc_orig - acc_honest
                    print(acc_orig, acc_honest)
                i += 1

        return total_diff

def plot_acc_diff(dataset_name = 'fmnist'):
    dir_networks = '/root/programming/Purdue-Research-Programs-Notes/data/full_decentralized/network_topologies'
    dir_data = '/root/programming/Purdue-Research-Programs-Notes/data/full_decentralized/%s/' % dataset_name
    for network_topology in os.listdir(dir_networks):
        file_network_toplogy = os.path.join(dir_networks, network_topology)
        adj_matrix = np.loadtxt(file_network_toplogy)
        hash_adj_matrix = hash_np_arr(adj_matrix)
        data_dir_name = dir_data + str(hash_adj_matrix) + '/' 
        print(data_dir_name)
if __name__ == '__main__':
    plot_acc_diff()