import csv
import matplotlib.pyplot as plt
import numpy as np
import ast
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
        print(np.array(np.array(acc_data[0])[:, 0]).shape)
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


if __name__ == '__main__':
    plot_acc_dec_data('../../data/full_decentralized/fmnist/atk_FGSM_advs_3_adv_pow_200_clients_30_atk_time_25_arch_star_seed_0_iid_type_iid/accuracy_data_clientsb5eedee3462c80d48f214640d04d09452fe83e58.csv')
    # /root/programming/Purdue-Research-Programs-Notes/data/full_decentralized/fmnist/atk_FGSM_advs_3_adv_pow_200_clients_10_atk_time_25_arch_star_seed_0_iid_type_iid/accuracy_data_clientscc44f2e37c946852f085faaf2fd8a351a3bf5c49.csv
    # /root/programming/Purdue-Research-Programs-Notes/data/full_decentralized/fmnist/atk_FGSM_advs_3_adv_pow_200_clients_30_atk_time_25_arch_star_seed_0_iid_type_iid/accuracy_data_clientsb5eedee3462c80d48f214640d04d09452fe83e58.csv