import csv
import matplotlib.pyplot as plt
import numpy as np

# Read parameters
seed = 0 # Seed for PRNGs 

def plot_acc_loss_data(csv_file_name):
    epochs, loss_data, acc_data = [], [], []
    with open(csv_file_name, 'r') as file_data:
        reader = csv.reader(file_data)
        for row in reader:
            epochs.append(int(row[0]) + 1)
            loss_data.append(float(row[1]))
            acc_data.append(float(row[2]))

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex = False, figsize = (18,8))

    ax1.plot(epochs, loss_data, label = 'Loss', color = 'darkblue')
    ax2.plot(epochs, acc_data, label = 'Accuracy', color = 'darkred')
    
    ax1.set_xlabel('Epoch', fontsize = 20)
    ax1.set_ylabel('Global Loss', fontsize = 20)
    ax2.set_xlabel('Epoch', fontsize = 20)
    ax2.set_ylabel('Global Accuracy', fontsize = 20) 
    ax1.set_title('Global Loss after each aggregation', fontsize = 20)
    ax2.set_title('Global Accuracy after each aggregation', fontsize = 20)
    ax1.axvline(x = 5, color = 'r', label = 'Attack Begins', lw=2.5)
    ax2.axvline(x = 5, color = 'r', label = 'Attack Begins', lw=2.5)

    ax1.grid(True)
    ax2.grid(True)
    plt.savefig(csv_file_name[:-4] + '.png')

def print_adj_matrix(npy_filenme):
    adj_matrix = np.load(npy_filenme)
    print(adj_matrix)

if __name__ == '__main__':
    