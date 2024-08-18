import networkx as nx
import matplotlib.pyplot as plt
from plot_data import *

def generate_graphs(num_nodes):

    
    # Create DG graph
    dg_graph = gen_dir_geom_graph(num_nodes)[0]

    
    # Plot the graphs
    fig, ax1= plt.subplots(1, 1, figsize=(7, 7))
    pos_dg = nx.spring_layout(dg_graph)
    nx.draw(dg_graph, pos_dg, ax=ax1, node_color='royalblue', node_size=500, with_labels=False, arrows=True)

    
    plt.tight_layout()
    plt.savefig('graphs.png')

# Generate graphs with 7 nodes (you can change this number)
generate_graphs(10)