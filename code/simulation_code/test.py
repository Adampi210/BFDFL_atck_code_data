import networkx as nx
import matplotlib.pyplot as plt

# Erdos-Renyi graph
seed = 0
'''
is_strongly_connected = False
while not is_strongly_connected:
        graph = nx.gnp_random_graph(n = 7, p = 0.45, seed = seed, directed = True) 
        is_strongly_connected = nx.is_strongly_connected(graph)
        print(is_strongly_connected)
        seed += 100
plt.figure(figsize=(4,4))
nx.draw_networkx(graph, with_labels=False)
plt.savefig("ERGraph.png")  # Save the figure
plt.close()  # Close the figure to free up memory

# Preferential Attachment graph
is_strongly_connected = False
while not is_strongly_connected:
        graph = nx.barabasi_albert_graph(n = 10, m = 3, seed = seed)
        seed += 100
        graph = nx.DiGraph(graph)
        is_strongly_connected = nx.is_strongly_connected(graph)
plt.figure(figsize=(4,4))
nx.draw_networkx(graph, with_labels=False)
plt.savefig("PAGraph.png")  # Save the figure
plt.close()  # Close the figure to free up memory

# Random Geometric graph
is_strongly_connected = False
while not is_strongly_connected:
        graph = nx.random_geometric_graph(n = 7, radius = 0.15, dim = 2, seed = seed) 
        graph = nx.DiGraph(graph)
        is_strongly_connected = nx.is_strongly_connected(graph)
        seed += 100
plt.figure(figsize=(4,4))
nx.draw_networkx(graph, with_labels=False)
plt.savefig("RGGGraph.png")  # Save the figure
plt.close()  # Close the figure to free up memory
'''
# k-out graph
is_strongly_connected = False
seed = 0
while not is_strongly_connected:
        graph = nx.random_k_out_graph(n = 7, k = 3, alpha = 1, self_loops = False, seed = seed)
        seed += 100
        is_strongly_connected = nx.is_strongly_connected(graph)
plt.figure(figsize=(4,4))
nx.draw_networkx(graph, with_labels=False)
plt.savefig("KOutGraph.png")  # Save the figure
plt.close()  # Close the figure to free up memory
