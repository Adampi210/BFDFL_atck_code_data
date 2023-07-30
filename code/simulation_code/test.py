import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# Define your color map
cmap = plt.get_cmap('tab10')

# Generate colors from the color map
colors = [cmap(i) for i in range(7)]
label_arr = ['In-Degree Centrality', 'Out-Degree Centrality', 'Closeness Centrality', 'Betweenness Centrality', 'Eigenvector Centrality']  # replace with your actual labels

# Create a list of patches to add to the legend
patches = [mpatches.Patch(color=colors[0], label='No Attack'),
           mpatches.Patch(color=colors[1], label='In-Degree Centrality Based Attack'),
           mpatches.Patch(color=colors[2], label='Out-Degree Centrality Based Attack'),
           mpatches.Patch(color=colors[3], label='Closeness Centrality Based Attack'),
           mpatches.Patch(color=colors[4], label='Betweenness Centrality Based Attack'),
           mpatches.Patch(color=colors[5], label='Eigenvector Centrality Based Attack'),
           mpatches.Patch(color=colors[6], label='Attack Begins')]

# Create a new figure for the legend
legend_fig = plt.figure(figsize=(10, 1))  # Adjust as needed
legend_ax = legend_fig.add_subplot(111)

# Add the legend to the figure
legend_ax.legend(handles=patches, ncol=len(patches), loc='center')

# Hide the axes
legend_ax.axis('off')

# Save the figure
legend_fig.savefig('legend.png', dpi=300, bbox_inches='tight')