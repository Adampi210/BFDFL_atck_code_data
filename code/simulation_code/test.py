import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# Define your color map
cmap = plt.get_cmap('tab10')

# Generate colors from the color map
colors = [cmap(i) for i in range(7)]
label_arr = ['In-Degree Centrality', 'Out-Degree Centrality', 'Closeness Centrality', 'Betweenness Centrality', 'Eigenvector Centrality']  # replace with your actual labels

# Create a list of patches to add to the legend
patches1 = [mpatches.Patch(color=colors[0], label='No Attack'),
           mpatches.Patch(color=colors[1], label='In-Degree Centrality Based Attack'),
           mpatches.Patch(color=colors[2], label='Out-Degree Centrality Based Attack'),
           mpatches.Patch(color=colors[3], label='Closeness Centrality Based Attack'),
           mpatches.Patch(color=colors[4], label='Betweenness Centrality Based Attack'),
           mpatches.Patch(color=colors[5], label='Eigenvector Centrality Based Attack')]

patches2 = [mpatches.Patch(color='black', label='Attack Begins')]

# Create a new figure for the legend
legend_fig = plt.figure(figsize=(12, 1))  # Adjust as needed
legend_ax = legend_fig.add_subplot(111)

# Add the legends to the figure
legend1 = legend_ax.legend(handles=patches1, ncol=3, loc='upper center')
legend_ax.add_artist(legend1)
legend_ax.legend(handles=patches2, ncol=1, loc='lower center', bbox_to_anchor = (0.5, -0.1))

# Hide the axes
legend_ax.axis('off')

# Save the figure
legend_fig.savefig('legend_1.png', dpi=300, bbox_inches='tight')
