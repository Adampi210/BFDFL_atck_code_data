import matplotlib.pyplot as plt
import numpy as np

# Generate x values
x = np.linspace(0, 100, 100)

# Generate y values for the first line
y1 = 0.5 * (1 -  np.exp(-0.05 * x)) + 0.1
plt.figure(figsize=(10, 6))

# Add some noise to y1
noise = np.random.normal(0, 0.005, size=y1.shape)
y1 += noise

# Plot the first line
plt.plot(x, y1, label='No attack')
cent_name_dir = {0:'No Attack', 1: 'In-Degree Centrality Based Attack', 2: 'Out-Degree Centrality Based Attack', 3 :'Closeness Centrality Based Attack', 4 :'Betweenness Centrality Based Attack', 5: 'Eigenvector Centrality Based Attack'}

# Generate and plot the other lines
for i in range(5):
    if i <= 1:
        y = [_ if j < 25 else _ - _ * 0.03 for j, _ in enumerate(y1)]
    else:
        y = [_ if j < 25 else _ - _ * (i + 1) * 0.03 for j, _ in enumerate(y1)]
    noise = np.random.normal(0, 0.005, size=len(y))
    y += noise
    plt.plot(x, y, label=cent_name_dir[i + 1])

# Set the title and labels
plt.title('Example Model Accuracy over Epochs \n under Different Attacks', fontsize=16)
plt.xlabel('Epoch') 
plt.ylabel('Accuracy') 
plt.minorticks_on()
plt.grid(True)
plt.ylim(0.1, plt.ylim()[-1])
plt.xlim(0, plt.xlim()[-1])
plt.legend()
plt.vlines(x=25, ymin=0, ymax=plt.ylim()[1], colors='black', linestyles='dashed', label='Attack starts')

# Show the plot
plt.savefig('example_graph.png')
