import numpy as np
import matplotlib.pyplot as plt

# Create a gradient image
gradient = np.linspace(0, 1, 256)  # values range from 0 to 1
gradient = np.vstack((gradient, gradient))  # stack to get a 2D image

fig, ax = plt.subplots()
ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap('Reds'))
plt.axis('off')

plt.savefig('gradient.png', bbox_inches='tight', pad_inches=0)
plt.show()