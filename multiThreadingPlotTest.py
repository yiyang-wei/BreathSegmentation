import matplotlib.pyplot as plt
import numpy as np
import threading

# Function to plot on a subplot
def plot_subplots(ax, i, j):
    x = np.linspace(0, 10, 100)
    y = np.sin((i + j) * x)
    ax[i, j].plot(x, y)
    ax[i, j].set_title(f'Plot {i+1}-{j+1}')

# Initialize 5x5 subplots
fig, ax = plt.subplots(5, 5, figsize=(10, 10))

# Create a lock for thread-safety
plot_lock = threading.Lock()

# Create and start threads
threads = []
for i in range(5):
    for j in range(5):
        t = threading.Thread(target=plot_subplots, args=(ax, i, j))
        threads.append(t)
        t.start()

# Wait for all threads to complete
for t in threads:
    t.join()

# Show the final plot
plt.tight_layout()
plt.show()
