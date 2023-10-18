import numpy as np
import matplotlib.pyplot as plt

# Generate sample time series data
time = np.linspace(0, 90, 900)
signal = np.sin(time)

def on_click(event):
    """Handle click events on subplots."""
    for i in range(3):
        for j in range(3):
            if subplot_axes[i][j].contains(event)[0]:
                if selected_breath[i][j] == 0:
                    for spine in subplot_axes[i][j].spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(2)
                    selected_breath[i][j] = 1
                else:
                    for spine in subplot_axes[i][j].spines.values():
                        spine.set_edgecolor('black')
                        spine.set_linewidth(0.5)
                    selected_breath[i][j] = 0
    fig.canvas.draw()

def on_key(event):
    """Handle key press events."""
    if event.key == 'enter':
        plt.close()

# Create main figure and axes
fig = plt.figure(figsize=(10, 15))

# Create the top plot for the whole time series
top_ax = plt.subplot2grid((4, 3), (0, 0), colspan=3)
top_ax.plot(time, signal)
top_ax.set_title("Complete Time Series")

# Create the 3x3 grid for the subplots
subplot_axes = [[None for _ in range(3)] for _ in range(3)]
selected_breath = [[0 for _ in range(3)] for _ in range(3)]
for i in range(3):
    for j in range(3):
        start_idx = (i * 3 + j) * 100
        end_idx = start_idx + 100

        ax = plt.subplot2grid((4, 3), (i+1, j))
        ax.plot(time[start_idx:end_idx], signal[start_idx:end_idx])
        ax.set_title(f"Segment {i*3 + j + 1}")
        subplot_axes[i][j] = ax

# Connect event handlers
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)

plt.tight_layout()
plt.show()
