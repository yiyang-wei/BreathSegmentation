import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np

# Global variables to keep track of the current highlight and span
highlight = None
span = None
mask = None
annotation = None

def onclick(event):
    global highlight, span, mask, annotation
    if highlight:
        for spine in highlight:
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
    if span:
        span.remove()
        span = None

    # Left-click event
    if event.button == 1 and event.inaxes == top_ax:
        xdata = event.xdata
        selected = int(xdata // 0.1111)  # Assuming [0, 1] range is divided into 9 segments

        # Highlight the subplot
        row, col = divmod(selected, 3)
        highlight = list(axes[row, col, 0].spines.values()) + list(axes[row, col, 1].spines.values())
        highlight.pop(7)
        highlight.pop(2)
        for spine in highlight:
            spine.set_edgecolor('red')
            spine.set_linewidth(4)

        # Highlight the section in the top plot
        span = top_ax.axvspan(selected * 0.1111, (selected + 1) * 0.1111, color='red', alpha=0.5)

    # Right-click event
    elif event.button == 3:
        ax_clicked = event.inaxes
        if ax_clicked is not top_ax:
            pos = np.where(axes == ax_clicked)
            row, col = pos[0][0], pos[1][0]
            ax1, ax2 = axes[row, col]
            mask = patches.Rectangle((0, 0), 1, 2, transform=ax2.transAxes, color='grey', alpha=0.4, clip_on=False)
            ax2.add_patch(mask)
            annotation = ax2.annotate('Detailed Info\nLine2\nLine3\nLine4\nLine5\nLine6', xy=(0.5, 1), fontsize=12, ha='center', va='center', xycoords='axes fraction', color='white')
            fig.canvas.draw()

    plt.draw()

# Generate some sample data
t = np.linspace(0, 1, 100)
feature1 = np.sin(2 * np.pi * t)
feature2 = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(100)

fig = plt.figure(figsize=(15, 10))

# Create outer GridSpec, add one more row for the top plot
outer_grid = gridspec.GridSpec(4, 3, wspace=0.3, hspace=0.3)

# Add the top plot that spans the entire row
top_ax = plt.subplot(outer_grid[0, :])
top_ax.plot(t, feature1 + feature2, picker=True)  # Enable picking on the line
top_ax.set_title('Aggregate Data')

axes = np.empty((3, 3, 2), dtype=object)

for i in range(3):
    for j in range(3):
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                      subplot_spec=outer_grid[i + 1, j], wspace=0.1,
                                                      hspace=0.0)  # Note the change in row index

        ax1 = plt.Subplot(fig, inner_grid[0])
        ax2 = plt.Subplot(fig, inner_grid[1])

        # Plot the data
        ax1.plot(t, feature1, color='C0')
        ax2.plot(t, feature2, color='C1')

        # Color and move y-ticks and tick labels
        ax1.tick_params(axis='y', colors='C0')
        ax2.tick_params(axis='y', colors='C1', left=False, right=True, labelleft=False, labelright=True)

        # Remove y-labels to save space
        ax1.set_ylabel('')
        ax2.set_ylabel('')

        # Remove x-axis label from the upper subplot
        ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # Add subplots to figure
        fig.add_subplot(ax1)
        fig.add_subplot(ax2)

        axes[i, j, :] = ax1, ax2  # Store axes for later

# Connect the onclick event
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
