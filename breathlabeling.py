import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

READ_PATH = "data/20220418_EVLP818_converted.csv"

SAVE_PATH = "data/labels.csv"

rows = 4
cols = 5

labels = {0: {"label": "", "color": "black"},
          1: {"label": "Assessment", "color": "green"},
          2: {"label": "Bronch", "color": "blue"},
          3: {"label": "Deflation", "color": "purple"},
          4: {"label": "Noise", "color": "red"}}

n_labels = len(labels)


def quick_filter(duration):
    """Quickly filter the breaths based on their duration."""
    if duration < 4 or duration > 10:
        return 3
    elif duration < 7:
        return 1
    else:
        return 0


def read_data(file_path):
    """Read data from a CSV file and return preprocess data."""
    try:
        breath = pd.read_csv(file_path, header=0, sep=',', usecols=["Timestamp", "Pressure", "Flow", "B_phase"])
        # extract the columns as numpy arrays
        timestamp = breath['Timestamp'].to_numpy()
        pressure = breath['Pressure'].to_numpy()
        flow = breath['Flow'].to_numpy()
        phase = breath['B_phase'].to_numpy()
        # segment each breath
        phase_diff = np.diff(phase)
        boundry = np.where(phase_diff == 1)[0]
        return timestamp, pressure, flow, phase, boundry
    except Exception as e:
        print(f"Error reading file: {e}")
        exit()


timestamp, pressure, flow, phase, boundry = read_data(READ_PATH)

n_breaths = boundry.shape[0] - 1
print(f"Total number of breaths: {n_breaths}")

breath_labels = np.zeros(n_breaths, dtype=np.int8)

# print the timestamps of the first five breath
for i in range(5):
    print(f"Breath {i} starts and ends at [{timestamp[boundry[i]]}, {timestamp[boundry[i+1]-1]}] with row index [{boundry[i]}, {boundry[i+1]-1}]")

# input the offset of the breath index
offset = int(input("Enter the offset to add on breath index: "))


subplot_axes = [(None, None)] * (rows * cols)
selected_breath = np.zeros(rows * cols)


def update_plot(axes, label, idx):
    """Update the plot of the i-th breath."""
    if label == 0:
        axes[0].set_title(f"Breath {idx}")
    else:
        axes[0].set_title(f"Breath {idx} ({labels[label]['label']})")
    axes[0].title.set_color(labels[label]['color'])
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor(labels[label]['color'])
            if label == 0:
                spine.set_linewidth(0.5)
            else:
                spine.set_linewidth(4)

def on_click(event):
    """Handle click events on subplots."""
    for i in range(rows * cols):
        if subplot_axes[i][0].contains(event)[0]:
            if event.button == 1:
                selected_breath[i] = n_labels - 1 if selected_breath[i] != n_labels - 1 else 0
            else:
                selected_breath[i] = (selected_breath[i] + 1) % (n_labels - 1)
            update_plot(subplot_axes[i], selected_breath[i], offset + index + i)
            fig.canvas.draw()
            break

def on_key(event):
    """Handle key press events."""
    if event.key == 'backspace':
        for i in range(rows * cols):
            selected_breath[i] = 0
            update_plot(subplot_axes[i], selected_breath[i], offset + index + i)
        fig.canvas.draw()
    elif event.key == ' ':
        for i in range(rows * cols):
            if selected_breath[i] != n_labels-1:
                selected_breath[i] = (selected_breath[i] + 1) % (n_labels - 1)
                update_plot(subplot_axes[i], selected_breath[i], offset + index + i)
        fig.canvas.draw()
    elif event.key == 'enter':
        plt.close()

index = 0
while index < n_breaths:
    fig = plt.figure(figsize=(15, 10))

    # Create the top plot for the whole time series
    top_ax = plt.subplot2grid((rows + 1, cols), (0, 0), colspan=cols)
    top_ax.plot(timestamp, flow, linewidth=0.5)
    top_ax.set_yticks([np.min(flow), np.max(flow)])
    top_ax.tick_params(axis='y', rotation=90)
    top_ax.axvspan(timestamp[boundry[index]], timestamp[boundry[min(index + rows * cols, n_breaths)]], alpha=0.2, color='grey')
    top_ax.set_title("Flow")

    for i in range(rows * cols):
        start_idx = boundry[offset + index + i]
        end_idx = boundry[offset + index + i + 1]

        part_timestamp = timestamp[start_idx:end_idx]
        part_pressure = pressure[start_idx:end_idx]
        part_flow = flow[start_idx:end_idx]

        # Create ax1 for pressure
        ax1 = plt.subplot(rows + 1, cols, i + cols + 1)
        ax1.plot(part_timestamp, part_pressure, '.-', color='C1', alpha=0.5)

        # Create ax2 (twin axis of ax1) for flow
        ax2 = ax1.twinx()
        ax2.plot(part_timestamp, part_flow, '.-', color='C0', alpha=0.5)

        # Calculate the margins for pressure and flow based on their respective ranges
        margin_factor = 0.15
        pressure_margin = margin_factor * (part_pressure.max() - part_pressure.min())
        flow_margin = margin_factor * (part_flow.max() - part_flow.min())

        # Adjust y-limits with margins for both top and bottom of the plots
        ax1.set_ylim(2 * part_pressure.min() - part_pressure.max() - 2*pressure_margin,
                     part_pressure.max() + pressure_margin)
        ax2.set_ylim(part_flow.min() - flow_margin, 2 * part_flow.max() - part_flow.min() + 2*flow_margin)

        # don't show x ticks, show timestamp difference as label
        duration = (part_timestamp[-1] - part_timestamp[0]) / 10000
        ax1.set_xticks([])
        ax1.set_xlabel(f"{duration:.2f} s")
        # show only min and max pressure
        ax1.set_yticks([np.min(pressure[start_idx:end_idx]), np.max(pressure[start_idx:end_idx])])
        ax1.tick_params(axis='y', colors='C1')
        # show only min and max flow
        ax2.set_yticks([np.min(flow[start_idx:end_idx]), np.max(flow[start_idx:end_idx])])
        ax2.tick_params(axis='y', colors='C0')
        # set the title of each plot to the index of the breath
        ax1.set_title(f"Breath {offset + index + i}")
        # make y axis number vertical
        ax1.tick_params(axis='y', rotation=90)
        ax2.tick_params(axis='y', rotation=90)
        # setup plot border color and width
        selected_breath[i] = quick_filter(duration)
        subplot_axes[i] = ax1, ax2
        update_plot(subplot_axes[i], selected_breath[i], offset + index + i)
        if index + i >= n_breaths - 1:
            break

    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    # Show the plot in the top left corner of the screen
    plt.get_current_fig_manager().window.wm_geometry("+0+0")
    plt.show()
    for i in range(rows * cols):
        if selected_breath[i] != 0:
            print(f"Breath {offset + index + i} is labeled as {labels[selected_breath[i]]['label']}")
            breath_labels[index + i] = selected_breath[i]
            selected_breath[i] = 0
    index += rows * cols


def save_labels(file_path, labels):
    """Save labels to a csv file."""
    # create the folder if not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # save the labels to a csv file
    np.savetxt(file_path, labels, delimiter=",", fmt="%d")


# save the labels to a csv file
save_labels("data/labels.csv", breath_labels)