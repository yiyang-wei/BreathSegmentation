"""
1. Setup on the top
2. Print the setup for reference
3. Check if provided file is already partially labeled
4. Start from the last labeled page
5. Button to adjust label offset
6. Button to adjust default label

"""

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from enum import Enum
import os

# Fix macOS issue
import matplotlib
matplotlib.use('TkAgg')

# do not include slash at the end
READ_FOLDER = "data"
SAVE_FOLDER = "breathlabels"
IN_FILE_NAME = "20190616_EVLP551_converted.csv"
OUT_FILE_NAME = IN_FILE_NAME[:-13] + "labeled.csv"

# subplots layout in each page
ROWS = 4
COLS = 5

# window size
WIDTH = 16
HEIGHT = 9


class BreathLabel(Enum):
    def __new__(cls, value, color):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.color = color
        return obj

    Normal = (1, "black")
    Assessment = (2, "green")
    Bronch = (3, "blue")
    Deflation = (4, "violet")
    Question = (5, "orange")
    InPause = (6, "gold")
    ExPause = (7, "indigo")
    Noise = (8, "red")


N_LABELS = len(BreathLabel)


def quick_filter(duration):
    """Quickly filter the breaths based on their duration."""
    if duration < 4 or duration > 10:
        return BreathLabel.Noise.value
    elif duration < 7:
        return BreathLabel.Assessment.value
    else:
        return BreathLabel.Normal.value


READ_PATH = os.path.join(READ_FOLDER, IN_FILE_NAME)
SAVE_PATH = os.path.join(SAVE_FOLDER, OUT_FILE_NAME)


class BreathLoader:

    def __init__(self, in_file_path):
        """Read data from a CSV file and segment each breath."""
        try:
            print(f"Reading data from {in_file_path}")
            breath = pd.read_csv(in_file_path, header=0, sep=',', usecols=["Timestamp", "Pressure", "Flow", "B_phase"])
            # extract the columns as numpy arrays
            self.timestamp = breath['Timestamp'].to_numpy()
            self.pressure = breath['Pressure'].to_numpy()
            self.flow = breath['Flow'].to_numpy()
            self.phase = breath['B_phase'].to_numpy()
            # segment each breath
            diff_phase = np.diff(self.phase)
            self.boundary = np.where(diff_phase == 1)[0] + 1
            self.switch = np.where(diff_phase == -1)[0] + 1
            self.n_breaths = self.boundary.shape[0] - 1
            print(f"Successfully loaded {self.n_breaths} breaths from {in_file_path}")
        except Exception as e:
            print(f"Error reading file: {e}")
            exit()

    def get_breath_boundry(self, start_breath_index, end_breath_index=None):
        """Get the breaths in the given range."""
        if end_breath_index is None:
            end_breath_index = start_breath_index
        start_breath_index = max(start_breath_index, 0)
        end_breath_index = min(end_breath_index, self.n_breaths - 1)
        start_idx = self.boundary[start_breath_index] + 1
        end_idx = self.boundary[end_breath_index + 1] + 1
        return start_idx, end_idx

    def get_volume(self, breath_index):
        """Get the volume of the breath at the given index."""
        start_idx, end_idx = self.get_breath_boundry(breath_index)
        mid_idx = self.switch[breath_index] + 1
        return np.sum(self.flow[start_idx:mid_idx]) / 1000, np.sum(self.flow[mid_idx:end_idx]) / 1000

    def get_breath_data(self, start_breath_index, end_breath_index=None):
        """Get the breaths in the given range."""
        start_idx, end_idx = self.get_breath_boundry(start_breath_index, end_breath_index)
        return self.timestamp[start_idx:end_idx], self.pressure[start_idx:end_idx], self.flow[start_idx:end_idx]


class LabelRecorder:

    def __init__(self, out_file_path):
        """Create a file if not exist or read the labels if exists."""
        self.out_file_path = out_file_path
        try:
            print(f"Preparing save directory {out_file_path}")
            # create the folder if not exist
            os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
            # read the labels from the csv file if exists or create a new one
            if os.path.exists(out_file_path):
                print(f"Past record exists, reading labels from {out_file_path}")
                self.df = pd.read_csv(out_file_path, header=0, sep=',')
                label_names = self.df["Label"]
                # convert the label names to label values
                self.breath_labels = self.to_values(label_names)
            else:
                self.df = None
                self.breath_labels = None
        except Exception as e:
            print(f"Error preparing save directory: {e}")
            exit()

    def to_values(self, label_names):
        """Convert the label names to label values."""
        return np.array([BreathLabel[label_name].value for label_name in label_names])

    def to_names(self, label_values):
        """Convert the label values to label names."""
        return np.array([BreathLabel(label_value).name for label_value in label_values])

    def save(self):
        """Save the labels to a CSV file."""
        self.df["Label"] = self.to_names(self.breath_labels)
        try:
            # save the df to a csv file
            self.df.to_csv(self.out_file_path, index=False)
        except Exception as e:
            print(f"Error saving file: {e}")


class LabelerUI:

    def __init__(self, breath_data, rows, cols, width, height):
        self.breath_data = breath_data
        self.rows = rows
        self.cols = cols
        self.total = rows * cols
        self.fig = self.fig = plt.figure(figsize=(width, height))

        # Create outer GridSpec, add one more row for the top plot
        self.outer_grid = gridspec.GridSpec(rows + 1, cols,
                                            wspace=0.25, hspace=0.5,
                                            left=0.05, right=0.95, bottom=0.05, top=0.95)

        # Add the top plot that spans the entire row
        self.top_ax = plt.subplot(self.outer_grid[0, :])
        self.top_ax.tick_params(axis='y', colors='C0', rotation=90)
        self.top_ax.set_title('Entire Flow')

        # Create the axes for the subplots
        self.sub_axes = np.empty((rows, cols, 2), dtype=object)

        # Temp walk around
        self.offset = 1

        for r in range(rows):
            for c in range(cols):
                inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=self.outer_grid[r + 1, c],
                                                              wspace=0.1, hspace=0.0)

                ax1 = plt.Subplot(self.fig, inner_grid[0])
                ax2 = plt.Subplot(self.fig, inner_grid[1])

                # Color and move y-ticks and tick labels
                ax1.tick_params(axis='y', colors='C1', rotation=90, labelsize=9)
                ax2.tick_params(axis='y', colors='C0', rotation=90, labelsize=9,
                                left=False, right=True, labelleft=False, labelright=True)

                # Remove y-labels to save space
                ax1.set_ylabel('')
                ax2.set_ylabel('')

                # Remove x-axis label from the upper subplot
                ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

                # Add subplots to figure
                self.fig.add_subplot(ax1)
                self.fig.add_subplot(ax2)

                # Store axes for later
                self.sub_axes[r, c, :] = ax1, ax2

    def update(self, breath_index):

        # Create the top plot for the whole time series
        self.top_ax.plot(self.breath_data.timestamp, self.breath_data.flow, linewidth=0.5)
        self.top_ax.set_yticks([np.min(self.breath_data.flow), np.max(self.breath_data.flow)])
        window_start, window_end = self.breath_data.get_breath_boundry(breath_index, breath_index + self.total - 1)
        self.top_ax.axvspan(self.breath_data.timestamp[window_start], self.breath_data.timestamp[window_end], alpha=0.2, color='grey')

        for r in range(self.rows):
            for c in range(self.cols):
                i = r * self.cols + c
                if breath_index + i >= self.breath_data.n_breaths:
                    break
                part_timestamp, part_pressure, part_flow = self.breath_data.get_breath_data(breath_index + i)
                ax1, ax2 = self.sub_axes[r, c, :]

                # Clear the axes
                ax1.clear()
                ax2.clear()

                # Create ax1 for pressure and ax2 for flow
                ax1.plot(part_timestamp, part_pressure, '.-', color='C1', alpha=0.5)
                ax2.plot(part_timestamp, part_flow, '.-', color='C0', alpha=0.5)

                # Calculate and display the in_vol and ex_vol on the plot
                # in_vol, ex_vol = self.breath_data.get_volume(breath_index + i)
                # ax2.text(0.05, 0.1, f"In: {in_vol:.2f}", transform=ax2.transAxes, color="forestgreen",
                #          bbox=dict(facecolor='ghostwhite', edgecolor='silver', boxstyle='round', alpha=0.5, pad=0.2))
                # ax2.text(0.65, 0.1, f"Ex: {ex_vol:.2f}", transform=ax2.transAxes, color="orangered",
                #          bbox=dict(facecolor='ghostwhite', edgecolor='silver', boxstyle='round', alpha=0.5, pad=0.2))

                # # Calculate the margins for pressure and flow based on their respective ranges
                # margin_factor = 0.15
                # pressure_margin = margin_factor * (part_pressure.max() - part_pressure.min())
                # flow_margin = margin_factor * (part_flow.max() - part_flow.min())
                #
                # # Adjust y-limits with margins for both top and bottom of the plots
                # ax1.set_ylim(2 * part_pressure.min() - part_pressure.max() - 2 * pressure_margin,
                #              part_pressure.max() + pressure_margin)
                # ax2.set_ylim(part_flow.min() - flow_margin, 2 * part_flow.max() - part_flow.min() + 2 * flow_margin)

                # don't show x ticks, show timestamp difference as label
                duration = (part_timestamp[-1] - part_timestamp[0]) / 10000
                ax2.set_xlabel(f"start: {part_timestamp[0]} duration: {duration:.2f}s")

                # show only min and max pressure
                ax1.set_yticks([np.min(part_pressure), np.max(part_pressure)])

                # show only min and max flow
                ax2.set_yticks([np.min(part_flow), np.max(part_flow)])

                # set the title of each plot to the index of the breath
                ax1.set_title(f"Breath {self.offset + breath_index + i}")

                # setup plot border color and width
                self.sub_axes[r, c, :] = ax1, ax2
                # self.update_plot(self.subplot_axes[i], self.breath_labels[self.breath_index + i],
                #                  self.offset + breath_index + i)

        # Connect event handlers
        # self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        # self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Show the plot in the top left corner of the screen
        plt.get_current_fig_manager().window.wm_geometry("+0+0")
        plt.show()


breath_loader = BreathLoader(READ_PATH)
ui = LabelerUI(breath_loader, ROWS, COLS, WIDTH, HEIGHT)
ui.update(0)