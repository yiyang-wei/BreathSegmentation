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
import matplotlib.patches as patches
from enum import Enum
import os

# Fix macOS issue
import matplotlib
matplotlib.use('TkAgg')

# do not include slash at the end
READ_FOLDER = "../ventilator converted files"
SAVE_FOLDER = "../ventilator breathlabels"
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

    Unvisited = (0, "black")
    Normal = (1, "black")
    Assessment = (2, "green")
    Bronch = (3, "blue")
    Deflation = (4, "violet")
    Question = (5, "orange")
    InPause = (6, "gold")
    ExPause = (7, "indigo")
    Noise = (8, "red")


N_LABELS = len(BreathLabel)


def quick_filter(duration, in_vol, ex_vol):
    """Quickly filter the breaths based on their duration."""
    if duration < 4 or duration > 10:
        return BreathLabel.Noise.value
    elif duration < 7:
        return BreathLabel.Assessment.value
    elif abs(in_vol + ex_vol) > 0.1 * abs(in_vol - ex_vol):
        return BreathLabel.Bronch.value
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

    def __init__(self, out_file_path, breath_data):
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
                self.breath_labels = np.zeros(breath_data.n_breaths, dtype=np.int8)
                self.df = pd.DataFrame()
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

    def __init__(self, breath_data, breath_recorder, rows, cols, width, height):
        self.breath_data = breath_data
        self.breath_recorder = breath_recorder
        self.rows = rows
        self.cols = cols
        self.total = rows * cols
        self.fig = self.fig = plt.figure(figsize=(width, height))

        # Create outer GridSpec, add one more row for the top plot
        self.outer_grid = gridspec.GridSpec(rows + 1, cols,
                                            wspace=0.25, hspace=0.5,
                                            left=0.05, right=0.95, bottom=0.05, top=0.95)

        self.top_ax = None
        self.init_top_ax()

        # Create the axes for the subplots
        self.sub_axes = np.empty((rows, cols, 2), dtype=object)
        self.init_sub_axes()

        self.info = np.empty((rows, cols), dtype=object)

        self.start_breath_index = 0

        self.span = None
        self.mask = np.empty((rows, cols), dtype=object)
        self.annotation = np.empty((rows, cols), dtype=object)

        # Temp walk around
        self.offset = 1

    def init_top_ax(self):
        # Add the top plot that spans the entire row
        self.top_ax = plt.subplot(self.outer_grid[0, :])
        self.top_ax.plot(self.breath_data.timestamp, self.breath_data.flow, linewidth=0.5)
        self.top_ax.set_yticks([np.min(self.breath_data.flow), np.max(self.breath_data.flow)])
        self.top_ax.tick_params(axis='y', colors='C0', rotation=90)
        self.top_ax.set_title('Entire Flow')

    def init_sub_axes(self):
        for r in range(self.rows):
            for c in range(self.cols):
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

    def get_breath_index(self, i):
        return self.start_breath_index + i

    def get_breath_number(self, i):
        return self.offset + self.get_breath_index(i)

    def get_breath_label(self, i):
        return self.breath_recorder.breath_labels[self.get_breath_index(i)]

    def set_breath_label(self, i, label):
        self.breath_recorder.breath_labels[self.get_breath_index(i)] = label

    def get_axes(self, i):
        return self.sub_axes[i // self.cols, i % self.cols, :]

    def find_rc(self, ax):
        if ax is None or ax is self.top_ax:
            return None, None
        pos = np.where(self.sub_axes == ax)
        r, c = pos[0][0], pos[1][0]
        return r, c

    def mark_subplot(self, i):
        """Update the plot of the i-th breath."""
        ax1, ax2 = self.get_axes(i)
        label = self.get_breath_label(i)
        if label <= 1:
            ax1.set_title(f"Breath {self.get_breath_number(i)}")
        else:
            ax1.set_title(f"Breath {self.get_breath_number(i)} ({BreathLabel(label).name})")
        ax1.title.set_color(BreathLabel(label).color)
        spines = list(ax1.spines.values()) + list(ax2.spines.values())
        spines.pop(7)
        spines.pop(2)
        for spine in spines:
            spine.set_edgecolor(BreathLabel(label).color)
            if label <= 1:
                spine.set_linewidth(1)
            else:
                spine.set_linewidth(4)

    def on_click(self, event):
        """Handle the click event."""
        ax_clicked = event.inaxes
        if event.button == 1:
            if ax_clicked == self.top_ax:
                xdata = event.xdata
                # find the index of the clicked point
                clicked_index = np.searchsorted(self.breath_data.timestamp, xdata, side='left')
                # find the index of the breath that contains the clicked point
                breath_index = np.searchsorted(self.breath_data.boundary, clicked_index, side='right') - 1
                # get the page start index of the breath
                page_index = breath_index // self.total
                # update the plot
                self.update(page_index * self.total)
            else:
                r, c = self.find_rc(ax_clicked)
                if r is None or c is None:
                    return
                i = r * self.cols + c
                ax1, ax2 = self.sub_axes[r, c, :]
                label = self.get_breath_label(i)
                self.set_breath_label(i, (label + 1) % N_LABELS)
                self.mark_subplot(i)
                self.fig.canvas.draw()
        elif event.button == 3:
            if ax_clicked is not self.top_ax:
                r, c = self.find_rc(ax_clicked)
                if r is None or c is None:
                    return
                if self.mask[r, c] is not None:
                    self.mask[r, c].remove()
                    self.mask[r, c] = None
                    self.annotation[r, c].remove()
                    self.annotation[r, c] = None
                else:
                    ax1, ax2 = self.sub_axes[r, c, :]
                    self.mask[r, c] = patches.Rectangle((0, 0), 1, 2, transform=ax2.transAxes, color='grey', alpha=0.75, clip_on=False)
                    ax2.add_patch(self.mask[r,c])
                    self.annotation[r, c] = ax2.annotate(self.info[r, c], xy=(0.5, 1), fontsize=12,
                                              ha='center', va='center', xycoords='axes fraction', color='white')
                self.fig.canvas.draw()
    def update(self, start_breath_index):
        self.start_breath_index = start_breath_index

        # Create the top plot for the whole time series
        window_start, window_end = self.breath_data.get_breath_boundry(start_breath_index, start_breath_index + self.total - 1)
        if self.span is not None:
            self.span.remove()
        self.span = self.top_ax.axvspan(self.breath_data.timestamp[window_start], self.breath_data.timestamp[window_end], alpha=0.2, color='grey')

        for r in range(self.rows):
            for c in range(self.cols):
                i = r * self.cols + c
                ax1, ax2 = self.sub_axes[r, c, :]

                # Clear the axes
                ax1.clear()
                ax2.clear()

                if start_breath_index + i >= self.breath_data.n_breaths:
                    continue

                part_timestamp, part_pressure, part_flow = self.breath_data.get_breath_data(start_breath_index + i)

                # Create ax1 for pressure and ax2 for flow
                ax1.plot(part_timestamp, part_pressure, '.-', color='C1', alpha=0.5)
                ax2.plot(part_timestamp, part_flow, '.-', color='C0', alpha=0.5)

                # Calculate and display the in_vol and ex_vol on the plot
                in_vol, ex_vol = self.breath_data.get_volume(start_breath_index + i)
                ax2.text(0.02, 0.12, f"In: {in_vol:.2f}", transform=ax2.transAxes, color="forestgreen",
                         bbox=dict(facecolor='ghostwhite', edgecolor='silver', boxstyle='round', alpha=0.5, pad=0.2))
                ax2.text(0.65, 0.12, f"Ex: {ex_vol:.2f}", transform=ax2.transAxes, color="orangered",
                         bbox=dict(facecolor='ghostwhite', edgecolor='silver', boxstyle='round', alpha=0.5, pad=0.2))

                self.info[r, c] = f"In: {in_vol:.2f}\nEx {ex_vol:.2f}"

                # Calculate the margins for pressure and flow based on their respective ranges
                margin_factor = 0.15
                pressure_margin = margin_factor * (part_pressure.max() - part_pressure.min())
                flow_margin = margin_factor * (part_flow.max() - part_flow.min())
                # Adjust y-limits with margins for both top and bottom of the plots
                ax1.set_ylim(part_pressure.min() -  pressure_margin, part_pressure.max() + pressure_margin)
                ax2.set_ylim(part_flow.min() - flow_margin, part_flow.max() + flow_margin)

                # don't show x ticks, show timestamp difference as label
                duration = (part_timestamp[-1] - part_timestamp[0]) / 10000
                ax2.set_xlabel(f"start: {part_timestamp[0]} duration: {duration:.2f}s")

                # show only min and max pressure
                ax1.set_yticks([np.min(part_pressure), np.max(part_pressure)])

                # show only min and max flow
                ax2.set_yticks([np.min(part_flow), np.max(part_flow)])

                # set the title of each plot to the index of the breath
                ax1.set_title(f"Breath {self.offset + start_breath_index + i}")

                # setup plot border color and width
                self.sub_axes[r, c, :] = ax1, ax2
                if self.get_breath_label(i) == 0:
                    self.set_breath_label(i, quick_filter(duration, in_vol, ex_vol))
                self.mark_subplot(i)

        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        # self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Show the plot in the top left corner of the screen
        plt.get_current_fig_manager().window.wm_geometry("+0+0")
        plt.show()


breath_loader = BreathLoader(READ_PATH)
breath_recorder = LabelRecorder(SAVE_PATH, breath_loader)
ui = LabelerUI(breath_loader, breath_recorder, ROWS, COLS, WIDTH, HEIGHT)
ui.update(0)