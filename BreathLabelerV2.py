import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from enum import Enum
import os
import time

# Fix macOS issue
import matplotlib

matplotlib.use('TkAgg')

# do not include slash at the end
READ_FOLDER = "data"
SAVE_FOLDER = "breathlabels"
IN_FILE_NAME = "20190616_EVLP551_converted.csv"
OUT_FILE_NAME = IN_FILE_NAME[:-13] + "labeled.csv"

# window size
WIDTH = 16
HEIGHT = 9

# subplots layout in each page
ROWS = 4
COLS = 5

# font size, recommended (10, 10) for 4x5, (8, 8) for 5x6
INFO_FONT_SIZE = 10
VOL_FONT_SIZE = 10

# line type '-' or '.' or '.-'
LINE_TYPE = '.-'

# count from 1 or 0
OFFSET = 1


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


def quick_filter(duration, vol_ratio, PEEP):
    """Quickly filter the breaths based on their duration."""
    if duration < 4 or duration > 10:  # Too short or too long
        return BreathLabel.Noise.value  # Noise
    elif duration < 7:  # Shorter Breaths
        return BreathLabel.Assessment.value  # Assessment
    elif abs(vol_ratio - 1) > 0.1:  # In_volume and ex_volume are not balanced
        return BreathLabel.Bronch.value  # Bronchospasm
    elif abs(PEEP - 5) > 1:  # PEEP is far from 5
        return BreathLabel.Deflation.value  # Deflation
    else:
        return BreathLabel.Normal.value


READ_PATH = os.path.join(READ_FOLDER, IN_FILE_NAME)
SAVE_PATH = os.path.join(SAVE_FOLDER, OUT_FILE_NAME)


class BreathLoader:
    """Raw data handler."""

    def __init__(self, in_file_path):
        """Read data from a CSV file and segment each breath."""
        try:
            print(f"Reading data from {in_file_path}")
            breath = pd.read_csv(in_file_path, header=0, usecols=["Timestamp", "Pressure", "Flow", "B_phase"])
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
            print(f"Successfully loaded {self.n_breaths} breaths from {in_file_path}\n")
        except Exception as e:
            print(f"Error reading file: {e}")
            exit()

    def get_breath_boundary(self, start_breath_index, end_breath_index=None):
        """Get the start and end data index of the breaths in the given range."""
        if end_breath_index is None:
            end_breath_index = start_breath_index
        start_breath_index = np.clip(start_breath_index, 0, self.n_breaths - 1)
        end_breath_index = np.clip(end_breath_index, 0, self.n_breaths - 1)
        start_idx = self.boundary[start_breath_index] + 1
        end_idx = self.boundary[end_breath_index + 1] + 1
        return start_idx, end_idx

    def get_breath_data(self, start_breath_index, end_breath_index=None):
        """Get the partial data in the given breaths range."""
        start_idx, end_idx = self.get_breath_boundary(start_breath_index, end_breath_index)
        return self.timestamp[start_idx:end_idx], self.pressure[start_idx:end_idx], self.flow[start_idx:end_idx]

    def get_phase(self, breath_index):
        """Get the inhaling and exhaling data of the breath at the given index."""
        start_idx, end_idx = self.get_breath_boundary(breath_index)
        mid_idx = self.switch[breath_index] + 1
        return start_idx, mid_idx, end_idx

    def calc_params(self, breath_index):
        """Get the volume of the breath at the given index."""
        start_idx, mid_idx, end_idx = self.get_phase(breath_index)
        in_timestamp = self.timestamp[start_idx:mid_idx]
        ex_timestamp = self.timestamp[mid_idx:end_idx]
        in_pressure = self.pressure[start_idx:mid_idx]
        in_flow = self.flow[start_idx:mid_idx]
        ex_flow = self.flow[mid_idx:end_idx]

        params = {}
        params["Max_gap(ms)"] = np.max(np.diff(self.timestamp[start_idx:end_idx]))
        params["In_vol(ml)"] = np.trapz(in_flow, in_timestamp) / 100000
        params["Ex_vol(ml)"] = np.trapz(ex_flow, ex_timestamp) / 100000
        params["IE_vol_ratio"] = - params["In_vol(ml)"] / params["Ex_vol(ml)"]
        params["Duration(s)"] = (self.timestamp[end_idx] - self.timestamp[start_idx]) / 10000
        params["In_duration(s)"] = (self.timestamp[mid_idx] - self.timestamp[start_idx]) / 10000
        params["Ex_duration(s)"] = (self.timestamp[end_idx] - self.timestamp[mid_idx]) / 10000
        params["IE_duration_ratio"] = params["In_duration(s)"] / params["Ex_duration(s)"]
        params["P_peak"] = np.max(in_pressure) / 100
        params["PEEP"] = self.pressure[start_idx - 1] / 100
        params["Dy_comp"] = - params["Ex_vol(ml)"] / (params["P_peak"] - 5)
        return params


class ParamTable:

    def __init__(self, out_file_path, breath_data):
        """Create a file if not exist or read the labels if exists."""
        self.out_file_path = out_file_path
        try:
            print(f"Preparing save directory {out_file_path}")
            # create the folder if not exist
            os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
            # read the labels from the csv file if exists or create a new one
            if os.path.exists(out_file_path):
                print(f"Past record exists, reading params from {out_file_path}")
                self.df = pd.read_csv(out_file_path, header=0, sep=',')
                label_names = self.df["Label"]
                # convert the label names to label values
                self.breath_labels = self.to_values(label_names)
                print(f"Successfully loaded {self.breath_labels.shape[0]} rows from {out_file_path}\n")
                self.first_unvisited = self.find_first_unvisited()
            else:
                self.breath_labels = np.zeros(breath_data.n_breaths, dtype=np.int8)
                self.init_df(breath_data)
                self.first_unvisited = 0
                print(f"Successfully created a new table with {self.breath_labels.shape[0]} rows\n")
            self.save()
            print(self.df.head())
            print(self.df.tail())
            print()
            if self.first_unvisited is None:
                print("All breaths have been visited, starting from the beginning\n")
                self.first_unvisited = 0
            else:
                print(f"First unvisited breath: {OFFSET + self.first_unvisited}\n")
        except Exception as e:
            print(f"Error preparing save directory: {e}")
            exit()

    def to_values(self, label_names):
        """Convert the label names to label values."""
        return np.array([BreathLabel[label_name].value for label_name in label_names])

    def to_names(self, label_values):
        """Convert the label values to label names."""
        return np.array([BreathLabel(label_value).name for label_value in label_values])

    def init_df(self, breath_data):
        # Initialize the df with params
        self.df = pd.DataFrame()
        self.df["Breath_num"] = np.arange(OFFSET, OFFSET + breath_data.n_breaths)
        for i in range(breath_data.n_breaths):
            params = breath_data.calc_params(i)
            # add a record to the df with the params
            for key, value in params.items():
                self.df.loc[i, key] = value

        # use Breath_num as index
        self.df.set_index("Breath_num", inplace=True)

    def find_first_unvisited(self):
        """Return the index of the first unvisited breath."""
        unvisited = np.where(self.breath_labels == BreathLabel.Unvisited.value)[0]
        if unvisited.shape[0] > 0:
            return unvisited[0]
        else:
            return None

    def get_label(self, breath_index):
        """Get the label of the breath at the given index."""
        return self.breath_labels[breath_index]

    def get_labels(self, start_breath_index, end_breath_index):
        """Get the label of the breath at the given index."""
        return self.breath_labels[start_breath_index:end_breath_index]

    def set_label(self, breath_index, label):
        """Set the label of the breath at the given index."""
        self.breath_labels[breath_index] = label

    def set_labels(self, start_breath_index, labels):
        """Set the label of the breath at the given index."""
        for r in range(labels.shape[0]):
            for c in range(labels.shape[1]):
                idx = start_breath_index + r * labels.shape[1] + c
                if idx < self.breath_labels.shape[0]:
                    self.breath_labels[idx] = labels[r, c]

    def get_param(self, breath_index):
        """Get the params of the breath at the given index."""
        return self.df.iloc[breath_index]

    def save(self):
        """Save the labels to a CSV file."""
        self.df["Label"] = self.to_names(self.breath_labels)
        try:
            # save the df to a csv file
            print(f"Saving params to {self.out_file_path}")
            self.df.to_csv(self.out_file_path, index=False)
        except Exception as e:
            print(f"Error saving file: {e}")


class LabelerUI:

    def __init__(self, rows, cols, width, height):
        self.rows = rows
        self.cols = cols
        self.fig = plt.figure(figsize=(width, height))

        # Create outer GridSpec, add one more row for the top plot
        self.outer_grid = gridspec.GridSpec(rows + 1, cols,
                                            wspace=0.25, hspace=0.5,
                                            left=0.05, right=0.95, bottom=0.05, top=0.95)

        # Top plot that spans the entire row
        self.top_ax = plt.subplot(self.outer_grid[0, :])
        self.span = None
        self.page = 0
        self.total_pages = 0

        # Subplots with size rows x cols
        self.sub_axes = np.empty((rows, cols, 2), dtype=object)
        self.init_sub_axes()

        self.breath_labels = np.zeros((rows, cols), dtype=np.int8)
        self.info = np.empty((rows, cols), dtype=object)
        self.mask = np.empty((rows, cols), dtype=object)
        self.annotation = np.empty((rows, cols), dtype=object)

    def init_sub_axes(self):
        for r in range(self.rows):
            for c in range(self.cols):
                # Create inner GridSpec of 2 rows and 1 column
                inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=self.outer_grid[r + 1, c],
                                                              wspace=0.1, hspace=0.0)
                ax1 = plt.Subplot(self.fig, inner_grid[0])
                ax2 = plt.Subplot(self.fig, inner_grid[1])

                # Rotate y-tick labels by 90 degrees to save space
                # Move ax2's y-tick labels to the right side of the plot for better readability
                ax1.tick_params(axis='y', colors='C1', rotation=90, labelsize=9)
                ax2.tick_params(axis='y', colors='C0', rotation=90, labelsize=9,
                                left=False, right=True, labelleft=False, labelright=True)

                # Remove y-labels to save space
                ax1.set_ylabel('')
                ax2.set_ylabel('')

                # Remove x-axis labels from both subplots
                ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

                # Add subplots to figure
                self.fig.add_subplot(ax1)
                self.fig.add_subplot(ax2)

                # Store axes for later access
                self.sub_axes[r, c, :] = ax1, ax2

    def draw_top_ax(self, timestamp, flow, total_pages):
        self.total_pages = total_pages
        self.top_ax.plot(timestamp, flow, linewidth=0.5)
        self.top_ax.set_yticks([np.min(flow), np.max(flow)])
        self.top_ax.tick_params(axis='y', colors='C0', rotation=90)
        self.update_title(-1)

    def update_title(self, page):
        self.page = page
        self.top_ax.set_title(f"Page {page + 1}/{self.total_pages}")

    def get_breath_number(self, r, c):
        return OFFSET + self.page * self.rows * self.cols + r * self.cols + c

    def get_breath_label(self, r, c):
        return self.breath_labels[r, c]

    def set_breath_label(self, r, c, label):
        self.breath_labels[r, c] = label
        self.mark_subplot(r, c)

    def update_span(self, start_x, end_x):
        if self.span is not None:
            self.span.remove()
        self.span = self.top_ax.axvspan(start_x, end_x, alpha=0.2, color='grey')

    def masked(self, r, c):
        """Check if the breath at the given position is masked."""
        return self.mask[r, c] is not None

    def clear_mask(self, r, c):
        if self.masked(r, c):
            self.mask[r, c].remove()
            self.mask[r, c] = None
            self.annotation[r, c].remove()
            self.annotation[r, c] = None
            return True
        return False

    def clear_subplot(self, r, c):
        ax1, ax2 = self.sub_axes[r, c, :]
        ax1.clear()
        ax2.clear()
        self.info[r, c] = None
        self.clear_mask(r, c)

    def find_ax(self, ax):
        if ax is None or ax is self.top_ax:
            return None, None
        pos = np.where(self.sub_axes == ax)
        r, c = pos[0][0], pos[1][0]
        return r, c

    def switch_breath_label(self, r, c):
        self.breath_labels[r, c] = self.breath_labels[r, c] % (N_LABELS - 1) + 1
        self.mark_subplot(r, c)

    def mark_subplot(self, r, c):
        """Update the plot of the i-th breath."""
        ax1, ax2 = self.sub_axes[r, c, :]
        label = self.breath_labels[r, c]
        breath_num = self.get_breath_number(r, c)
        if label <= 1:
            ax1.set_title(f"Breath {breath_num}")
        else:
            ax1.set_title(f"Breath {breath_num} ({BreathLabel(label).name})")
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

    def toggle_detail(self, r, c):
        if r is None or c is None:
            return
        if not self.clear_mask(r, c):
            ax1, ax2 = self.sub_axes[r, c, :]
            self.mask[r, c] = patches.Rectangle((0, 0), 1, 2, transform=ax2.transAxes,
                                                color='grey', alpha=0.75, clip_on=False)
            ax2.add_patch(self.mask[r, c])
            self.annotation[r, c] = ax2.annotate(self.info[r, c], xy=(0.5, 1), fontsize=INFO_FONT_SIZE,
                                                 ha='center', va='center', xycoords='axes fraction', color='white')

    def to_info(self, params):
        """Convert the params to info."""
        return f"""\
max_gap: {params['Max_gap(ms)'] / 10:.2f}ms
ie_dur_ratio: {params['IE_duration_ratio']:.2f}
ie_vol_ratio: {params['IE_vol_ratio']:.2f}
PEEP: {params['PEEP']:.2f}
Dy_comp: {params['Dy_comp']:.2f}"""

    def update_subplot(self, r, c, part_data, params, label):
        """Update the plot of the breath at the given position."""
        ax1, ax2 = self.sub_axes[r, c, :]

        breath_num = self.get_breath_number(r, c)
        part_timestamp, part_pressure, part_flow = part_data

        # Create ax1 for pressure and ax2 for flow
        ax1.plot(part_timestamp, part_pressure, LINE_TYPE, color='C1', alpha=0.5)
        ax2.plot(part_timestamp, part_flow, LINE_TYPE, color='C0', alpha=0.5)

        # Calculate and display the in_vol and ex_vol on the plot
        in_vol = params["In_vol(ml)"]
        ex_vol = params["Ex_vol(ml)"]
        ax2.text(0.02, 0.12, f"In: {in_vol:.2f}", transform=ax2.transAxes, color="forestgreen", fontsize=VOL_FONT_SIZE,
                 bbox=dict(facecolor='ghostwhite', edgecolor='silver', boxstyle='round', alpha=0.5, pad=0.2))
        ax2.text(0.65, 0.12, f"Ex: {ex_vol:.2f}", transform=ax2.transAxes, color="orangered", fontsize=VOL_FONT_SIZE,
                 bbox=dict(facecolor='ghostwhite', edgecolor='silver', boxstyle='round', alpha=0.5, pad=0.2))

        self.info[r, c] = self.to_info(params)

        # Calculate the margins for pressure and flow based on their respective ranges
        margin_factor = 0.15
        pressure_margin = margin_factor * (part_pressure.max() - part_pressure.min())
        flow_margin = margin_factor * (part_flow.max() - part_flow.min())
        # Adjust y-limits with margins for both top and bottom of the plots
        ax1.set_ylim(part_pressure.min() - pressure_margin, part_pressure.max() + pressure_margin)
        ax2.set_ylim(part_flow.min() - flow_margin, part_flow.max() + flow_margin)

        # don't show x ticks, show timestamp difference as label
        duration = params["Duration(s)"]
        ax2.set_xlabel(f"duration: {duration:.2f}s")

        # show only min and max of pressure and flow
        ax1.set_yticks([part_pressure.min(), part_pressure.max()])
        ax2.set_yticks([part_flow.min(), part_flow.max()])

        # set the title of each plot to the index of the breath
        ax1.set_title(f"Breath {breath_num}")

        self.set_breath_label(r, c, label)


class BreathLabeler:

    def __init__(self, breath_data, param_table, rows, cols, width, height):
        self.breath_data = breath_data
        self.param_table = param_table
        self.rows = rows
        self.cols = cols
        self.per_page = rows * cols
        self.ui = LabelerUI(rows, cols, width, height)
        self.total_pages = np.ceil(breath_data.n_breaths / self.per_page).astype(int)
        self.ui.draw_top_ax(breath_data.timestamp, breath_data.flow, self.total_pages)
        self.page = -1
        self.update(self.param_table.first_unvisited // self.per_page)
        # Show the plot in the top left corner of the screen
        plt.gcf().canvas.manager.window.wm_geometry("+0+0")
        plt.show()

    def in_range(self, r, c):
        if r is None or c is None:
            return False
        return self.page * self.per_page + r * self.cols + c < self.breath_data.n_breaths

    def page_clicked(self, xdata):
        # find the index of the clicked point
        clicked_index = np.searchsorted(self.breath_data.timestamp, xdata, side='left')
        # find the index of the breath that contains the clicked point
        breath_index = np.searchsorted(self.breath_data.boundary, clicked_index, side='right') - 1
        # get the page start index of the breath
        page_index = breath_index // self.per_page
        return page_index

    def on_click(self, event):
        """Handle the click event."""
        ax_clicked = event.inaxes
        if event.button == 1:
            if ax_clicked == self.ui.top_ax:
                page_index = self.page_clicked(event.xdata)
                if page_index >= self.total_pages:
                    return
                else:
                    self.page = page_index
                    self.update(self.page)
                    self.ui.fig.canvas.draw()
            else:
                r, c = self.ui.find_ax(ax_clicked)
                if self.in_range(r, c):
                    self.ui.switch_breath_label(r, c)
                    self.ui.fig.canvas.draw()
        elif event.button == 3:
            if ax_clicked is not self.ui.top_ax:
                r, c = self.ui.find_ax(ax_clicked)
                if self.in_range(r, c):
                    self.ui.toggle_detail(r, c)
                    self.ui.fig.canvas.draw()

    def on_key(self, event):
        """Handle key press events."""
        if event.key == 'backspace':
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.in_range(r, c):
                        self.ui.clear_mask(r, c)
                        self.ui.set_breath_label(r, c, BreathLabel.Normal.value)
            self.ui.fig.canvas.draw()
        elif event.key == ' ':
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.in_range(r, c):
                        self.ui.toggle_detail(r, c)
            self.ui.fig.canvas.draw()
        elif event.key == 'enter' or event.key == 'right':
            self.update(self.page + 1)
        elif event.key == 'left':
            self.update(self.page - 1)
        elif event.key in '12345678':
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.in_range(r, c) and self.ui.masked(r, c):
                        self.ui.set_breath_label(r, c, int(event.key))
            self.ui.fig.canvas.draw()
        elif event.key == 'escape':
            self.save()
            plt.close()

    def update(self, page_index):
        """Update the plot."""
        if page_index < 0 or page_index >= self.total_pages:
            return
        start_time = time.time()
        self.save()
        self.page = page_index
        start_index = page_index * self.per_page
        self.ui.update_title(page_index)
        start_idx, end_idx = self.breath_data.get_breath_boundary(start_index, start_index + self.per_page - 1)
        self.ui.update_span(self.breath_data.timestamp[start_idx], self.breath_data.timestamp[end_idx])
        for r in range(self.rows):
            for c in range(self.cols):
                self.ui.clear_subplot(r, c)
                if self.in_range(r, c):
                    breath_index = start_index + r * self.cols + c
                    part_data = self.breath_data.get_breath_data(breath_index)
                    params = self.param_table.get_param(breath_index)
                    label = self.param_table.get_label(breath_index)
                    if label == BreathLabel.Unvisited.value:
                        label = quick_filter(params["Duration(s)"], params["IE_vol_ratio"], params["PEEP"])
                    self.ui.update_subplot(r, c, part_data, params, label)
        # Connect event handlers
        self.ui.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.ui.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.ui.fig.canvas.draw()
        end_time = time.time()
        print(f"Updated to page {page_index + 1}/{self.total_pages} in {end_time - start_time:.2f}s\n")

    def save(self):
        """Save the labels to a CSV file."""
        if self.page < 0 or self.page >= self.total_pages:
            return
        print(f"Saving page {self.page + 1}/{self.total_pages}")
        self.param_table.set_labels(self.page * self.per_page, self.ui.breath_labels)
        self.param_table.save()

breath_data = BreathLoader(READ_PATH)
param_table = ParamTable(SAVE_PATH, breath_data)
labeler = BreathLabeler(breath_data, param_table, ROWS, COLS, WIDTH, HEIGHT)
