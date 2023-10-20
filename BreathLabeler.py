# Configurations

READ_FOLDER = "data"
SAVE_FOLDER = "breathlabels"
FILE_NAME = "20190616_EVLP551_converted.csv"

ROWS = 4
COLS = 5

WIDTH = 16
HEIGHT = 9

LABELS = {0: {"label": "Normal", "color": "black"},
          1: {"label": "Assessment", "color": "green"},
          2: {"label": "Bronch", "color": "blue"},
          3: {"label": "Deflation", "color": "violet"},
          4: {"label": "Question", "color": "orange"},
          5: {"label": "In-Pause", "color": "gold"},
          6: {"label": "Ex-Pause", "color": "indigo"},
          7: {"label": "Noise", "color": "red"}}

def quick_filter(duration):
    """Quickly filter the breaths based on their duration."""
    if duration < 4 or duration > 10:
        return 7
    elif duration < 7:
        return 1
    else:
        return 0


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Fix macOS issue
import matplotlib
matplotlib.use('TkAgg')

READ_PATH = os.path.join(READ_FOLDER, FILE_NAME)
SAVE_PATH = os.path.join(SAVE_FOLDER, FILE_NAME)

class BreathLoader:

    def __init__(self, file_path):
        """Read data from a CSV file and segment each breath."""
        self.file_path = file_path
        try:
            breath = pd.read_csv(file_path, header=0, sep=',', usecols=["Timestamp", "Pressure", "Flow", "B_phase"])
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
        except Exception as e:
            print(f"Error reading file: {e}")
            exit()

    def get_volume(self, breath_index):
        """Get the volume of the breath at the given index."""
        start_idx, end_idx = self.get_breath_boundry(breath_index)
        mid_idx = self.switch[breath_index] + 1
        return np.sum(self.flow[start_idx:mid_idx]) / 1000, np.sum(self.flow[mid_idx:end_idx]) / 1000

    def get_breath_boundry(self, start_breath_index, end_breath_index=None):
        """Get the breaths in the given range."""
        if end_breath_index is None:
            end_breath_index = start_breath_index
        start_breath_index = max(start_breath_index, 0)
        end_breath_index = min(end_breath_index, self.n_breaths - 1)
        start_idx = self.boundary[start_breath_index] + 1
        end_idx = self.boundary[end_breath_index + 1] + 1
        return start_idx, end_idx

    def get_breath_data(self, start_breath_index, end_breath_index=None):
        """Get the breaths in the given range."""
        start_idx, end_idx = self.get_breath_boundry(start_breath_index, end_breath_index)
        return self.timestamp[start_idx:end_idx], self.pressure[start_idx:end_idx], self.flow[start_idx:end_idx]


class BreathLabeler:

    def __init__(self, file_path, rows=ROWS, cols=COLS, labels=LABELS):
        self.file_path = file_path
        self.breath_data = BreathLoader(file_path)
        self.rows = rows
        self.cols = cols
        self.total = rows * cols
        self.labels = labels
        self.n_labels = len(labels)
        self.breath_labels = np.zeros(self.breath_data.n_breaths, dtype=np.int8)
        for i in range(self.breath_data.n_breaths):
            breath_start, breath_end = self.breath_data.get_breath_boundry(i)
            breath_duration = (self.breath_data.timestamp[breath_end - 1] - self.breath_data.timestamp[breath_start]) / 10000
            self.breath_labels[i] = quick_filter(breath_duration)
        self.breath_index = 0
        self.subplot_axes = [(None, None)] * self.total
        self.fig = None
        self.prev = False
        self.stop = False

    def input_offset(self):
        # print the timestamps of the first five breath
        for i in range(5):
            start_index, end_index = self.breath_data.get_breath_boundry(i)
            start_timestamp = self.breath_data.timestamp[start_index]
            end_timestamp = self.breath_data.timestamp[end_index - 1]
            print(f"Breath {i} starts and ends at [{start_timestamp}, {end_timestamp}]")

        # input the offset of the breath index
        self.offset = int(input("Enter the offset to add on breath index: "))

    def update_plot(self, axes, label, idx):
        """Update the plot of the i-th breath."""
        if label == 0:
            axes[0].set_title(f"Breath {idx}")
        else:
            axes[0].set_title(f"Breath {idx} ({self.labels[label]['label']})")
        axes[0].title.set_color(self.labels[label]['color'])
        for ax in axes:
            for spine in ax.spines.values():
                spine.set_edgecolor(self.labels[label]['color'])
                if label == 0:
                    spine.set_linewidth(0.5)
                else:
                    spine.set_linewidth(4)

    def on_click(self, event):
        """Handle click events on subplots."""
        for i in range(self.total):
            if self.subplot_axes[i][0].contains(event)[0]:
                if event.button == 1:
                    self.breath_labels[self.breath_index + i] = self.n_labels - 1 if self.breath_labels[self.breath_index + i] != self.n_labels - 1 else 0
                else:
                    self.breath_labels[self.breath_index + i] = (self.breath_labels[self.breath_index + i] + 1) % (self.n_labels - 1)
                self.update_plot(self.subplot_axes[i], self.breath_labels[self.breath_index + i], self.offset + self.breath_index + i)
                self.fig.canvas.draw()
                break

    def on_key(self, event):
        """Handle key press events."""
        if event.key == 'backspace':
            for i in range(self.total):
                self.breath_labels[self.breath_index + i] = 0
                self.update_plot(self.subplot_axes[i], self.breath_labels[self.breath_index + i], self.offset + self.breath_index + i)
            self.fig.canvas.draw()
        elif event.key == ' ':
            for i in range(self.total):
                if self.breath_labels[self.breath_index + i] != self.n_labels - 1:
                    self.breath_labels[self.breath_index + i] = (self.breath_labels[self.breath_index + i] + 1) % (self.n_labels - 1)
                    self.update_plot(self.subplot_axes[i], self.breath_labels[self.breath_index + i],
                                     self.offset + self.breath_index + i)
            self.fig.canvas.draw()
        elif event.key == 'enter' or event.key == 'right':
            plt.close()
        elif event.key == 'left':
            self.prev = True
            plt.close()
        elif event.key == 'escape':
            self.stop = True
            plt.close()

    def show(self, width=WIDTH, height=HEIGHT):
        self.fig = plt.figure(figsize=(width, height))

        # Create the top plot for the whole time series
        top_ax = plt.subplot2grid((self.rows + 1, self.cols), (0, 0), colspan=self.cols)
        top_ax.plot(self.breath_data.timestamp, self.breath_data.flow, linewidth=0.5)
        top_ax.set_yticks([np.min(self.breath_data.flow), np.max(self.breath_data.flow)])
        top_ax.tick_params(axis='y', rotation=90)
        window_start, window_end = self.breath_data.get_breath_boundry(self.breath_index,
                                                                       self.breath_index + self.total - 1)
        top_ax.axvspan(self.breath_data.timestamp[window_start], self.breath_data.timestamp[window_end], alpha=0.2, color='grey')
        top_ax.set_title("Flow")

        for i in range(self.total):
            if self.breath_index + i >= self.breath_data.n_breaths:
                break
            part_timestamp, part_pressure, part_flow = self.breath_data.get_breath_data(self.breath_index + i)

            # Create ax1 for pressure
            ax1 = plt.subplot(self.rows + 1, self.cols, i + self.cols + 1)
            ax1.plot(part_timestamp, part_pressure, '.-', color='C1', alpha=0.5)

            # Create ax2 (twin axis of ax1) for flow
            ax2 = ax1.twinx()
            ax2.plot(part_timestamp, part_flow, '.-', color='C0', alpha=0.5)

            # Calculate the volume of the breath
            in_vol, ex_vol = self.breath_data.get_volume(self.breath_index + i)
            # display the in_vol and ex_vol on the plot at bottom left and right corner respectively with a rounded corner box
            ax1.text(0.05, 0.1, f"In: {in_vol:.2f}", transform=ax1.transAxes, color="forestgreen", bbox=dict(facecolor='ghostwhite', edgecolor='silver', boxstyle='round', alpha=0.5, pad=0.2))
            ax1.text(0.65, 0.1, f"Ex: {ex_vol:.2f}", transform=ax2.transAxes, color="orangered", bbox=dict(facecolor='ghostwhite', edgecolor='silver', boxstyle='round', alpha=0.5, pad=0.2))


            # Calculate the margins for pressure and flow based on their respective ranges
            margin_factor = 0.15
            pressure_margin = margin_factor * (part_pressure.max() - part_pressure.min())
            flow_margin = margin_factor * (part_flow.max() - part_flow.min())

            # Adjust y-limits with margins for both top and bottom of the plots
            ax1.set_ylim(2 * part_pressure.min() - part_pressure.max() - 2 * pressure_margin,
                         part_pressure.max() + pressure_margin)
            ax2.set_ylim(part_flow.min() - flow_margin, 2 * part_flow.max() - part_flow.min() + 2 * flow_margin)

            # don't show x ticks, show timestamp difference as label
            duration = (part_timestamp[-1] - part_timestamp[0]) / 10000
            ax1.set_xticks([])
            ax1.set_xlabel(f"start: {part_timestamp[0]} duration: {duration:.2f}s")
            # show only min and max pressure
            ax1.set_yticks([np.min(part_pressure), np.max(part_pressure)])
            ax1.tick_params(axis='y', colors='C1')
            # show only min and max flow
            ax2.set_yticks([np.min(part_flow), np.max(part_flow)])
            ax2.tick_params(axis='y', colors='C0')
            # set the title of each plot to the index of the breath
            ax1.set_title(f"Breath {self.offset + self.breath_index + i}")
            # make y axis number vertical
            ax1.tick_params(axis='y', rotation=90)
            ax2.tick_params(axis='y', rotation=90)
            # setup plot border color and width
            self.subplot_axes[i] = ax1, ax2
            self.update_plot(self.subplot_axes[i], self.breath_labels[self.breath_index + i], self.offset + self.breath_index + i)

        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.tight_layout()
        # Show the plot in the top left corner of the screen
        plt.get_current_fig_manager().window.wm_geometry("+0+0")
        plt.show()
        for i in range(self.total):
            if self.breath_index + i >= self.breath_data.n_breaths:
                break
            if self.breath_labels[self.breath_index + i] != 0:
                print(f"Breath {self.offset + self.breath_index + i} is labeled as {self.labels[self.breath_labels[self.breath_index + i]]['label']}")
        if self.prev:
            self.breath_index -= self.total
            self.breath_index = max(self.breath_index, 0)
            self.prev = False
        else:
            self.breath_index += self.total

    def start(self):
        self.input_offset()
        while self.breath_index < self.breath_data.n_breaths and not self.stop:
            self.show()
        for i in range(self.breath_index, self.breath_data.n_breaths):
            self.breath_labels[i] = self.n_labels - 1

    def save(self, file_path):
        """Save the labels to a CSV file."""
        # create a df from the labels with two columns: breath index and label name
        df = pd.DataFrame({"Breath": np.arange(self.breath_data.n_breaths)+self.offset, "Label": self.breath_labels})
        # use the label name as the value of the label
        df["Label"] = df["Label"].apply(lambda x: self.labels[x]["label"])
        try:
            # create the folder if not exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # save the df to a csv file
            df.to_csv(file_path, index=False)
        except Exception as e:
            print(f"Error saving file: {e}")


labeler = BreathLabeler(READ_PATH)
labeler.start()
labeler.save(SAVE_PATH)

