import numpy as np


class SlidingWindow:

    def __init__(self, left_window, right_window, left_focus, right_focus, left_clip, right_clip, step):
        self.left_window = left_window
        self.right_window = right_window
        self.left_focus = left_focus
        self.right_focus = right_focus
        self.left_clip = left_clip
        self.right_clip = right_clip
        self.step = step
        self.window_size = left_window + right_window + 1

    def get_windows(self, length):
        """Get the current sliding window."""
        n = (length - self.left_clip - self.right_clip - self.left_window - self.right_window - 1) // self.step + 1
        windows = np.zeros((n, 5), dtype=np.int32)
        mid = self.left_clip + self.left_window

        for i in range(n):
            windows[i, :] = (mid - self.left_window, mid - self.left_focus, mid,
                             mid + self.right_focus + 1, mid + self.right_window + 1)
            mid += self.step
        return windows


if __name__ == "__main__":
    sd = SlidingWindow(2, 4, 1, 2, 5, 5, 2)
    print(sd.get_windows(20))
    print(sd.get_windows(21))
    print(sd.get_windows(22))
