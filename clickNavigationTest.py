import matplotlib.pyplot as plt
import numpy as np

# Initialize the plot
fig, ax = plt.subplots()
x = np.linspace(0, 1000, 1000)
y = np.sin(x * 0.01)
ax.plot(x, y)

# Store initial span, set to None
current_span = None


# Function to handle click events
def onclick(event):
    global current_span  # Use the global variable to track the current span

    # Remove the old span if it exists
    if current_span:
        current_span.remove()

    # Calculate the start and end x-coordinates for the new span
    x_start = 100 * (event.xdata // 100)  # Floor division to get the start
    x_end = x_start + 100  # End of the span is start + 100

    # Add the new span
    current_span = ax.axvspan(x_start, x_end, facecolor='y', alpha=0.5)

    # Redraw the plot
    fig.canvas.draw()


# Connect the click event to the onclick function
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
