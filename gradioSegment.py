import altair as alt
import gradio as gr
import numpy as np
import pandas as pd
import os
import re
# import vegafusion as vf
#
#
# vf.enable_widget()

alt.data_transformers.disable_max_rows()

def read_cases(folder):
    cases = {}
    for file_name in os.listdir(folder):
        if file_name.endswith(".csv"):
            case_id = int(re.search(r"evlp\d+", file_name).group()[4:])
            cases[case_id] = os.path.join(folder, file_name)
    return cases


# def segment_breath(flow, threshold01=5, threshold10=5, slope_threshold01=1, slope_threshold10=1, forward=4, flow_threshold01=9, flow_threshold10=-0.5):
#     B_phase = np.zeros(flow.shape[0])
#     phase = 0
#     idx = 1
#     while idx < flow.shape[0] - 10:
#         if ((flow[idx] < threshold01 <= flow[idx+1] or flow[idx+1] - flow[idx] > slope_threshold01) and flow[idx+forward] > flow_threshold01) or flow[idx] > flow_threshold01:
#             phase = 1
#         elif ((flow[idx-1] > threshold10 >= flow[idx] or flow[idx] - flow[idx-1] < -slope_threshold10) and flow[idx+forward] < flow_threshold10) or flow[idx] < flow_threshold10:
#             phase = 0
#         B_phase[idx] = phase
#         idx += 1
#     return B_phase


def segment_breath(flow, threshold01=5, threshold10=5, slope_threshold01=1.6, slope_threshold10=2, forward=3, flow_threshold01=7.5, flow_threshold10=-1, flow_hard_threshold01=100):
    B_phase = np.zeros(flow.shape[0])
    phase = 0
    idx = 1
    while idx < flow.shape[0] - 10:
        if ((flow[idx] < threshold01 <= flow[idx+1] or flow[idx+1] - flow[idx] > slope_threshold01) and flow[idx+forward] > flow_threshold01) or flow[idx] > flow_hard_threshold01:
            phase = 1
        elif ((flow[idx-1] > threshold10 >= flow[idx] or flow[idx] - flow[idx-1] < -slope_threshold10) and flow[idx+forward] < flow_threshold10) or flow[idx] < flow_threshold10:
            phase = 0
        B_phase[idx] = phase
        idx += 1
    return B_phase


def get_data(case, start, length):
    filename = bellavista_cases[case]
    # cols = ["Pressure", "Flow", "Volume"]
    cols = ["Flow"]
    df = pd.read_csv(filename, header=0, usecols=cols)
    df["B_phase"] = segment_breath(df["Flow"].values) * 40 - 20
    cols = ["Flow", "B_phase"]
    df = df.iloc[start:start+length]
    df["x"] = np.arange(start, start+length)
    return df.melt(id_vars=["x"], value_vars=cols, var_name="category", value_name="y")

def create_interactive_multiline_charts(case, start, length):
    # Ensure the length is an integer
    case = int(case)
    start = int(start)
    length = int(length)

    # Generate random data
    # url = generate_data(length)
    # source = alt.pd.read_parquet(url)
    source = get_data(case, start, length)

    # Create the nearest point selection
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['x'], empty=False)

    # Create an interval selection for range selection
    interval = alt.selection_interval(encodings=['x'])

    # Base multiline chart
    base_chart = alt.Chart(source).mark_line(interpolate='linear').encode(
        x='x:Q',
        y='y:Q',
        # y=alt.Y('y:Q', scale=alt.Scale(zero=False)),
        color='category:N'
    ).properties(width=600, height=300)
    # ).properties(width=600, height=200).facet(row='category:N')

    # Full series chart with interval selection
    full_chart = base_chart.add_selection(
        interval
    ).properties(
        title='Full Chart'
    )

    # Chart for displaying the selected range
    range_chart = base_chart.transform_filter(interval)
    range_chart = range_chart.properties(title='Selected Range')

    # Transparent selectors across the chart of selected range
    selectors = range_chart.mark_point().encode(
        opacity=alt.value(0)
    ).add_selection(
        nearest
    )

    # Points on the line
    points = range_chart.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Text labels
    text = range_chart.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'y:Q', alt.value(' '))
    )

    # Rule at the selection
    rules = alt.Chart(source).mark_rule(color='gray').encode(
        x='x:Q',
    ).transform_filter(
        nearest
    )

    # Combine layers
    interactive_plot = alt.layer(
        range_chart, selectors, points, rules, text
    ).properties(
        width=600, height=300
    )

    # Combine and return the charts as JSON for Gradio compatibility
    combined_charts = alt.vconcat(full_chart, interactive_plot)
    # combined_charts = alt.vconcat(full_chart, range_chart)
    return combined_charts


bellavista_folder = r"..\EVLP data\Bellavista data"
bellavista_cases = read_cases(bellavista_folder)

# Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        case_input = gr.Dropdown(label="EVLP Case", choices=list(bellavista_cases.keys()))
        start_input = gr.Number(label="Start Index", value=800, step=6000)
        length_input = gr.Number(label="Length of Data", value=6000, step=1, minimum=1, maximum=12000)
        submit_btn = gr.Button(value="Submit")
    plot_output = gr.Plot(label="Interactive Multi-Line Plots")
    case_input.change(create_interactive_multiline_charts, inputs=[case_input, start_input, length_input], outputs=plot_output)
    start_input.change(create_interactive_multiline_charts, inputs=[case_input, start_input, length_input], outputs=plot_output)
    length_input.change(create_interactive_multiline_charts, inputs=[case_input, start_input, length_input], outputs=plot_output)

if __name__ == "__main__":
    demo.launch()

