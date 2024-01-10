import altair as alt
import gradio as gr
import numpy as np
import pandas as pd
import os
import re
from sklearn.preprocessing import MinMaxScaler

# import vegafusion as vf
#
#
# vf.enable_widget()

alt.data_transformers.disable_max_rows()


COLS = ["Flow", "B_phase"]


def select_cols(flow, volume, pressure, B_phase):
    COLS.clear()
    if flow:
        COLS.append("Flow")
    if volume:
        COLS.append("Volume")
    if pressure:
        COLS.append("Pressure")
    if B_phase:
        COLS.append("B_phase")
    return COLS


def read_cases(folder):
    cases = {}
    for file_name in os.listdir(folder):
        if file_name.endswith(".csv"):
            case_id = int(re.search(r"evlp\d+", file_name).group()[4:])
            cases[case_id] = os.path.join(folder, file_name)
    return cases


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
    df = pd.read_csv(filename, header=0)
    if "B_phase" in COLS:
        df["B_phase"] = segment_breath(df["Flow"].values)
    df = df[COLS]
    df = df.iloc[start:start+length]
    df["x"] = np.arange(start, start+length)

    df_original = df.copy()
    df_original = df_original.melt(id_vars='x', var_name='category', value_name='original_y')

    print(df_original)

    # Normalize the data for plotting
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df[COLS])
    df_scaled = pd.DataFrame(scaled_values, columns=COLS)
    df_scaled['x'] = np.arange(start, start+length)
    df_scaled = df_scaled.melt(id_vars='x', var_name='category', value_name='y')

    print(df_scaled)

    # merge the original and scaled dataframes
    df_scaled['original_y'] = df_original['original_y']

    print(df_scaled)

    return df_scaled

def create_interactive_multiline_charts(case, start, length):
    # Ensure the length is an integer
    case = int(case)
    start = int(start)
    length = int(length)

    df_scaled = get_data(case, start, length)

    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['x'], empty=False)
    interval = alt.selection_interval(encodings=['x'])

    base_chart = alt.Chart(df_scaled).mark_line(interpolate='linear').encode(
        x='x:Q',
        y='y:Q',
        color='category:N'
    )

    full_chart = base_chart.add_params(
        interval
    ).properties(
        title='Full Chart'
    )

    # Chart for displaying the selected range
    range_chart = base_chart.transform_filter(interval)
    range_chart = range_chart.properties(title='Selected Range')

    selectors = alt.Chart(df_scaled).transform_filter(interval).mark_point().encode(
        x='x:Q',
        opacity=alt.value(0),
    ).add_params(
        nearest
    )

    points = range_chart.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    text = range_chart.mark_text(align='left', dx=5, dy=-5).encode(
        x='x:Q',
        text=alt.condition(nearest, 'original_y:Q', alt.value(' '), format='.3f')
    ).transform_filter(
        nearest
    )

    rules = alt.Chart(df_scaled).mark_rule(color='gray').encode(
        x='x:Q',
    ).transform_filter(
        nearest
    )

    upper_plot = alt.layer(
        full_chart, rules
    ).properties(
        width=600, height=300
    )

    lower_plot = alt.layer(
        range_chart, selectors, points, rules, text
    ).properties(
        width=600, height=300
    )

    combined_charts = alt.vconcat(upper_plot, lower_plot)

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
    with gr.Row():
        plot_flow = gr.Checkbox(label="Plot Flow", value=True)
        plot_volume = gr.Checkbox(label="Plot Volume", value=False)
        plot_pressure = gr.Checkbox(label="Plot Pressure", value=False)
        plot_B_phase = gr.Checkbox(label="Plot B Phase", value=True)
    plot_flow.change(select_cols, inputs=[plot_flow, plot_volume, plot_pressure, plot_B_phase])
    plot_volume.change(select_cols, inputs=[plot_flow, plot_volume, plot_pressure, plot_B_phase])
    plot_pressure.change(select_cols, inputs=[plot_flow, plot_volume, plot_pressure, plot_B_phase])
    plot_B_phase.change(select_cols, inputs=[plot_flow, plot_volume, plot_pressure, plot_B_phase])
    plot_output = gr.Plot(label="Interactive Multi-Line Plots")
    submit_btn.click(create_interactive_multiline_charts, inputs=[case_input, start_input, length_input], outputs=plot_output)
    case_input.change(create_interactive_multiline_charts, inputs=[case_input, start_input, length_input], outputs=plot_output)
    start_input.change(create_interactive_multiline_charts, inputs=[case_input, start_input, length_input], outputs=plot_output)
    length_input.change(create_interactive_multiline_charts, inputs=[case_input, start_input, length_input], outputs=plot_output)

if __name__ == "__main__":
    demo.launch()

