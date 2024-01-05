import altair as alt
import gradio as gr
import numpy as np
import pandas as pd
# import vegafusion as vf
#
#
# vf.enable_widget()

alt.data_transformers.disable_max_rows()


def generate_data(length):
    df = pd.DataFrame(
        np.cumsum(np.random.randn(length, 3), 0).round(2),
        columns=['A', 'B', 'C'], index=pd.RangeIndex(length, name='x')
    )
    df = df.reset_index().melt('x', var_name='category', value_name='y')
    df.to_parquet('data.parquet')
    return 'data.parquet'


def create_interactive_multiline_charts(length):
    # Ensure the length is an integer
    length = int(length)

    # Generate random data
    url = generate_data(length)
    source = alt.pd.read_parquet(url)

    # Create the nearest point selection
    nearest = alt.selection_point(nearest=True, on='mouseover',
                                  fields=['x'], empty=False)

    # Create an interval selection for range selection
    interval = alt.selection_interval(encodings=['x'])

    # Base multiline chart
    base_chart = alt.Chart(source).mark_line(interpolate='basis').encode(
        x='x:Q',
        y='y:Q',
        color='category:N'
    ).properties(width=600, height=300)

    # Full series chart with interval selection
    full_chart = base_chart.add_selection(interval)
    full_chart = full_chart.properties(title='Full Chart')

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
    return combined_charts


# Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        length_input = gr.Number(label="Length of Data", value=6000, step=1)
        submit_button = gr.Button("Generate Charts")
    plot_output = gr.Plot(label="Interactive Multi-Line Plots")
    submit_button.click(create_interactive_multiline_charts, inputs=length_input, outputs=plot_output)

if __name__ == "__main__":
    demo.launch()