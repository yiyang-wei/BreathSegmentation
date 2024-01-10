import altair as alt
import gradio as gr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_data(start, length):
    np.random.seed(42)
    x = np.arange(start, start + length)
    y1 = np.cumsum(np.random.randn(length))  # Cumulative sum
    y2 = np.sin(x / 50) * 100  # Sine wave
    y3 = np.log(x + 1) * 50  # Logarithmic

    df = pd.DataFrame({'x': x, 'A': y1, 'B': y2, 'C': y3})
    df_original = df.copy()
    df_original = df_original.melt(id_vars='x', var_name='category', value_name='original_y')

    # Normalize the data for plotting
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df[['A', 'B', 'C']])
    df_scaled = pd.DataFrame(scaled_values, columns=['A', 'B', 'C'])
    df_scaled['x'] = df['x']
    df_scaled = df_scaled.melt(id_vars='x', var_name='category', value_name='y')

    # merge the original and scaled dataframes
    df_scaled['original_y'] = df_original['original_y']

    return df_scaled

def create_interactive_multiline_charts(start, length):
    df_scaled = get_data(start, length)

    highlight = alt.selection_point(on='mouseover', fields=['symbol'], nearest=True)
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['x'], empty=False)
    interval = alt.selection_interval(encodings=['x'])

    base_chart = alt.Chart(df_scaled).mark_line(interpolate='linear').encode(
        x='x:Q',
        y='y:Q',
        color='category:N'
    )

    full_chart = base_chart.add_selection(
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

# Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        start_input = gr.Number(label="Start", value=0, step=1)
        length_input = gr.Number(label="Length of Data", value=100, step=1)
        submit_button = gr.Button("Generate Charts")
    plot_output = gr.Plot(label="Interactive Multi-Line Plot")
    submit_button.click(create_interactive_multiline_charts, inputs=[start_input, length_input], outputs=plot_output)

if __name__ == "__main__":
    demo.launch()
