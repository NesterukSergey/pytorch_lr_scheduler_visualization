import streamlit as st
import plotly.graph_objects as go


# @st.cache
def plot(state, history):
    layout = go.Layout(
        autosize=False,
        width=900,
        height=500,

        xaxis=go.layout.XAxis(linecolor='black',
                              linewidth=1,
                              mirror=True),

        yaxis=go.layout.YAxis(linecolor='black',
                              linewidth=1,
                              mirror=True),

        margin=go.layout.Margin(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0
        )
    )

    fig = go.Figure(layout=layout)

    for scheduler in history:
        fig.add_trace(go.Scatter(
            x=list(range(len(scheduler['lr']))),
            y=scheduler['lr'],
            mode='lines',
            name=scheduler['name']
        ))

    if state.logscale:
        axis_scale = 'log'
    else:
        axis_scale = 'linear'

    fig.update_layout(yaxis_type=axis_scale)
    st.plotly_chart(fig)
