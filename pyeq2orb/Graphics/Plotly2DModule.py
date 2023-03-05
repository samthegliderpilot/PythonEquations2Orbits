import pyeq2orb.Graphics.Primitives as prim
import plotly.express as px
import plotly.graph_objects as go
from typing import List

def plot2DLines(data : List[prim.XAndYPlottableLineData], title : str) :

    fig = go.Figure()    
    for plottableData in data :
        lineData = {"color":plottableData.color, "width":plottableData.lineWidth}
        markerData = {"color":plottableData.color, "size":plottableData.markerSize}
        mode = ""
        if plottableData.lineWidth != 0 :
            mode = "lines"
        if plottableData.markerSize != 0 :
            if mode != "" :
                mode += "+"
            mode += "markers"

        fig.add_trace(go.Scatter(x=plottableData.x, y=plottableData.y, marker=markerData, line=lineData, name =plottableData.label, mode=mode))           
    fig.update_layout(
        showlegend=True,
        font={'size': 10},
        title={'text': title, 'font': {'size': 20}}
    )
    fig.show()