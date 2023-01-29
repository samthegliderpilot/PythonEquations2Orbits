import pyeq2orb.Graphics.Primitives as prim
import plotly.express as px
import plotly.graph_objects as go
from typing import List

def plot2DLines(data : List[prim.XAndYPlottableLineData], title : str) :
    fig = go.Figure()    
    for plotableData in data :
        lineData = {"color":plotableData.color, "width":plotableData.lineWidth}
        markerData = {"color":plotableData.color, "size":plotableData.markerSize}
        mode = ""
        if plotableData.lineWidth != 0 :
            mode = "lines"
        if plotableData.markerSize != 0 :
            if mode != "" :
                mode += "+"
            mode += "markers"

        fig.add_trace(go.Scatter(x=plotableData.x, y=plotableData.y, marker=markerData, line=lineData, name =plotableData.label, mode=mode))           
    fig.update_layout(
        showlegend=True,
        font={'size': 10},
        title={'text': title, 'font': {'size': 20}}
    )
    fig.show()