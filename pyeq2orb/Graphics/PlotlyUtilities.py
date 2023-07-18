from typing import List
import pyeq2orb.Graphics.Primitives as prim
from pandas import DataFrame # type: ignore
import plotly.graph_objects as go # type: ignore
import numpy as np

class PlotlyDataAndFramesAccumulator :
    def __init__(self):
        self.data = []
        self.frames = []
        
    @staticmethod
    def CreatePlotlyEphemerisDataFrame(ephemeris : prim.EphemerisArrays) :
        x = ephemeris.X
        y = ephemeris.Y
        z = ephemeris.Z
        t = ephemeris.T
        df = DataFrame({"x": x, "y":y, "z":z, "t":t})
        return df
        
    def AddMarkerPrimitives(self, tArray, markerPrimitives : List[prim.MarkerPrimitive]) :
        mapOfEphemerises = {}
        traceArray = []
        i=0
        for mark in markerPrimitives :
            mapOfEphemerises[mark]=PlotlyDataAndFramesAccumulator.CreatePlotlyEphemerisDataFrame(mark.ephemeris)
            traceArray.append(i)
            i=i+1
        for k in range(0, len(tArray)) :
            dataForThisFrame = []
            for marker in markerPrimitives :
                eph = mapOfEphemerises[marker]
                pt = go.Scatter3d(x=np.array(eph['x'][k]), y=np.array(eph['y'][k]), z=np.array(eph['z'][k]), mode="markers", marker=dict(color=marker.color, size=marker.size))
                dataForThisFrame.append(pt)
            self.frames.append(go.Frame(data=dataForThisFrame, traces= traceArray, name=f'frame{k}'))

    def AddLinePrimitive(self, pathPrimitve : prim.PathPrimitive) :
        df = PlotlyDataAndFramesAccumulator.CreatePlotlyEphemerisDataFrame(pathPrimitve.ephemeris)
        theLine = go.Scatter3d(x=df["x"], y=df["y"], z=df["z"], mode="lines", line=dict(color=pathPrimitve.color, width=pathPrimitve.width))
        self.data.append(theLine)

    def AddLinePrimitives(self, pathPrimitves : List[prim.PathPrimitive]) :
        for prim in pathPrimitves :
            self.AddLinePrimitive(prim)

    def AddScalingPoints(self, primitives : List[prim.Primitive]) :
        maxVal = -1.0
        for prim in primitives :
            if prim.maximumValue() > maxVal :
                maxVal = prim.maximumValue()
        
        markers = go.Scatter3d(name="",
        visible=True,
        showlegend=False,
        opacity=0,
        hoverinfo='none',
        x=[0,maxVal],
        y=[0,maxVal],
        z=[0,maxVal])
        self.data.append(markers)