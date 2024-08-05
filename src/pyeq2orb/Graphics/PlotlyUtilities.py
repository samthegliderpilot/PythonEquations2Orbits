from typing import List, Dict, Sequence, Optional, Iterable
import pyeq2orb.Graphics.Primitives as prim
from pandas import DataFrame # type: ignore
import plotly.graph_objects as go # type: ignore
import numpy as np
from scipy.interpolate import splev, splrep #type: ignore
import plotly.express as px#type: ignore


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

    def AddLinePrimitive(self, pathPrimitive : prim.PathPrimitive) :
        df = PlotlyDataAndFramesAccumulator.CreatePlotlyEphemerisDataFrame(pathPrimitive.ephemeris)
        theLine = go.Scatter3d(x=df["x"], y=df["y"], z=df["z"], mode="lines", line=dict(color=pathPrimitive.color, width=pathPrimitive.width))
        self.data.append(theLine)

    def AddLinePrimitives(self, pathPrimitives : List[prim.PathPrimitive]) :
        for prim in pathPrimitives :
            self.AddLinePrimitive(prim)

    
def CreateScalingMarker(primitives : Sequence[prim.Primitive]) ->go.Scatter3d :
    xB, yB, zB = prim.Primitive.GetEquidistantBoundsForEvenPlotting(primitives)
    
    scalingMarker = go.Scatter3d(name="",
        visible=True,
        showlegend=False,
        opacity=0, # full transparent
        hoverinfo='none',
        x=[xB[0], xB[1]],
        y=[yB[0], zB[1]],
        z=[yB[0], zB[1]])        
    return scalingMarker

def PlotAndAnimatePlanetsWithPlotly(title : str, wanderers : List[prim.PathPrimitive], tArray : Iterable[float], thrustVector : Optional[List[go.Scatter3d]]=None) :
    lines = []

    #animation arrays
    xArrays = []
    yArrays = []
    zArrays = []

    for planet in wanderers :
        dataDict = DataFrame({"x":planet.ephemeris.X, "y":planet.ephemeris.Y, "z": planet.ephemeris.Z })
        thisLine = go.Scatter3d(x=dataDict["x"], y=dataDict["y"], z=dataDict["z"], mode="lines", line=dict(color=planet.color, width=planet.width))
        
        lines.append(thisLine)     

        # for the animation, we can only have 1 scatter_3d and we need to shuffle all of the 
        # points for all of the planets to be at the same time 
        xForAni = splev(tArray, splrep(planet.ephemeris.T, planet.ephemeris.X))
        yForAni = splev(tArray, splrep(planet.ephemeris.T, planet.ephemeris.Y))
        zForAni = splev(tArray, splrep(planet.ephemeris.T, planet.ephemeris.Z))
        xArrays.append(xForAni)
        yArrays.append(yForAni)
        zArrays.append(zForAni)


    dataDictionary = {"x":[], "y":[], "z":[], "t":[], "color":[], "size":[]} #type: Dict[str, List[float]]
    t = dataDictionary["t"]
    k = 0
    for step in tArray :
        p = 0
        for cnt in wanderers:
            t.append(step/86400)
            dataDictionary["x"].append(xArrays[p][k])
            dataDictionary["y"].append(yArrays[p][k])
            dataDictionary["z"].append(zArrays[p][k])
            dataDictionary["color"].append(cnt.color)
            dataDictionary["size"].append(7)
            p=p+1
        k=k+1
    
    fig = px.scatter_3d(dataDictionary, title=title, x="x", y="y", z="z", animation_frame="t", color="color", size="size")    

    # make the scaling item
    scalingMarker = CreateScalingMarker(wanderers)    
    fig.add_trace(scalingMarker)

    for item in lines :
        fig.add_trace(item)
    if thrustVector != None : #Um, mypy, here's the ****ing null check!
        for thrust in thrustVector : #type: ignore 
            fig.add_trace(thrust)
    return fig    