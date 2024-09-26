from __future__ import annotations
from typing import List, cast, Optional, Tuple, Sequence, Iterator
import numpy as np 
from pyeq2orb.Coordinates.CartesianModule import Cartesian
from pyeq2orb.Utilities.Typing import SymbolOrNumber

class EphemerisArrays :
    def __init__(self) :
        self._t = [] #type: List[float]
        self._x = [] #type: List[float]
        self._y = [] #type: List[float]
        self._z = [] #type: List[float]

    def InitFromMotions(self, timeArray, motionArray) :
        self.InitFromEphemeris(timeArray, [motion.Position for motion in motionArray])

    def AppendFromMotions(self, timeArray, motionArray) :
        self.AppendFromEphemeris(timeArray, [motion.Position for motion in motionArray])

    def InitFromEphemeris(self, timeArray, cartesianArray) :
        self.AppendFromEphemeris(timeArray, cartesianArray)

    def AppendFromEphemeris(self, timeArray, cartesianArray) :
        self._t = np.array(timeArray)
        self._x = np.array([float(pos.X) for pos in cartesianArray])
        self._y = np.array([float(pos.Y) for pos in cartesianArray])
        self._z = np.array([float(pos.Z) for pos in cartesianArray])

    @property
    def T(self) -> List[float] :
        return self._t

    @property
    def X(self) -> List[float] :
        return self._x

    @property
    def Y(self) -> List[float] :
        return self._y

    @property
    def Z(self) -> List[float] :
        return self._z     

    def AddMotion(self, t : float, position : Cartesian) :
        self.AppendValues(t, cast(float, position.X), cast(float, position.Y), cast(float, position.Z))

    def AppendValues(self, t : float, x : float, y : float, z : float) :
        self.T.append(t)
        self.X.append(x)
        self.Y.append(y)
        self.Z.append(z)

    def ExtendValues(self, t : List[float], x : List[float], y : List[float], z : List[float]) :
        self.T.extend(t)
        self.X.extend(x)
        self.Y.extend(y)
        self.Z.extend(z)

    def GetMaximumAbsoluteValue(self) :
        return max([max(self.X, key=abs), max(self.Y, key=abs), max(self.Z, key=abs)])

    def BoundsX(self):
        return (min(self.X), max(self.X))

    def BoundsY(self):
        return (min(self.Y), max(self.Y))

    def BoundsZ(self):
        return (min(self.Z), max(self.Z))


    @staticmethod
    def GetEquidistantBoundsForEvenPlotting(ephemerisList : List[EphemerisArrays]) :
        xBounds = ephemerisList[0].BoundsX()
        yBounds = ephemerisList[0].BoundsY()
        zBounds = ephemerisList[0].BoundsZ()
        minX = xBounds[0]
        maxX = xBounds[1]
        minY = yBounds[0]
        maxY = yBounds[1]
        minZ = zBounds[0]
        maxZ = zBounds[1]
        first= True
        for planet in ephemerisList :           
            if not first:
                first = False
                xBounds = planet.BoundsX()
                yBounds = planet.BoundsY()
                zBounds = planet.BoundsZ()
                if xBounds[0] < minX:
                    minX = xBounds[0]
                if xBounds[1] > maxX:
                    maxX = xBounds[1]
                if yBounds[0] < minY:
                    minY = yBounds[0]
                if yBounds[1] > maxY:
                    maxY = yBounds[1]
                if zBounds[0] < minZ:
                    minZ = zBounds[0]
                if zBounds[1] > maxZ:
                    maxZ = zBounds[1]   

        # make the scaling item
        spanX = maxX-minX
        spanY = maxY-minY
        spanZ = maxZ-minZ
        halfSpan = max([spanX, spanY, spanZ])/2
        spanToUse = halfSpan*1.25
        centerX = minX + (maxX-minX)/2
        centerY = minY + (maxY-minY)/2
        centerZ = minZ + (maxZ-minZ)/2

        x=(centerX-spanToUse, centerX+spanToUse)
        y=(centerY-spanToUse, centerY+spanToUse)
        z=(centerZ-spanToUse, centerZ+spanToUse)

        return x, y, z

class Primitive :
    def __init__(self) :
        self._color = "#000000"
        self._id = ""
        self._ephemeris = EphemerisArrays()

    def maximumAbsoluteValue(self) -> float : 
        return self.maximumValueFromEphemeris(self._ephemeris)

    def maximumValueFromEphemeris(self, ephemeris):
        return ephemeris.GetMaximumAbsoluteValue()

    @property
    def color(self) : 
        return self._color

    @color.setter
    def color(self, value) :
        self._color = value        

    @property
    def id(self) :
        return self._id
    
    @id.setter
    def id(self, value) :
        self._id = value

    @staticmethod
    def GetEquidistantBoundsForEvenPlotting(primitiveList : Sequence[Primitive]) :        
        return EphemerisArrays.GetEquidistantBoundsForEvenPlotting([prim._ephemeris for prim in primitiveList])


class PathPrimitive(Primitive) :
    def __init__(self, ephemeris = EphemerisArrays(), color: Optional[str]=None, width : int=2) :
        Primitive.__init__(self)
        self._ephemeris = ephemeris
        if(color == None) :
            color = '#0000ff'
        self._color = color
        self._width = width

    @property
    def ephemeris(self) -> EphemerisArrays :
        return self._ephemeris     

    @property
    def width(self) ->int: 
        return self._width

    @width.setter
    def width(self, value : int) :
        self._width = value 

    def maximumAbsoluteValue(self) -> float:
        return super().maximumAbsoluteValue()

class MarkerPrimitive(Primitive) :
    def __init__(self, ephemeris = EphemerisArrays()) :
        Primitive.__init__(self)
        self._ephemeris = ephemeris
        self._color = '#000000'
        self._size = 1

    @property
    def ephemeris(self) -> EphemerisArrays :
        return self._ephemeris     

    @property
    def size(self) : 
        return self._size

    @size.setter
    def size(self, value) :
        self._size = value
  

class Sphere(Primitive) :
    def __init__(self, ephemeris = EphemerisArrays()) :
        Primitive.__init__(self)
        self._ephemeris = ephemeris
        self._color = '#000000'
        self._radius = 1

    @property
    def ephemeris(self) -> EphemerisArrays :
        return self._ephemeris     

    @property
    def radius(self) : 
        return self._radius

    @radius.setter
    def radius(self, value) :
        self._radius = value


class PlanetPrimitive(MarkerPrimitive, PathPrimitive) :
    def __init__(self, positionCartesians, markerSize, lineWidth, color, planetRadius, name):
        PathPrimitive.__init__(self, positionCartesians)
        MarkerPrimitive.__init__(self, positionCartesians)
        self._color = color
        self._size = markerSize
        self._width = lineWidth
        self._radius = planetRadius
        self.name = name

    @property
    def radius(self) :
        return self._radius

    @radius.setter
    def radius(self, value) :
        self._radius = value


    @staticmethod
    def fromMotionEphemeris(tArray, motions, color):
        ephemeris = EphemerisArrays()
        ephemeris.InitFromMotions(tArray, motions)
        planetPath = PathPrimitive(ephemeris)
        planetPath.color = color
        return planetPath

class XAndYPlottableLineData :
    """
    A simple class grouping together common data needed to plot a 2D line.
    """
    def __init__(self, x : Iterator[float], y: Iterator[float], label : str, color : object, lineWidth=0, markerSize=0):
        self.x = x
        self.y = y
        self.label = label
        self.color = color
        self.lineWidth = lineWidth
        self.markerSize = markerSize        

def makeSphere(x, y, z, radius, resolution=10):
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    #colors = ['#00ff00']*len(X)
    #size = [2]*len(X)
    return (X, Y, Z)#, colors, size)        