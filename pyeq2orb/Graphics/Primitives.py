from typing import List
import numpy as np

class EphemerisArrays :
    def __init__(self) :
        self._t = []
        self._x = []
        self._y = []
        self._z = []

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
    def T(self) -> np.array :
        return self._t

    @property
    def X(self) -> np.array :
        return self._x

    @property
    def Y(self) -> np.array :
        return self._y

    @property
    def Z(self) -> np.array :
        return self._z     

    def GetMaximumValue(self) :
        return max([max(self.X, key=abs), max(self.Y, key=abs), max(self.Z, key=abs)])

class Primitive :
    def __init__(self) :
        pass
        self._color = "#000000"

    def maximumValue(self) -> float : 
        return self.maximumValueFromEphemeris(self.ephemeris)

    def maximumValueFromEphemeris(self, ephemeris):
        return ephemeris.GetMaximumValue()

    @property
    def color(self) : 
        return self._color

    @color.setter
    def color(self, value) :
        self._color = value        


class PathPrimitive(Primitive) :
    def __init__(self, ephemeris = EphemerisArrays()) :
        Primitive.__init__(self)
        self._ephemeris = ephemeris
        self._color = '#0000ff'
        self._width = 1

    @property
    def ephemeris(self) -> EphemerisArrays :
        return self._ephemeris     

    @property
    def width(self) : 
        return self._width

    @width.setter
    def set_width(self, value) :
        self._width = value 

    def maximumValue(self) -> float:
        return super().maximumValue()

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

class XAndYPlottableLineData :
    """
    A simple class grouping together common data needed to plot a 2D line.
    """
    def __init__(self, x : List[float], y: List[float], label : str, color : object, lineWidth=0, markerSize=0):
        self.x = x
        self.y = y
        self.label = label
        self.color = color
        self.lineWidth = lineWidth
        self.markerSize = markerSize        