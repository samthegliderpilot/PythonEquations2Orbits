from typing import List

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
        self._t = []
        self._x = []
        self._y = []
        self._z = []
        self.AppendFromEphemeris(timeArray, cartesianArray)

    def AppendFromEphemeris(self, timeArray, cartesianArray) :
        for i in range(0, len(timeArray)) :
            self._t.append(timeArray[i])
            here = cartesianArray[i]
            self._x.append(here.X)
            self._y.append(here.Y)
            self._z.append(here.Z)

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



class Primitive :
    def __init__(self) :
        pass
        #self._color = "#000000"

    def MaxumumValue(self) -> float : 
        return 0.0

    # @property
    # def Color(self) : 
    #     return self._color

    # @Color.setter
    # def set_Color(self, value) :
    #     self._color = value        


class PathPrimitive(Primitive) :
    def __init__(self, ephemeris = EphemerisArrays()) :
        Primitive.__init__(self)
        self._ephemeris = ephemeris
        self._color = '#0000ff'
        self._width = 1

    @property
    def Ephemeris(self) -> List[float] :
        return self._ephemeris     

    @property
    def Width(self) : 
        return self._width

    @Width.setter
    def set_Width(self, value) :
        self._width = value
    @property
    def Color(self) : 
        return self._color

    @Color.setter
    def set_Color(self, value) :
        self._color = value     

class MarkerPrimitive(Primitive) :
    def __init__(self, ephemeris = EphemerisArrays()) :
        Primitive.__init__(self)
        self._ephemeris = ephemeris
        self._color = '#000000'
        self._size = 1

    @property
    def Ephemeris(self) -> List[float] :
        return self._ephemeris     

    @property
    def Size(self) : 
        return self._size

    @Size.setter
    def set_Size(self, value) :
        self._size = value
    @property
    def Color(self) : 
        return self._color

    @Color.setter
    def set_Color(self, value) :
        self._color = value     

class Sphere(Primitive) :
    def __init__(self, ephemeris = EphemerisArrays()) :
        Primitive.__init__(self)
        self._ephemeris = ephemeris
        self._color = '#000000'
        self._radius = 1

    @property
    def Ephemeris(self) -> List[float] :
        return self._ephemeris     

    @property
    def Radius(self) : 
        return self._radius

    @Radius.setter
    def set_Radius(self, value) :
        self._radius = value
    @property
    def Color(self) : 
        return self._color

    @Color.setter
    def set_Color(self, value) :
        self._color = value     