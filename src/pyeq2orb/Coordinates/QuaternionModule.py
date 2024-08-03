from __future__ import annotations
import sympy as sy
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from typing import Any, Optional, cast, Union
from pyeq2orb.Coordinates.CartesianModule import Cartesian

class Quaternion(sy.Matrix) :

    @staticmethod
    def _addWXYZ(matrix) :
        matrix.wx=matrix[0]
        matrix._x=matrix[0]
        matrix._y=matrix[1]
        matrix._z=matrix[2]

    """Represents a Quaternion (x,y,z) position in space in some reference frame or vector in 
    some axes that are tracked separately.
    
    Intended to be immutable.

    Often W, X, Y and Z are floats, or symbols, but this is python so...
    """
    def __new__(cls, w:SymbolOrNumber, x :SymbolOrNumber, y:SymbolOrNumber, z:SymbolOrNumber) :
        self = super().__new__(cls, [[w],[x],[y],[z]])
        """Initialize a new instance.  

        Args:
            w : The scalar component
            x : The x component. 
            y : The y component.
            z : The z component.
        """
        return self

    def __init__(self, w, x, y, z) : # note that the __new__ function will take care of hte x,y,z
        super().__init__()
        """Initialize a new instance.  

        Args:
            w : The scalar component
            x : The x component. 
            y : The y component.
            z : The z component.
        """
        Quaternion._addWXYZ(self)

    # def transpose(self) -> Quaternion:
    #     """Returns the transpose of this Quaternion with the x, y and z properties set

    #     Returns:
    #         Quaternion: The transposed Quaternion
    #     """
    #     newOne = super().transpose()
    #     Quaternion._addWXYZ(newOne)
    #     return newOne

    # def cross(self, other : Quaternion) -> Quaternion :
    #     """Crosses this Quaternion with another.

    #     Args:
    #         other (Quaternion): The other vector to cross.

    #     Returns:
    #         Quaternion: The cross product of this vector and the other one.
    #     """
    #     newOne = super().cross(other)
    #     Quaternion._addWXYZ(newOne)
    #     return newOne

    @property
    def W(self) ->SymbolOrNumber:
        """Gets the W component.

        Returns:
            The W Component.
        """
        return self._w #type: ignore

    @property
    def X(self) ->SymbolOrNumber:
        """Gets the X component.

        Returns:
            The X Component.
        """
        return self._x #type: ignore

    @property
    def Y(self) ->SymbolOrNumber:
        """Gets the Y component.

        Returns:
            The Y Component.
        """        
        return self._y#type: ignore

    @property
    def Z(self) ->SymbolOrNumber:
        """Gets the Z component.

        Returns:
            The Z Component.
        """        
        return self._z#type: ignore
    
    # def __add__(self, other : Quaternion) ->Quaternion :
    #     """Returns a new Quaternion which is the other one added to this one.

    #     Args:
    #         other (Quaternion): The Quaternion to add to this one.

    #     Returns:
    #         Quaternion: A new Quaternion added to the original Quaternion.
    #     """
    #     newOne = super().__add__(other)
    #     Quaternion._addXYZ(newOne)
    #     return newOne

    # def __sub__(self, other :Quaternion) ->Quaternion:
    #     """Returns a new Quaternion which is this one minus the.

    #     Args:
    #         other (Quaternion): The Quaternion to subtract from this one.

    #     Returns:
    #         Quaternion: A new Quaternion which is this one minus the.
    #     """
    #     newOne = super().__sub__(other)
    #     Quaternion._addXYZ(newOne)
    #     return newOne

    # def __mul__(self, value: SymbolOrNumber)->Quaternion :
    #     """Returns a new Quaternion multiplied by the supplied value.

    #     Args:
    #         value : The value to multiply by.

    #     Returns:
    #         _type_: A new Quaternion multiplied by the supplied value.
    #     """
    #     newOne = super().__mul__(value)
    #     if len(newOne) == 3 :
    #         Quaternion._addXYZ(newOne)
    #     return newOne

    # def __rmul__(self, other: SymbolOrNumber) ->Quaternion:
    #     newOne = super().__rmul__(other)
    #     Quaternion._addXYZ(newOne)
    #     return newOne

    # def __truediv__(self, value: SymbolOrNumber) ->Quaternion:
    #     newOne = super().__truediv__(value)
    #     Quaternion._addXYZ(newOne)
    #     return newOne

    def EqualsWithinTolerance(self, other : Quaternion, tolerance: SymbolOrNumber) -> bool :
        """ Returns if this Quaternion is equal to the other Quaternion to within the passed in tolerance

        Args: 
            other (Quaternion): The motion to compare to this one
            tolerance: The tolerance

        Returns:
            True if the Quaternions are equal to within the tolerance, False otherwise
        """        
        return other.shape == self.shape and abs(self[0]-other[0]) <= tolerance and abs(self[1]-other[1]) <= tolerance and abs(self[2]-other[2]) <= tolerance and abs(self[3]-other[3]) <= tolerance

    def Magnitude(self)-> SymbolOrNumber :
        """Evaluates the magnitude of this Quaternion.

        Returns:
            The magnitude of this Quaternion.
        """
        return self.norm()

    def Normalize(self)->Quaternion :
        """Returns a new Quaternion normalized to 1 (or as close as numerical precision will allow).

        Returns:
            Quaternion: The normalized Quaternion.
        """
        return self / self.Magnitude()
    
    @staticmethod
    def CreateSymbolic(t : Optional[sy.Symbol] = None, prefix : Optional[str] = None) -> Quaternion :
        nameFormattingCb = lambda xyz : xyz
        if prefix != None :
            prefixStr = cast(str, prefix)
            nameFormattingCb = lambda xyz: prefixStr + "{" + xyz + "}"
        if t == None :
            return Quaternion(sy.Symbol(nameFormattingCb("w"), real=True), sy.Symbol(nameFormattingCb("x"), real=True), sy.Symbol(nameFormattingCb("y"), real=True), sy.Symbol(nameFormattingCb("z"), real=True))
        else :
            return Quaternion(sy.Function(nameFormattingCb("w"), real=True)(t), sy.Function(nameFormattingCb("x"), real=True)(t), sy.Function(nameFormattingCb("y"), real=True)(t), sy.Function(nameFormattingCb("z"), real=True)(t))


class MotionQuaternion :
    """Holds a position and velocity Quaternion.  Intended to be immutable.
    """
    def __init__(self, position : Quaternion, velocity : Cartesian) : 
        """Initializes a new instance.

        Args:
            position (Quaternion): The position component of this motion.
            velocity (Quaternion): The velocity component of this motion. This may be None.
        """
        self._position = position
        self._velocity = velocity

    @property
    def Position(self) ->Quaternion:
        """The position of the motion.

        Returns:
            Quaternion: The position.
        """
        return self._position
    
    @property 
    def Velocity(self) ->Cartesian:
        """The angular velocity of the motion.  This may be None.

        Returns:
            Quaternion: The velocity of the motion.
        """
        return self._velocity

    @property
    def Order(self) -> int :
        """Gets the order of the motion, 0 for just position, 1 for position and velocity;

        Returns:
            int: The order of the motion;
        """
        if self.Velocity == None:
            return 0
        return 1

    def __eq__(self, other : Any) ->bool:
        """Returns true if the included Quaternion's equal the other's Quaternion's

        Args:
            other (MotionQuaternion): The other Motion to compare to.

        Returns:
            bool: True if the owned Quaternion's are equal, false otherwise.
        """
        if not isinstance(other, MotionQuaternion) :
            return False
        return self.Position == other.Position and self.Velocity==other.Velocity


    def __getitem__(self, i : int) -> Union[Quaternion, Cartesian]:
        """Gets the i'th order of this motion.

        Args:
            i (int): The order of the motion to retrieve.

        Raises:
            Exception: If the i'th order is not available.

        Returns:
            Quaternion: The order of the motion matching i if available
        """
        if i == 0 :
            return self.Position
        if i == 1 :
            return self.Velocity # return the None, it is ok.
        raise Exception("Motions does not support indexes outside the range of 0 to 1 inclusive, but was passed " +str(i))

    def EqualsWithinTolerance(self, other : MotionQuaternion, positionTolerance : SymbolOrNumber, velocityTolerance : SymbolOrNumber) -> bool :
        """ Returns if this motion is equal to the other motion to within the passed in tolerances

        Args: 
            other (MotionQuaternion): The motion to compare to this one
            positionTolerance: The position tolerance
            velocityTolerance: The velocity tolerance

        Returns:
            True if the motions are equal to within the tolerance, False otherwise
        """
        if type(other) != type(self):
            return False
        if other == None :
            return False
        posEqual = self.Position.EqualsWithinTolerance(other.Position, positionTolerance)
        if other.Velocity == None and self.Velocity == None :
            return posEqual
        if other.Velocity == None :
            return False
        return posEqual and self.Velocity.EqualsWithinTolerance(other.Velocity, velocityTolerance)
    

    @staticmethod
    def CreateSymbolicMotion(t : Optional[sy.Symbol] = None) -> MotionQuaternion :
        return MotionQuaternion(Quaternion.CreateSymbolic(t), Cartesian.CreateSymbolic(t, r'\dot'))