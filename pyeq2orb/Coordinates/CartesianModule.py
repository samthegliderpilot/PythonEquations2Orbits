from __future__ import annotations
import sympy as sy
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from typing import Any, Optional, cast

class Cartesian(sy.Matrix) :

    @staticmethod
    def _addXYZ(matrix) :
        matrix._x=matrix[0]
        matrix._y=matrix[1]
        matrix._z=matrix[2]

    """Represents a cartesian (x,y,z) position in space in some reference frame or vector in 
    some axes that are tracked separately.
    
    Intended to be immutable.

    Often X, Y and Z are floats, or symbols, but this is python so...
    """
    def __new__(cls, x :SymbolOrNumber, y:SymbolOrNumber, z:SymbolOrNumber) :
        self = super().__new__(cls, [[x],[y],[z]])
        """Initialize a new instance.  

        Args:
            x : The x component. 
            y : The y component.
            z : The z component.
        """
        return self

    def __init__(self, x, y, z) : # note that the __new__ function will take care of hte x,y,z
        super().__init__()
        """Initialize a new instance.  

        Args:
            x : The x component. 
            y : The y component.
            z : The z component.
        """
        Cartesian._addXYZ(self)

    def transpose(self) -> Cartesian:
        """Returns the transpose of this Cartesian with the x, y and z properties set

        Returns:
            Cartesian: The transposed Cartesian
        """
        newOne = super().transpose()
        Cartesian._addXYZ(newOne)
        return newOne

    def cross(self, other : Cartesian) -> Cartesian :
        """Crosses this Cartesian with another.

        Args:
            other (Cartesian): The other vector to cross.

        Returns:
            Cartesian: The cross product of this vector and the other one.
        """
        newOne = super().cross(other)
        Cartesian._addXYZ(newOne)
        return newOne

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
    
    def __add__(self, other : Cartesian) ->Cartesian :
        """Returns a new cartesian which is the other one added to this one.

        Args:
            other (Cartesian): The cartesian to add to this one.

        Returns:
            Cartesian: A new cartesian added to the original cartesian.
        """
        newOne = super().__add__(other)
        Cartesian._addXYZ(newOne)
        return newOne

    def __sub__(self, other :Cartesian) ->Cartesian:
        """Returns a new cartesian which is this one minus the.

        Args:
            other (Cartesian): The cartesian to subtract from this one.

        Returns:
            Cartesian: A new cartesian which is this one minus the.
        """
        newOne = super().__sub__(other)
        Cartesian._addXYZ(newOne)
        return newOne

    def __mul__(self, value: SymbolOrNumber)->Cartesian :
        """Returns a new cartesian multiplied by the supplied value.

        Args:
            value : The value to multiply by.

        Returns:
            _type_: A new cartesian multiplied by the supplied value.
        """
        newOne = super().__mul__(value)
        if len(newOne) == 3 :
            Cartesian._addXYZ(newOne)
        return newOne

    def __rmul__(self, other: SymbolOrNumber) ->Cartesian:
        newOne = super().__rmul__(other)
        Cartesian._addXYZ(newOne)
        return newOne

    def __truediv__(self, value: SymbolOrNumber) ->Cartesian:
        newOne = super().__truediv__(value)
        Cartesian._addXYZ(newOne)
        return newOne

    def EqualsWithinTolerance(self, other : Cartesian, tolerance: SymbolOrNumber) -> bool :
        """ Returns if this cartesian is equal to the other cartesian to within the passed in tolerance

        Args: 
            other (Cartesian): The motion to compare to this one
            tolerance: The tolerance

        Returns:
            True if the cartesians are equal to within the tolerance, False otherwise
        """        
        return other.shape == self.shape and abs(self[0]-other[0]) <= tolerance and abs(self[1]-other[1]) <= tolerance and abs(self[2]-other[2]) <= tolerance

    def Magnitude(self)-> SymbolOrNumber :
        """Evaluates the magnitude of this cartesian.

        Returns:
            The magnitude of this cartesian.
        """
        return self.norm()

    def Normalize(self)->Cartesian :
        """Returns a new Cartesian normalized to 1 (or as close as numerical precision will allow).

        Returns:
            Cartesian: The normalized Cartesian.
        """
        return self / self.Magnitude()
    
    @staticmethod
    def CreateSymbolic(t : Optional[sy.Symbol] = None, prefix : Optional[str] = None) -> Cartesian :
        nameFormattingCb = lambda xyz : xyz
        if prefix != None :
            prefixStr = cast(str, prefix)
            nameFormattingCb = lambda xyz: prefixStr + "{" + xyz + "}"
        if t == None :
            return Cartesian(sy.Symbol(nameFormattingCb("x"), real=True), sy.Symbol(nameFormattingCb("y"), real=True), sy.Symbol(nameFormattingCb("z"), real=True))
        else :
            return Cartesian(sy.Function(nameFormattingCb("x"), real=True)(t), sy.Function(nameFormattingCb("y"), real=True)(t), sy.Function(nameFormattingCb("z"), real=True)(t))


class MotionCartesian :
    """Holds a position and velocity Cartesian.  Intended to be immutable.
    """
    def __init__(self, position : Cartesian, velocity : Cartesian) : 
        """Initializes a new instance.

        Args:
            position (Cartesian): The position component of this motion.
            velocity (Cartesian): The velocity component of this motion. This may be None.
        """
        self._position = position
        self._velocity = velocity

    @property
    def Position(self) ->Cartesian:
        """The position of the motion.

        Returns:
            Cartesian: The position.
        """
        return self._position
    
    @property 
    def Velocity(self) ->Cartesian:
        """The velocity of the motion.  This may be None.

        Returns:
            Cartesian: The velocity of the motion.
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
        """Returns true if the included Cartesian's equal the other's Cartesian's

        Args:
            other (MotionCartesian): The other Motion to compare to.

        Returns:
            bool: True if the owned Cartesian's are equal, false otherwise.
        """
        if not isinstance(other, MotionCartesian) :
            return False
        return self.Position == other.Position and self.Velocity==other.Velocity


    def __getitem__(self, i : int) -> Cartesian:
        """Gets the i'th order of this motion.

        Args:
            i (int): The order of the motion to retrieve.

        Raises:
            Exception: If the i'th order is not available.

        Returns:
            Cartesian: The order of the motion matching i if available
        """
        if i == 0 :
            return self.Position
        if i == 1 :
            return self.Velocity # return the None, it is ok.
        raise Exception("Motions does not support indexes outside the range of 0 to 1 inclusive, but was passed " +str(i))

    def EqualsWithinTolerance(self, other : MotionCartesian, positionTolerance : SymbolOrNumber, velocityTolerance : SymbolOrNumber) -> bool :
        """ Returns if this motion is equal to the other motion to within the passed in tolerances

        Args: 
            other (MotionCartesian): The motion to compare to this one
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
    def CreateSymbolicMotion(t : Optional[sy.Symbol] = None) -> MotionCartesian :
        return MotionCartesian(Cartesian.CreateSymbolic(t), Cartesian.CreateSymbolic(t, r'\dot'))