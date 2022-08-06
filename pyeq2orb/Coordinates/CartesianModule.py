from __future__ import annotations
import sympy as sy

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
    def __new__(cls, x, y, z) :
        self = super().__new__(cls, [[x],[y],[z]])
        """Initialize a new instance.  

        Args:
            x : The x component. 
            y : The y component.
            z : The z component.
        """
        return self

    def __init__(self, x, y, z) :
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

    def cross(self, other) -> Cartesian :
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
    def X(self) :
        """Gets the X component.

        Returns:
            The X Component.
        """
        return self._x

    @property
    def Y(self) :
        """Gets the Y component.

        Returns:
            The Y Component.
        """        
        return self._y

    @property
    def Z(self) :
        """Gets the Z component.

        Returns:
            The Z Component.
        """        
        return self._z


    def __add__(self, other) :
        """Returns a new cartesian which is the other one added to this one.

        Args:
            other (Cartesian): The cartesian to add to this one.

        Returns:
            Cartesian: A new cartesian added to the original cartesian.
        """
        newOne = super().__add__(other)
        Cartesian._addXYZ(newOne)
        return newOne

    def __sub__(self, other) :
        """Returns a new cartesian which is this one minus the.

        Args:
            other (Cartesian): The cartesian to subtract from this one.

        Returns:
            Cartesian: A new cartesian which is this one minus the.
        """
        newOne = super().__sub__(other)
        Cartesian._addXYZ(newOne)
        return newOne

    def __mul__(self, value) :
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

    def __rmul__(self, other) :
        newOne = super().__rmul__(other)
        Cartesian._addXYZ(newOne)
        return newOne

    def __truediv__(self, value) :
        newOne = super().__truediv__(value)
        Cartesian._addXYZ(newOne)
        return newOne

    def EqualsWithinTolerance(self, other : Cartesian, tolerance: float) :
        return other.shape == self.shape and abs(self[0]-other[0]) <= tolerance and abs(self[1]-other[1]) <= tolerance and abs(self[2]-other[2]) <= tolerance

    def Magnitude(self) :
        """Evaluates the magnitude of this cartesian.

        Returns:
            The magnitude of this cartesian.
        """
        return self.norm()

    def Normalize(self) :
        """Returns a new Cartesian normalized to 1 (or as close as numerical precision will allow).

        Returns:
            Cartesian: The normalized Cartesian.
        """
        return self / self.Magnitude()

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

    def __eq__(self, other) ->bool:
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

    def EqualsWithinTolerance(self, other, posTolerance, velTolerance) :
        if type(other) != type(self):
            return False
        return self.Position.EqualsWithinTolerance(other.Position, posTolerance) and self.Velocity.EqualsWithinTolerance(other.Velocity, velTolerance)


# def Plot3DOrbit(x, y, z) :
#     from mpl_toolkits import mplot3d
#     import numpy as np
#     import matplotlib.pyplot as plt
#     def _set_axes_radius(ax, origin, radius):
#         x, y, z = origin
#         ax.set_xlim3d([x - radius, x + radius])
#         ax.set_ylim3d([y - radius, y + radius])
#         ax.set_zlim3d([z - radius, z + radius])
#     def set_axes_equal(ax: plt.Axes):
#         """Set 3D plot axes to equal scale.

#         Make axes of 3D plot have equal scale so that spheres appear as
#         spheres and cubes as cubes.  Required since `ax.axis('equal')`
#         and `ax.set_aspect('equal')` don't work on 3D.
#         """
#         limits = np.array([
#             ax.get_xlim3d(),
#             ax.get_ylim3d(),
#             ax.get_zlim3d(),
#         ])
#         origin = np.mean(limits, axis=1)
#         radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
#         _set_axes_radius(ax, origin, radius)


    # fig = plt.figure()
    # figsize=plt.figaspect(1)
    # ax = plt.axes(projection ='3d', proj_type='ortho')
    # ax.auto_scale_xyz(x, y)
    # ax.plot3D(x, y, z, 'red')
    # ax.set_title('Orbit')

    # u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    # x = np.cos(u)*np.sin(v)*6378137
    # y = np.sin(u)*np.sin(v)*6378137
    # z = np.cos(v)*6378137
    # ax.plot_surface(x, y, z, color="b")
    # ax.set_box_aspect([1,1,1])
    # set_axes_equal(ax)
    # return plt
