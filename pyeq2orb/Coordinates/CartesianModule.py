class Cartesian :
    """Represents a cartesian (x,y,z) position in space in some reference frame or vector in 
    some axes that are tracked separately.
    
    Intended to be immutable.

    Often X, Y and Z are floats, or symbols, but this is python so...
    """
    def __init__(self, x, y, z) :
        """Initialize a new instance.  

        Args:
            x : The x component. 
            y : The y component.
            z : The z component.
        """
        self._x=x
        self._y=y
        self._z=z

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

    def __eq__(self, other) ->bool:
        """Returns true if the components of the other Cartesian equals the components of this Cartesian.

        Args:
            other (Cartesian): The other Cartesian to compare to.

        Returns:
            bool: True if the components are equal, false otherwise.
        """
        if not isinstance(other, Cartesian) :
            return False
        return self.X == other.X and self.Y==other.Y and self.Z==other.Z

    def Magnitude(self) :
        """Evaluates the magnitude of this cartesian.

        Returns:
            The magnitude of this cartesian.
        """
        return (self.X**2+ self.Y**2+ self.Z**2)**(1.0/2.0)

    def Cross(self, other):
        """Creates a new Cartesian that is this Cartesian crossed the the other one.

        Args:
            other (Cartesian): The other cartesian to cross with this one.

        Returns:
            Cartesian: A new Cartesian that is this Cartesian crossed the the other one.
        """
        return Cartesian(self.Y*other.Z - self.Z*other.Y, self.Z*other.X - self.X*other.Z, self.X*other.Y-self.Y*other.X)

    def Dot(self, other) :
        """Creates a new Cartesian that is this Cartesian dotted the the other one.

        Args:
            other (Cartesian): The other cartesian to dot with this one.

        Returns:
            scalar: A new Cartesian that is this Cartesian dotted the the other one.
        """        
        return self.X*other.X + self.Y * other.Y + self.Z*other.Z

    def __add__(self, other) :
        """Returns a new cartesian which is the other one added to this one.

        Args:
            other (Cartesian): The cartesian to add to this one.

        Returns:
            Cartesian: A new cartesian added to the original cartesian.
        """
        return Cartesian(self.X+other.X, self.Y+other.Y, self.Z+other.Z)

    def __sub__(self, other) :
        """Returns a new cartesian which is this one minus the.

        Args:
            other (Cartesian): The cartesian to subtract from this one.

        Returns:
            Cartesian: A new cartesian which is this one minus the.
        """
        return Cartesian(self.X-other.X, self.Y-other.Y, self.Z-other.Z)

    def __mul__(self, value) :
        """Returns a new cartesian multiplied by the supplied value.

        Args:
            value : The value to multiply by.

        Returns:
            _type_: A new cartesian multiplied by the supplied value.
        """
        return Cartesian(self.X*value, self.Y*value, self.Z*value)

    def __truediv__(self, value) :
        """Divides this cartesian by some value.

        Args:
            value : The divisor

        Returns:
            Cartesian: The divided cartesian.
        """
        return Cartesian(self.X/value, self.Y/value, self.Z/value)        

    def Normalize(self) :
        """Returns a new Cartesian normalized to 1 (or as close as numerical precision will allow).

        Returns:
            Cartesian: The normalized Cartesian.
        """
        return self / self.Magnitude()

    def __getitem__(self, pos) :
        i = 0
        j = 0
        if isinstance(pos, int) :
            i = pos
        else :
            j,i = pos # tuple
        if not j==0 :
            raise Exception("Cartesian are treated as a column matrix, and does not support a column other than 0, but was passed " + str(j))

        if i == 0 :
            return self.X
        if i == 1:
            return self.Y
        if i == 2:
            return self.Z
        raise Exception("Cartesian does not support indexes outside the range of 0 to 2 inclusive, but was passed " +str(i))

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
