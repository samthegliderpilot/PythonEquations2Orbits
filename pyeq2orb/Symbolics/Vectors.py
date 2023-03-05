import sympy as sy

class Vector(sy.Matrix) :
    """
    Some day I hope that this will be an instance of Sympy's Matrix that is meant to be a single column Vector, but
    for now it is just some static helper functions to work better with column Matrices that we want to treat as vectors.
    """
    # def __new__(cls, n) :
    #     obj = sy.Matrix.__new__(cls, n, 1)
    #     cls.length = n
    #     return obj

    # def __getitem__(self, i) :
    #     return self[i,1]

    # def __len__(self) :         
    #     return self.length
        
    @staticmethod
    def toArray(mat) :
        """Takes the (assume to be) single column Matrix and returns a python list of the values.

        Args:
            mat (sy.Matrix): A sympy matrix (with 1 column of data)

        Returns:
            list: the values of the column Matrix as a list            
        """
        arr = []
        for i in range(0, len(mat)) :
            arr.append(mat[i])
        return arr

    @staticmethod
    def addVectorPropertiesToMatrix(mat : sy.Matrix, n : int) :
        """Adds members to the matrix to make it easier to work with as a single column Vector

        Args:
            mat (sy.Matrix): Sympy Matrix
            n (int): How long the vector is
        """
        mat.__len__ = n        
        mat.toArray = lambda : Vector.toArray(mat)
        mat.Magnitude = lambda: Vector.Magnitude(mat)

    @staticmethod
    def zeros(n : int) :
        """Creates a Vector of length n with each element equal to 0

        Args:
            n (int): How long the Vector is

        Returns:
            sy.Matrix: A column Matrix with all the values equal to 0
        """
        mat = sy.Matrix.zeros(n, 1)
        Vector.addVectorPropertiesToMatrix(mat, n)
        return mat

    @staticmethod
    def fromArray(arr : list)  :
        """Static function to take a list and return a Vector.

        Args:
            arr (list): A list to turn into a Vector.

        Returns:
            Vector: A vector made from the arguments
        """
        n=len(arr)
        mat = Vector.zeros(n)
        for i in range(0, n) :
            mat[i, 0] = arr[i]        
        Vector.addVectorPropertiesToMatrix(mat, n)
        return mat

    @staticmethod
    def fromValues(*args) :
        """Static function to take the arguments and return a Vector.

        Args:
            arr (list): The arguments to turn into a Vector.

        Returns:
            Vector: A vector made from the arguments
        """
        return Vector.fromArray(args)

    def Magnitude(self) :
        """Evaluates the magnitude (2 norm) of this Vector.

        Returns:
            object: The magnitude of this Vector.  If the vector is made up of doubles, then it will be that magnitude.  But it 
            could also be symbolic expressions, and this will create that expressions too.
        """
        ansSquared = 0
        for i in range(0, len(self)) :
            ansSquared = ansSquared + self[i]**2
        return ansSquared**(1/2)
