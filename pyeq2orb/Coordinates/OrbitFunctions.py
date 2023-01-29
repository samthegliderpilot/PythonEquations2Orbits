from pyeq2orb.Coordinates.CartesianModule import MotionCartesian
import sympy as sy

def CreateComplicatedRicToInertialMatrix(asCart : MotionCartesian) :
    
    r = asCart.Position
    v = asCart.Velocity
    i_r = r.Normalize()
    rxv = r.cross(v)
    i_c = rxv.Normalize()
    i_t = i_c.cross(i_r)
    def simp(item) :
        return item.simplify()
    return sy.Matrix([i_r.applyfunc(simp).transpose(), i_t.applyfunc(simp).transpose(), i_c.applyfunc(simp).transpose()]).transpose()
