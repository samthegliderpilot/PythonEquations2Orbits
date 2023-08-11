from pyeq2orb.Coordinates.CartesianModule import MotionCartesian
import sympy as sy

def CreateComplicatedRicToInertialMatrix(asCart : MotionCartesian) -> sy.Matrix :
    def simp(item : sy.Expr) :
        return item.simplify()
    r = asCart.Position
    v = asCart.Velocity
    i_r = r.Normalize().applyfunc(simp)
    rxv = r.cross(v).applyfunc(simp)
    i_c = rxv.Normalize().applyfunc(simp)
    i_t = i_c.cross(i_r).applyfunc(simp)
    
    return sy.Matrix([[i_r[0], i_r[1], i_r[2]], [i_t[0], i_t[1], i_t[2]], [i_c[0], i_c[1], i_c[2]]]).transpose()
