import sympy as sy
from typing import Tuple
from pyeq2orb.Utilities.Typing import SymbolOrNumber

def convertRotationalCr3bpStateToInertial(t:SymbolOrNumber, x:SymbolOrNumber, y:SymbolOrNumber, z:SymbolOrNumber, vx:SymbolOrNumber, vy:SymbolOrNumber, vz:SymbolOrNumber) ->Tuple[SymbolOrNumber, SymbolOrNumber, SymbolOrNumber, SymbolOrNumber, SymbolOrNumber, SymbolOrNumber]:
    ct = sy.cos(t)
    st = sy.sin(t)
    xi = ct*x+st*y
    yi = -1*st*x + ct*y
    zi = z

    vxi = (vx-y)*ct+st*(vy+x)
    vyi = -1*(vx-y)*st+ct*(vy+x)
    vzi = vz

    return (xi, yi, zi, vxi, vyi, vzi)

def convertInertialToRotationalCr3bpState(t:SymbolOrNumber, x:SymbolOrNumber, y:SymbolOrNumber, z:SymbolOrNumber, vx:SymbolOrNumber, vy:SymbolOrNumber, vz:SymbolOrNumber) ->Tuple[SymbolOrNumber, SymbolOrNumber, SymbolOrNumber, SymbolOrNumber, SymbolOrNumber, SymbolOrNumber]:
    ct = sy.cos(-1*t)
    st = sy.sin(-1*t)
    xi = ct*x+st*y
    yi = -1*st*x + ct*y
    zi = z

    vxi = (vx+y)*ct+st*(vy-x)
    vyi = -1*(vx+y)*st+ct*(vy-x)
    vzi = vz

    return (xi, yi, zi, vxi, vyi, vzi)

def scaleNormalizedStateToUnnormalized(muPrimary:SymbolOrNumber, L:SymbolOrNumber, x:SymbolOrNumber, y:SymbolOrNumber, z:SymbolOrNumber, vx:SymbolOrNumber, vy:SymbolOrNumber, vz:SymbolOrNumber)->Tuple[SymbolOrNumber, SymbolOrNumber, SymbolOrNumber, SymbolOrNumber, SymbolOrNumber, SymbolOrNumber]:   
    T = 2*sy.pi*sy.sqrt(L**3/muPrimary)
    V = sy.sqrt(muPrimary/L)

    xs = L*x
    ys = L*y
    zs = L*z

    vxs = V*vx
    vys = V*vy
    vzs = V*vz

    return (xs, ys, zs, vxs, vys, vzs)

def scaleUnNormalizedStateToNormalized(muPrimary:SymbolOrNumber, L:SymbolOrNumber, x:SymbolOrNumber, y:SymbolOrNumber, z:SymbolOrNumber, vx:SymbolOrNumber, vy:SymbolOrNumber, vz:SymbolOrNumber)->Tuple[SymbolOrNumber, SymbolOrNumber, SymbolOrNumber, SymbolOrNumber, SymbolOrNumber, SymbolOrNumber]:   
    T = 2*sy.pi*sy.sqrt(L**3/muPrimary)
    V = sy.sqrt(muPrimary/L)

    xs = x/L
    ys = y/L
    zs = z/L

    vxs = vx/V
    vys = vy/V
    vzs = vz/V

    return (xs, ys, zs, vxs, vys, vzs)
