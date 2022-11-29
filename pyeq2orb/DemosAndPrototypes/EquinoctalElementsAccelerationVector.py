#%%
import sympy as sy
import os
import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0]))) # need to import 2 directories up (so pyeq2orb is a subfolder)

from pyeq2orb.ForceModels.TwoBodyForce import CreateTwoBodyMotionMatrix, CreateTwoBodyList
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Coordinates.KeplerianModule import KeplerianElements
from pyeq2orb.Coordinates.EquinoctialElements import EquinoctialElements, CreateSymbolicElements
import JupyterHelper as jh

jh.printMarkdown("# Non-TwoBody Acceleration Vector in terms of Equinoctial Elements")
jh.printMarkdown("While working with an optimization problem that uses Equinoctial elements, I found that the acceleration vector was still mostly in terms of cartesian elements.  In many cases, that isn't a problem, but for my uses it was something that I would prefer was also in terms of Equinoctial elements as well.  So, let's do that!")

t = sy.Symbol('t')
equiElements = CreateSymbolicElements(t)
cartesian = equiElements.ToMotionCartesian()
jh.showEquation("R", cartesian.Position)
jh.showEquation("V", cartesian.Velocity)
rMag = equiElements.PeriapsisRadius/(1+equiElements.EccentricityCosTermF*sy.cos(equiElements.TrueLongitude) + equiElements.EccentricitySinTermG*sy.sin(equiElements.TrueLongitude))

def expandAndSimplify(s) :
    return s.simplify()

jh.printMarkdown("From source 1, the unit vectors to convert LVLH unit vectors to inertial is:")
jh.showEquation("i_r", sy.Matrix([[sy.Symbol("x")/sy.Symbol("r_{mag}")],[sy.Symbol("y")/sy.Symbol("r_{mag}")],[sy.Symbol("z")/sy.Symbol("r_{mag}")]]))
jh.printMarkdown("And it also defines the magnitude of r as ")
jh.showEquation("r_{mag}", rMag)
jh.printMarkdown("We get a useful expression for the radial unit vector in terms of equinoctial elements")
iradial = (cartesian.Position/rMag).applyfunc(expandAndSimplify) # I doubt we will get it any simplier than this (maybe with w and s^2, but eh...)
jh.showEquation("i_r", iradial)

# note that Applied Nonsingular Astrodynamics has most of the rotation matrix.  Undo a 90 deg addition and also multiply by the true longitude, and that will probably work

#%%
rxv = cartesian.Position.cross(cartesian.Velocity).applyfunc(expandAndSimplify)
inormal = (rxv/(rxv.Magnitude())).applyfunc(expandAndSimplify)
jh.showEquation("i_n", inormal)
# source 1: https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/Source%20Docs/EquinoctalElements-modified.pdf