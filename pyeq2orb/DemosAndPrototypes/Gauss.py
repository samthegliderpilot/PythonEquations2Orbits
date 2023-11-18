#%%
import __init__ # type: ignore
import sympy as sy
import math
import os
import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0]))) # need to import 2 directories up (so pyeq2orb is a subfolder)
sy.init_printing()

from pyeq2orb.ForceModels.TwoBodyForce import CreateTwoBodyMotionMatrix, CreateTwoBodyListForModifiedEquinoctialElements
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Coordinates.KeplerianModule import KeplerianElements
import pyeq2orb.Coordinates.KeplerianModule as KepModule
import scipyPaperPrinter as jh #type: ignore
from scipy.optimize import fsolve, root #type: ignore
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
t = sy.Symbol('t')
t0 = sy.Symbol('t_0')
tf = sy.Symbol('t_f')

kepElements = KepModule.CreateSymbolicElements()
kepElements.TrueAnomaly = sy.Symbol(r'\nu')
kot =KepModule.CreateSymbolicElements(t)
kot.TrueAnomaly = sy.Function('ta')(t)
accel = Cartesian(0, 0, sy.Symbol('F_s'))
keplerianEquationsOfMotion =  KepModule.GaussianEquationsOfMotionVallado(kepElements, accel)

display(keplerianEquationsOfMotion.EccentricityDot)
display(keplerianEquationsOfMotion.ArgumentOfPeriapsisDot)

subsDict = {kepElements.Eccentricity: 0.35, kepElements.SemiMajorAxis:42162000, kepElements.GravitationalParameter:3.986004418e14, accel[2]:1}
eccCb = sy.lambdify(kepElements.TrueAnomaly, keplerianEquationsOfMotion.EccentricityDot.subs(subsDict))
aopCb = sy.lambdify(kepElements.TrueAnomaly, keplerianEquationsOfMotion.ArgumentOfPeriapsisDot.subs(subsDict))
taRange = np.arange(0.0, 2*math.pi, 0.01)
eccChange = eccCb(taRange)
aopChange = aopCb(taRange)
fig, ax = plt.subplots()
ax.plot(taRange, eccChange)
ax.plot(taRange, aopChange)

ax.set(xlabel='true anomalay (rad)', ylabel='ecc change',
       title='EccPlot')
ax.grid()

plt.show()

#%%
ta1 = sy.acos(-0.35)+ 0.35/2+0.001/2
display(ta1*180/math.pi)
ta2 = sy.acos(-0.35001)- 0.35001/2+0.001/2
display(ta2*180/math.pi)