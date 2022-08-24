#%%

import sys
from tkinter import E
sys.path.append(r'C:\src\PythonEquations2Orbits') # and this line is needed for running like a normal python script


from IPython.display import display
from scipy.integrate import solve_ivp
#import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
import math
from scipy.optimize import fsolve
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
from pyeq2orb.ScaledSymbolicProblem import ScaledSymbolicProblem
from pyeq2orb.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem
from pyeq2orb.Numerical import ScipyCallbackCreators
from pyeq2orb.Numerical.LambdifyModule import LambdifyHelper
from pyeq2orb.Utilities.SolutionDictionaryFunctions import GetValueFromStateDictionaryAtIndex
from pyeq2orb.Coordinates.EquinoctialElements import EquinoctialElements,CreateSymbolicElements
import JupyterHelper as jh
import EquinicotialDemo as ed
# constants
g = 9.80665
mu = 3.986004418e14  
# thrust = 20.0
# isp = 6000.0
# m0 = 1500.0


jh.showEquation("B", B)
alp = sy.Matrix([[sy.Symbol(r'alpha_x', real=True)],[sy.Symbol(r'alpha_y', real=True)],[sy.Symbol(r'alpha_z', real=True)]])
thrust = sy.Symbol('T')
m = sy.Symbol('m')
throttle = sy.Symbol('\delta')
overallThrust = thrust*B*alp*throttle/m

eoms = f + overallThrust
jh.showEquation("eom", eoms)

symbols = SymbolicProblem()
symbols.BoundaryConditions
