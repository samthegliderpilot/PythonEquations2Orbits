#%%
# type: ignore
# DECLARE all the things!
import __init__
import sys
import os
thisFile = os.path.abspath(__file__)
sys.path.append(os.path.abspath(thisFile + '..\\..\\..\\'))
# these two appends do not conflict with eachother
import math
import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# to get pyomo to work on windows, must also install this library:
# conda install -c conda-forge pynumero_libraries

from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
from pyeq2orb.ScaledSymbolicProblem import ScaledSymbolicProblem
from pyeq2orb.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem
from pyeq2orb.Numerical import ScipyCallbackCreators
import scipyPaperPrinter as jh
import pyomo.environ as poenv
import pyomo.dae as podae
from matplotlib.figure import Figure
from pyeq2orb.NumericalProblemFromSymbolic import NumericalProblemFromSymbolicProblem

# constants
g = 9.80665
mu = 3.986004418e14  
thrust = 20.0
isp = 6000.0
m0 = 1500.0

# initial values
r0 = 6678000.0
u0 = 0.0
v0 = sy.sqrt(mu/r0) # circular
lon0 = 0.0
# I know from many previous runs that this is the time needed to go from LEO to GEO.
# However, below works well wrapped in another fsolve to control the final time for a desired radius.
tfVal  = 3600*3.97152*24
tfOrg = tfVal

# these are options to switch to try different things
scale = True
scaleTime = scale and True
# your choice of the nu vector here controls which transversality condition we use



baseProblem = ContinuousThrustCircularOrbitTransferProblem()
initialStateValues = baseProblem.CreateVariablesAtTime0(baseProblem.StateVariables)
problem = baseProblem

if scale :
    newSvs = ScaledSymbolicProblem.CreateBarVariables(problem.StateVariables, problem.TimeSymbol) 
    problem = ScaledSymbolicProblem(baseProblem, newSvs, {problem.StateVariables[0]: initialStateValues[0], 
                                                          problem.StateVariables[1]: initialStateValues[2], 
                                                          problem.StateVariables[2]: initialStateValues[2], 
                                                          problem.StateVariables[3]: 1.0} , scaleTime)
rs = problem.StateVariables[0]
us = problem.StateVariables[1]
vs = problem.StateVariables[2]
lons = problem.StateVariables[3]

jh.t = problem._timeSymbol

# register constants
constantsSubsDict = problem.SubstitutionDictionary
constantsSubsDict[baseProblem.Isp] = isp
constantsSubsDict[baseProblem.MassInitial] = m0
constantsSubsDict[baseProblem.Gravity] = g
constantsSubsDict[baseProblem.Mu]= mu
constantsSubsDict[baseProblem.Thrust] = thrust

# register initial state values
constantsSubsDict.update(zip(initialStateValues, [r0, u0, v0, lon0]))
if scale :
    # and reset the real initial values using tau_0 instead of time
    initialValuesAtTau0 = SymbolicProblem.SafeSubs(initialStateValues, {baseProblem.TimeInitialSymbol: problem.TimeInitialSymbol})
    constantsSubsDict.update(zip(initialValuesAtTau0, [r0, u0, v0, lon0]))

    r0 = r0/r0
    u0 = u0/v0
    v0 = v0/v0
    lon0 = lon0/1.0
    # add the scaled initial values (at tau_0).  We should NOT need to add these at t_0
    initialScaledStateValues = problem.CreateVariablesAtTime0(problem.StateVariables)
    constantsSubsDict.update(zip(initialScaledStateValues, [r0, u0, v0, lon0])) 
    
lambdiafyFunctionMap = {'sqrt': poenv.sqrt, 'sin': poenv.sin, 'cos':poenv.cos} #TODO: MOOOORE!!!!

asNumericalProblem = NumericalProblemFromSymbolicProblem(problem, lambdiafyFunctionMap)

n=200
tSpace = np.linspace(0.0, 1.0, n)

model = poenv.ConcreteModel()
model.t = podae.ContinuousSet(initialize=tSpace, domain=poenv.NonNegativeReals)

velBound = float(1.5*abs(v0))
model.r = poenv.Var(model.t, bounds=(0.9, 8.0), initialize=float(r0))
model.u = poenv.Var(model.t, bounds=(-1.0*velBound, velBound), initialize=float(u0))
model.v = poenv.Var(model.t, bounds=(-1.0*velBound, velBound), initialize=float(v0))
model.lon = poenv.Var(model.t, bounds=(lon0, 29.0*2.0*math.pi), initialize=float(lon0))
model.control = poenv.Var(model.t, bounds=(-1.0*math.pi/2.2, math.pi/2.2))
model.tf = poenv.Var(bounds=(tfOrg-2, tfOrg+2), initialize=float(tfOrg))

model.r[0].fix(float(r0))
model.u[0].fix(float(u0))
model.v[0].fix(float(v0))
model.lon[0].fix(float(lon0))

model.rDot = podae.DerivativeVar(model.r, wrt=model.t)
model.uDot = podae.DerivativeVar(model.u, wrt=model.t)
model.vDot = podae.DerivativeVar(model.v, wrt=model.t)
model.lonDot = podae.DerivativeVar(model.lon, wrt=model.t)

def mapPyomoStateToProblemState(m, t, expre) :
    return expre([t, m.r[t], m.u[t], m.v[t], m.lon[t], m.control[t], m.tf])

model.rEom = poenv.Constraint(model.t, rule =lambda m, t2: m.rDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 0)))
model.uEom = poenv.Constraint(model.t, rule =lambda m, t2: m.uDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 1)))
model.vEom = poenv.Constraint(model.t, rule =lambda m, t2: m.vDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 2)))
model.lonEom = poenv.Constraint(model.t, rule =lambda m, t2: m.lonDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 3)))


def mapPyomoStateToProblemStatebc(m, t, expre) :
    return expre([t, m.r[t], m.u[t], m.v[t], m.lon[t], m.control[t], m.tf])

model.bc1 = poenv.Constraint(rule = lambda mod1 : 0 == mapPyomoStateToProblemStatebc(mod1, 1.0, asNumericalProblem.BoundaryConditionCallbacks[0]))
model.bc2 = poenv.Constraint(rule = lambda mod1 : 0 == mapPyomoStateToProblemState(mod1, 1.0, asNumericalProblem.BoundaryConditionCallbacks[1]))

def singlePyomoArrayToTerminalCostCallback(m, t, expr) :
    return expr(m.tf, [t, m.r[t], m.u[t], m.v[t], m.lon[t], m.control[t]])
finalRadiusCallback = lambda m : singlePyomoArrayToTerminalCostCallback(m, 1.0, asNumericalProblem.TerminalCost)
model.radiusObjective = poenv.Objective(expr = finalRadiusCallback, sense=poenv.maximize)

model.var_input = poenv.Suffix(direction=poenv.Suffix.LOCAL)
model.var_input[model.control] = {0: 0.03}
model.var_input[model.tf] = {0: tfOrg}

sim = podae.Simulator(model, package='scipy') 
tsim, profiles = sim.simulate(numpoints=n, varying_inputs=model.var_input, integrator='dop853', initcon=np.array([r0,u0, v0, lon0], dtype=float))

#poenv.TransformationFactory('dae.finite_difference').apply_to(model, wrt=model.t, nfe=n, scheme='BACKWARD')
poenv.TransformationFactory('dae.collocation').apply_to(model, wrt=model.t, nfe=n,ncp=3, scheme='LAGRANGE-RADAU')
#['LAGRANGE-RADAU', 'LAGRANGE-LEGENDRE']
sim.initialize_model()
solver = poenv.SolverFactory('cyipopt')
solver.solve(model, tee=True)

def plotPyomoSolution(model, stateSymbols):
    tSpace =np.array( [t for t in model.t]) * model.tf.value
    rSym = np.array([model.r[t]() for t in model.t])
    uSym = np.array([model.u[t]() for t in model.t])
    vSym = np.array([model.v[t]() for t in model.t])
    lonSim = np.array([model.lon[t]() for t in model.t])
    controls = np.array([model.control[t]() for t in model.t])
    print("control 0 = " + str(controls[0]))
    plt.title("Thrust Angle")
    plt.plot(tSpace/86400, controls*180.0/math.pi, label="Thrust Angle (deg)")
    plt.tight_layout()
    plt.grid(alpha=0.5)
    plt.legend(framealpha=1, shadow=True)
    plt.show()    
    ansAsDict = OrderedDict()
    ansAsDict[stateSymbols[0]]= rSym
    ansAsDict[stateSymbols[1]]= uSym
    ansAsDict[stateSymbols[2]]= vSym
    ansAsDict[stateSymbols[3]]=  lonSim

    return [tSpace, ansAsDict]

[tArray, solutionDictionary] = plotPyomoSolution(model, problem.StateVariables)
unscaledResults = problem.DescaleResults(solutionDictionary)
baseProblem.PlotSolution(tArray, unscaledResults, "Leo to Geo")

print("Tf = " + str(model.tf.value/86400))
jh.showEquation("r_f", unscaledResults[baseProblem.StateVariables[0]][-1]) 
jh.showEquation("u_f", unscaledResults[baseProblem.StateVariables[1]][-1]) 
jh.showEquation("v_f", unscaledResults[baseProblem.StateVariables[2]][-1])     

xyz = np.zeros((len(tArray), 3))
for i in range(0, len(unscaledResults[baseProblem.StateVariables[0]])) :
    r = unscaledResults[baseProblem.StateVariables[0]][i]
    theta = unscaledResults[baseProblem.StateVariables[3]][i]
    x = r*math.cos(theta)
    y = r*math.sin(theta)
    xyz[i,0] = x
    xyz[i,1] = y
    xyz[i,2] = 0

import plotly.express as px
from pandas import DataFrame
df = DataFrame(xyz)

x = np.array(xyz[:,0])
y = np.array(xyz[:,1])
z = np.array(xyz[:,2])
df = DataFrame({"x": x, "y":y, "z":z})
fig = px.line_3d(df, x="x", y="y", z="z")
fig.show()
