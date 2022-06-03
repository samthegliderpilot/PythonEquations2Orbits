#%%
# DECLARE all the things!
import __init__
import sys
sys.path.append("..") # treating this as a jupyter-like cell requires adding one directory up
sys.path.append("../PythonOptimizationWithNlp") # and this line is needed for running like a normal python script
# these two appends do not conflict with eachother
import math
import sympy as sy
from scipy.integrate import solve_ivp, solve_bvp, odeint
from sympy.utilities.lambdify import lambdify
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# to get pyomo to work on windows, must also install this library:
# conda install -c conda-forge pynumero_libraries

from PythonOptimizationWithNlp.SymbolicOptimizerProblem import SymbolicProblem
from PythonOptimizationWithNlp.ScaledSymbolicProblem import ScaledSymbolicProblem
from PythonOptimizationWithNlp.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem
from PythonOptimizationWithNlp.Numerical import ScipyCallbackCreators
import JupyterHelper as jh

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
nus = [sy.Symbol('B_{u_f}'), sy.Symbol('B_{v_f}')]
#nus = []

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
# make the time array
tArray = np.linspace(0.0, tfOrg, 1200)
if scaleTime:
    tfVal = 1.0
    tArray = np.linspace(0.0, 1.0, 1200)
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

    r0= r0/r0
    u0=u0/v0
    v0=v0/v0
    lon0=lon0/1.0
    # add the scaled initial values (at tau_0).  We should NOT need to add these at t_0
    initialScaledStateValues = problem.CreateVariablesAtTime0(problem.StateVariables)
    constantsSubsDict.update(zip(initialScaledStateValues, [r0, u0, v0, lon0])) 
    
    

from scipy.integrate import quadrature, LSODA, ode

import pyomo.environ as poenv
allOdes = []
allOdes.extend(problem.EquationsOfMotion.values())
allOdesEvaluable = [ ]
zState = [*problem.IntegrationSymbols, problem.ControlVariables[0], problem.TimeSymbol, baseProblem.TimeFinalSymbol]
for odet in allOdes:
    subsed = odet.subs(problem.SubstitutionDictionary)
    allOdesEvaluable.append(lambdify(zState, subsed, modules={'sqrt': poenv.sqrt, 'sin': poenv.sin, 'cos':poenv.cos}))

#allOdesEvaluable = ScipyCallbackCreators.CreateSimpleCallbackForSolveIvp(problem.TimeSymbol, problem.IntegrationSymbols,  problem.EquationsOfMotion, constantsSubsDict, problem.ControlVariables)

n=120
tSpace = np.linspace(0.0, 1.0, n)

# def odeCallback(tau, y0, tf) :
#     ans = []
#     for eq in allOdesEvaluable :
#         ans.append(eq(*y0, tau, tf))
#     ans.append(0.0)
#     return ans

import pyomo.dae as podae

model = poenv.ConcreteModel()
model.t = podae.ContinuousSet(initialize=tSpace, domain=poenv.NonNegativeReals)

velBound = 1.5*abs(v0)
model.r = poenv.Var(model.t, bounds=[0.9, 10.0], initialize=float(r0))
model.u = poenv.Var(model.t, bounds=[-1*velBound, velBound], initialize=float(u0))
model.v = poenv.Var(model.t,  bounds=[-1*velBound, velBound], initialize=float(v0))
model.theta = poenv.Var(model.t, bounds=[lon0, 29.0*2.0*math.pi], initialize=float(lon0))

model.control = poenv.Var(model.t, bounds=[-1*math.pi/2.0, math.pi/2.0])

model.rDot = podae.DerivativeVar(model.r, wrt=model.t)
model.uDot = podae.DerivativeVar(model.u, wrt=model.t)
model.vDot = podae.DerivativeVar(model.v, wrt=model.t)
model.thetaDot = podae.DerivativeVar(model.theta, wrt=model.t)
model.Tf = poenv.Var(bounds=[tfOrg-100, tfOrg+100], initialize=float(tfOrg))
#model.Thetaf = poenv.Var(bounds=[0.0, 2*math.pi], initialize=float((2.0/3.0)*math.pi))

def mapping(m, t, expre) :
    return expre(m.r[t], m.u[t], m.v[t], m.theta[t], m.control[t], t, m.Tf)

model.ode1 = poenv.Constraint(model.t, rule = lambda m, t: model.rDot[t] == mapping(m, t, allOdesEvaluable[0]))
model.ode2 = poenv.Constraint(model.t, rule = lambda m, t: model.uDot[t] == mapping(m, t, allOdesEvaluable[1]))
model.ode3 = poenv.Constraint(model.t, rule = lambda m, t: model.vDot[t] == mapping(m, t, allOdesEvaluable[2]))
model.ode4 = poenv.Constraint(model.t, rule = lambda m, t: model.thetaDot[t] == mapping(m, t, allOdesEvaluable[3]))

model.radiusObjective = poenv.Objective(expr = lambda mod : mod.r[1.0], sense=poenv.maximize) # max radius 
#model.timeObjective = poenv.Objective(expr = lambda mod : mod.Tf, sense=poenv.minimize) # minimize Tf

#model.control[0].fix(float(0.05))

model.r[0].fix(float(r0))
model.u[0].fix(float(u0))
model.v[0].fix(float(v0))
model.theta[0].fix(0.0)

def vIsCircularConstraint(mod) :
    return mod.v[1.0]/8000.0 == poenv.sqrt(float(constantsSubsDict[baseProblem.Mu])/(mod.r[1.0]*8000.0))

def uIsZero(mod) :
    return mod.u[1.0] <= 0.00001

def uIsZero2(mod) :
    return mod.u[1.0] >= -0.00001

model.vConst = poenv.Constraint(rule=vIsCircularConstraint)
model.uConst = poenv.Constraint(rule = lambda m : m.u[1.0]==0.0)
#model.thetaConst = poenv.Constraint(rule = lambda m : m.theta[1.0] == 26*2*math.pi + 1.15*math.pi)
#model.uConst1 = poenv.Constraint(rule = uIsZero)
#model.uConst2= poenv.Constraint(rule = uIsZero2)
#model.rConst = poenv.Constraint(rule = lambda m : m.r[1.0]==float(xfBcVals[rs]))

#model.controlConst = poenv.Constraint(rule = lambda m : m.control[0.0]==0.05)

model.var_input = poenv.Suffix(direction=poenv.Suffix.LOCAL)
sim = podae.Simulator(model, package='scipy') 

model.var_input[model.control] = {0: 0.05}
model.var_input[model.Tf] = {0: tfOrg}
tsim, profiles = sim.simulate(numpoints=n, varying_inputs=model.var_input, integrator='dop853', initcon=np.array([r0,u0, v0, lon0], dtype=float))
debugMessage = True
#plotOdeIntSolution(tsim*tfOrg, profiles[:,0], profiles[:,1], profiles[:,2], profiles[:,3], numScaleVector, 0.05)

#poenv.TransformationFactory('dae.finite_difference').apply_to(model, wrt=model.t, nfe=n, scheme='BACKWARD')
poenv.TransformationFactory('dae.collocation').apply_to(model, wrt=model.t, nfe=n,ncp=3, scheme='LAGRANGE-RADAU')
#['LAGRANGE-RADAU', 'LAGRANGE-LEGENDRE']
sim.initialize_model()
#model.display()
solver = poenv.SolverFactory('cyipopt')
#solver.options['tol'] = 1E-12
#solver.options['nlp_scaling_method'] = 'none'
#solver.options['print_level'] = 6
#solver.options['OMP_NUM_THREADS'] = 10
#solver.options['max_iter'] = 2000

#solver.options['halt_on_ampl_error'] = 'yes'
solver.solve(model, tee=True)
#%%
def plotPyomoSolution(model, stateSymbols):
    tSpace =np.array( [t for t in model.t]) * model.Tf.value
    rSym = np.array([model.r[t]() for t in model.t])
    uSym = np.array([model.u[t]() for t in model.t])
    vSym = np.array([model.v[t]() for t in model.t])
    lonSim = np.array([model.theta[t]() for t in model.t])
    controls = np.array([model.control[t]() for t in model.t])
    print("control 0 = " + str(controls[0]))
    ansAsDict = OrderedDict()
    stateSymbols[0]= rSym
    stateSymbols[1]= uSym
    stateSymbols[2]= vSym
    stateSymbols[3]=  lonSim

    return [tSpace, ansAsDict]

[tArray, solutionDictionary] = plotPyomoSolution(model, problem.StateVariables)
unscaledResults = problem.DescaleResults(solutionDictionary)
baseProblem.PlotSolution(tArray, unscaledResults, "Test")


#     plotOdeIntSolution(tSpace, rSym, uSym, vSym, lonSim, scaleVector, controls)
# plotPyomoSolution(model, numScaleVector)
print("Tf = " + str(model.Tf.value/86400))

