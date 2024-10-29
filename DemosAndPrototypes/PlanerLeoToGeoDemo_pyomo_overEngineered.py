#%%
# type: ignore
# these two appends do not conflict with each other
import math
import sympy as sy
from scipy.integrate import solve_ivp, solve_bvp, odeint
from sympy.utilities.lambdify import lambdify
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
nus = [sy.Symbol('B_{u_f}'), sy.Symbol('B_{v_f}')]
#nus = []

baseProblem = ContinuousThrustCircularOrbitTransferProblem()
initialStateValues = baseProblem.StateSymbolsInitial()
problem = baseProblem

if scale :
    newSvs = ScaledSymbolicProblem.CreateBarVariables(problem.StateSymbols, problem.TimeSymbol) 
    problem = ScaledSymbolicProblem(baseProblem, newSvs, {problem.StateSymbols[0]: initialStateValues[0], 
                                                          problem.StateSymbols[1]: initialStateValues[2], 
                                                          problem.StateSymbols[2]: initialStateValues[2], 
                                                          problem.StateSymbols[3]: 1.0} , scaleTime)
rs = problem.StateSymbols[0]
us = problem.StateSymbols[1]
vs = problem.StateSymbols[2]
longitudes = problem.StateSymbols[3]
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
    initialValuesAtTau0 = SafeSubs(initialStateValues, {baseProblem.TimeInitialSymbol: problem.TimeInitialSymbol})
    constantsSubsDict.update(zip(initialValuesAtTau0, [r0, u0, v0, lon0]))

    r0= r0/r0
    u0=u0/v0
    v0=v0/v0
    lon0=lon0/1.0
    # add the scaled initial values (at tau_0).  We should NOT need to add these at t_0
    initialScaledStateValues = problem.StateSymbolsInitial()
    constantsSubsDict.update(zip(initialScaledStateValues, [r0, u0, v0, lon0])) 
    
lambdifyFunctionMap = {'sqrt': poenv.sqrt, 'sin': poenv.sin, 'cos':poenv.cos} #TODO: MORE!!!!


asNumericalProblem = NumericalProblemFromSymbolicProblem(problem, lambdifyFunctionMap)

n=300
tSpace = np.linspace(0.0, 1.0, n)

model = poenv.ConcreteModel()
model.t = podae.ContinuousSet(initialize=tSpace, domain=poenv.NonNegativeReals)

def setEverythingOnPyomoModel(mdl, name, t, bounds, iv) :
    if t == None and iv != None :
        setattr(mdl, name, poenv.Var(bounds=bounds, initialize=float(iv)))
    elif iv is None :
        setattr(mdl, name, poenv.Var(t, bounds=bounds))
    else :
        setattr(mdl, name, poenv.Var(t, bounds=bounds, initialize=float(iv)))
        getattr(mdl, name)[0].fix(float(iv))        
    if t != None :
        setattr(mdl, name+"Dot", podae.DerivativeVar(getattr(mdl, name), wrt=t))

def setEom(mdl, name, t, eom) :
    setattr(mdl, name+"Eom", poenv.Constraint(t, rule =lambda m, t2: getattr(mdl, name+"Dot")[t2] == mapping(m, t2, eom)))

velBound = 1.5*abs(v0)

setEverythingOnPyomoModel(model, "r",     model.t, [0.9, 8.0],              float(r0))
setEverythingOnPyomoModel(model, "u",     model.t, [-1.0*velBound, velBound],  float(u0))
setEverythingOnPyomoModel(model, "v",     model.t, [-1.0*velBound, velBound],  float(v0))
setEverythingOnPyomoModel(model, "theta", model.t, [lon0, 29.0*2.0*math.pi], float(lon0))
setEverythingOnPyomoModel(model, "control", model.t, [-1.0*math.pi/2.0, math.pi/2.0], None)
setEverythingOnPyomoModel(model, "Tf", None, [tfOrg-2, tfOrg+2], tfOrg)

def mapping(m, t, expression) :
    return expression([t, m.r[t], m.u[t], m.v[t], m.theta[t], m.control[t], m.Tf])

setEom(model, "r",     model.t, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 0))
setEom(model, "u",     model.t, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 1))
setEom(model, "v",     model.t, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 2))
setEom(model, "theta", model.t, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 3))

def createTerminalCost(mdl, theProblem) :
    cb = theProblem._terminalCost    
    innerLmd = lambda mod1 : mapping(mod1, 1.0, cb)
    setattr(mdl, "objective", poenv.Objective(expr = innerLmd, sense=poenv.maximize))
createTerminalCost(model, asNumericalProblem)
#model.radiusObjective = poenv.Objective(expr = lambda mod : mod.r[1.0], sense=poenv.maximize) # max radius 


i = 1
for bc in asNumericalProblem.BoundaryConditionCallbacks :
    def makeInnerLmd(bc) :
        return lambda mod1 : 0 == mapping(mod1, 1.0, bc)
    innerLmd = makeInnerLmd(bc)
    setattr(model, "bc" + str(i), poenv.Constraint(rule = innerLmd))
    i=i+1

model.var_input = poenv.Suffix(direction=poenv.Suffix.LOCAL)
sim = podae.Simulator(model, package='scipy') 
#model.var_input[model.control] = {0: 0.05}
model.var_input[model.control] = {0: 0.00}
model.var_input[model.Tf] = {0: tfOrg}
tSim, profiles = sim.simulate(numpoints=n, varying_inputs=model.var_input, integrator='dop853', initcon=np.array([r0,u0, v0, lon0], dtype=float))
debugMessage = True
#plotOdeIntSolution(tSim*tfOrg, profiles[:,0], profiles[:,1], profiles[:,2], profiles[:,3], numScaleVector, 0.05)

#poenv.TransformationFactory('dae.finite_difference').apply_to(model, wrt=model.t, nfe=n, scheme='BACKWARD')
poenv.TransformationFactory('dae.collocation').apply_to(model, wrt=model.t, nfe=n,ncp=3, scheme='LAGRANGE-RADAU')
#['LAGRANGE-RADAU', 'LAGRANGE-LEGENDRE']
sim.initialize_model()
solver = poenv.SolverFactory('cyipopt')
solver.solve(model, tee=True)

def plotPyomoSolution(model, stateSymbols):
    tSpace =np.array( [t for t in model.t]) * model.Tf.value
    rSym = np.array([model.r[t]() for t in model.t])
    uSym = np.array([model.u[t]() for t in model.t])
    vSym = np.array([model.v[t]() for t in model.t])
    lonSim = np.array([model.theta[t]() for t in model.t])
    controls = np.array([model.control[t]() for t in model.t])
    print("control 0 = " + str(controls[0]))
    plt.title("Thrust Angle")
    plt.plot(tSpace/86400, controls*180.0/math.pi)
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

[tArray, solutionDictionary] = plotPyomoSolution(model, problem.StateSymbols)
unscaledResults = problem.DescaleResults(solutionDictionary, baseProblem.StateSymbols)
baseProblem.PlotSolution(tArray, unscaledResults, "Test")

print("Tf = " + str(model.Tf.value/86400))
jh.showEquation("r_f", unscaledResults[baseProblem.StateSymbols[0]][-1]) 
jh.showEquation("u_f", unscaledResults[baseProblem.StateSymbols[1]][-1]) 
jh.showEquation("v_f", unscaledResults[baseProblem.StateSymbols[2]][-1])     

xyz = np.zeros((len(tArray), 3))
for i in range(0, len(unscaledResults[baseProblem.StateSymbols[0]])) :
    r = unscaledResults[baseProblem.StateSymbols[0]][i]
    theta = unscaledResults[baseProblem.StateSymbols[3]][i]
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
