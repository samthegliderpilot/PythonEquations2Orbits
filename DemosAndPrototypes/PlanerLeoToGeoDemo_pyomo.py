#%%
# DECLARE all the things!
import __init__
import sys
sys.path.append("..") # treating this as a jupyter-like cell requires adding one directory up
sys.path.append("../pyeq2orb") # and this line is needed for running like a normal python script
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

from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
from pyeq2orb.ScaledSymbolicProblem import ScaledSymbolicProblem
from pyeq2orb.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem
from pyeq2orb.Numerical import ScipyCallbackCreators
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
    

import pyomo.environ as poenv

import pyomo.dae as podae
from typing import List, Dict
from pyeq2orb.NumericalOptimizerProblem import NumericalOptimizerProblemBase
from matplotlib.figure import Figure

lambdiafyFunctionMap = {'sqrt': poenv.sqrt, 'sin': poenv.sin, 'cos':poenv.cos} #TODO: MORE!!!!

class NumericalProblemFromSymbolicProblem(NumericalOptimizerProblemBase) :
    def __init__(self, wrappedProblem : SymbolicProblem, functionMap : Dict) :
        super().__init__(wrappedProblem.TimeSymbol)
        self._wrappedProblem = wrappedProblem
        self.State.extend(wrappedProblem.StateVariables)
        self.Control.extend(wrappedProblem.ControlVariables)

        entireState = [wrappedProblem.TimeSymbol, *wrappedProblem.StateVariables, *wrappedProblem.ControlVariables]
        

        if isinstance(wrappedProblem, ScaledSymbolicProblem) and wrappedProblem.ScaleTime :
            entireState.append(wrappedProblem.TimeFinalSymbolOriginal)
        
        finalState = SymbolicProblem.SafeSubs(entireState, {wrappedProblem.TimeSymbol: wrappedProblem.TimeFinalSymbol})
        self._terminalCost = lambdify([finalState], wrappedProblem.TerminalCost.subs(wrappedProblem.SubstitutionDictionary).simplify(), functionMap)
        self._unIntegratedPathCost = 0.0
        if wrappedProblem.UnIntegratedPathCost != None and wrappedProblem.UnIntegratedPathCost != 0.0 :
            self._unIntegratedPathCost = lambdify(entireState, wrappedProblem.UnIntegratedPathCost.subs(wrappedProblem.SubstitutionDictionary).simplify(), functionMap)
        self._equationOfMotionList = []
        for (sv, eom) in wrappedProblem.EquationsOfMotion.items() :
            eomCb = lambdify(entireState, eom.subs(wrappedProblem.SubstitutionDictionary).simplify(), functionMap)
            self._equationOfMotionList.append(eomCb) 

        for bc in wrappedProblem.BoundaryConditions :
            bcCallback = lambdify([finalState], bc.subs(wrappedProblem.SubstitutionDictionary).simplify(), functionMap)
            self.BoundaryConditionCallbacks.append(bcCallback)            

    #initial guess callback
    #initial conditions
    #final conditions

    @property
    def ContolValueAtTCallbackForInitialGuess(self):
        return self._controlCallback

    @ContolValueAtTCallbackForInitialGuess.setter
    def setContolValueAtTCallbackForInitialGuess(self, callback) :
        self._controlCallback = callback

    def InitialGuessCallback(self, t : float) -> List[float] :
        """A function to produce an initial state at t, 

        Args:
            t (float): _description_

        Returns:
            List[float]: A list the values in the state followed by the values of the controls at t.
        """
        pass

    def EquationOfMotion(self, t : float, stateAndControlAtT : List[float]) -> List[float] :
        """The equations of motion.  

        Args:
            t (float): The time.  
            stateAndControlAtT (List[float]): The current state and control at t to evaluate the equations of motion.

        Returns:
            List[float]: The derivative of the state variables 
        """  
        ans = []
        for i in range(0, len(stateAndControlAtT)) :
            ans.append(self.SingleEquationOfMotion(t, *stateAndControlAtT, i))
        return ans

    def SingleEquationOfMotion(self, t : float, stateAndControlAtT : List[float], indexOfEom : int) -> float :
        return self._equationOfMotionList[indexOfEom](t, stateAndControlAtT)

    def SingleEquationOfMotionWithTInState(self, state, indexOfEom) :
        return self._equationOfMotionList[indexOfEom](state[0], *state[1:])

    def UnIntegratedPathCost(self, t, stateAndControl) :
        return self._unIntegratedPathCost(t, stateAndControl)
    
    def TerminalCost(self, tf, finalStateAndControl) :
        return self._terminalCost(tf, finalStateAndControl)

    def AddResultsToFigure(self, figure : Figure, t : List[float], dictionaryOfValueArraysKeyedOffState : Dict[object, List[float]], label : str) -> None:
        """Adds the contents of dictionaryOfValueArraysKeyedOffState to the plot.

        Args:
            figure (matplotlib.figure.Figure): The figure the data is getting added to.
            t (List[float]): The time corresponding to the data in dictionaryOfValueArraysKeyedOffState.
            dictionaryOfValueArraysKeyedOffState (Dict[object, List[float]]): The data to get added.  The keys must match the values in self.State and self.Control.
            label (str): A label for the data to use in the plot legend.
        """
        self._wrappedProblem.AddStandardResultsToFigure(figure, t, dictionaryOfValueArraysKeyedOffState, label)


asNumericalProblem = NumericalProblemFromSymbolicProblem(problem, lambdiafyFunctionMap)

n=200
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

setEverythingOnPyomoModel(model, "r",     model.t, [0.9, 8.0],              r0)
setEverythingOnPyomoModel(model, "u",     model.t, [-1*velBound, velBound],  u0)
setEverythingOnPyomoModel(model, "v",     model.t, [-1*velBound, velBound],  v0)
setEverythingOnPyomoModel(model, "theta", model.t, [lon0, 29.0*2.0*math.pi], lon0)
setEverythingOnPyomoModel(model, "control", model.t, [-1*math.pi/2.0, math.pi/2.0], None)
setEverythingOnPyomoModel(model, "Tf", None, [tfOrg-2, tfOrg+2], tfOrg)

def mapping(m, t, expre) :
    return expre([t, m.r[t], m.u[t], m.v[t], m.theta[t], m.control[t], m.Tf])

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
tsim, profiles = sim.simulate(numpoints=n, varying_inputs=model.var_input, integrator='dop853', initcon=np.array([r0,u0, v0, lon0], dtype=float))
debugMessage = True
#plotOdeIntSolution(tsim*tfOrg, profiles[:,0], profiles[:,1], profiles[:,2], profiles[:,3], numScaleVector, 0.05)

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

[tArray, solutionDictionary] = plotPyomoSolution(model, problem.StateVariables)
unscaledResults = problem.DescaleResults(solutionDictionary)
baseProblem.PlotSolution(tArray, unscaledResults, "Test")

print("Tf = " + str(model.Tf.value/86400))
jh.showEquation("r_f", unscaledResults[baseProblem.StateVariables[0]][-1]) 
jh.showEquation("u_f", unscaledResults[baseProblem.StateVariables[1]][-1]) 
jh.showEquation("v_f", unscaledResults[baseProblem.StateVariables[2]][-1])     
