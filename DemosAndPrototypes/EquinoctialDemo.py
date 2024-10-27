#%%
import sympy as sy
from pyeq2orb.ForceModels.TwoBodyForce import CreateTwoBodyMotionMatrix, CreateTwoBodyListForModifiedEquinoctialElements
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule import ModifiedEquinoctialElements, CreateSymbolicElements
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from pyeq2orb import SafeSubs
from pyeq2orb.Numerical.LambdifyHelpers import OdeLambdifyHelper
from pyeq2orb.HighLevelHelpers.EquinoctialElementsHelpers import ModifiedEquinoctialElementsHelpers
import scipyPaperPrinter as jh#type: ignore
import numpy as np
import math as math
from scipy.integrate import solve_ivp
from typing import Union, Dict, List, Callable, Any, Optional, Tuple
import pyeq2orb.Graphics.Primitives as prim
from pyeq2orb.Graphics.Plotly2DModule import plot2DLines
from pyeq2orb.Graphics.PlotlyUtilities import PlotAndAnimatePlanetsWithPlotly
from pyeq2orb.Numerical.ScalingHelpers import scaledEquationOfMotionHolder
from IPython.display import display
from collections import OrderedDict
import matplotlib.pyplot as plt
from pandas import DataFrame #type: ignore
import plotly.graph_objects as go
import plotly.express as px#type: ignore
import pyeq2orb.Coordinates.OrbitFunctions as orb
import pyomo.environ as poenv#type: ignore
import pyomo.dae as podae#type: ignore

subsDict : Dict[Union[sy.Symbol, sy.Expr], SymbolOrNumber]= {}

t = sy.Symbol('t', real=True)
t0 = sy.Symbol('t_0', real=True)
tf = sy.Symbol('t_f', real=True)
mu = sy.Symbol(r'\mu', real=True, positive=True)

symbolicElements = CreateSymbolicElements(t, mu)
twoBodyOdeMatrix = CreateTwoBodyMotionMatrix(symbolicElements, subsDict)
twoBodyEvaluationHelper = OdeLambdifyHelper(t, symbolicElements.ToArray(), twoBodyOdeMatrix, [mu], subsDict)
twoBodyOdeCallback = twoBodyEvaluationHelper.CreateSimpleCallbackForSolveIvp()

#%%
tfVal = 793*86400.0
n = 252
tSpace = np.linspace(0.0, tfVal, n)

muVal = 1.32712440042e20
r0 = Cartesian(58252488010.7, 135673782531.3, 2845058.1)
v0 = Cartesian(-27844.5, 11659.9, 0000.3)
initialElements = ModifiedEquinoctialElements.FromMotionCartesian(MotionCartesian(r0, v0), muVal)

rf = Cartesian(36216277800.4, -211692395522.5, -5325189049.9)
vf = Cartesian(24798.8, 6168.2, -480.0)
finalElements = ModifiedEquinoctialElements.FromMotionCartesian(MotionCartesian(rf, vf), muVal)

#%%
# build up a perturbation matrix
Au = 149597870700.0
AuSy = sy.Symbol('A_u')
gVal = 9.8065 
gSy = sy.Symbol('g', real=True, positive=True) #9.8065
m0Val = 2000.0
ispVal = 3000.0
nRev = 2.0
thrustVal = 0.1997*1.2#0.25 # odd number pulled from just under Fig14

azi = sy.Function(r'\theta', real=True)(t)
elv = sy.Function(r'\phi', real=True)(t)
thrust = sy.Symbol('T', real=True, positive=True)
throttle = sy.Function('\delta', real=True, positive=True)(t)
m = sy.Function('m', real=True, positive=True)(t)
isp = sy.Symbol("I_{sp}", real=True, positive=True)

subsDict[gSy] = gVal

alp = sy.Matrix([[sy.cos(azi)*sy.cos(elv)], [sy.sin(azi)*sy.cos(elv)], [sy.sin(elv)]])
B = symbolicElements.CreatePerturbationMatrix(subsDict)
overallThrust = thrust*B*alp*(throttle)/(m) 
c = isp * gSy
mDot = -1*thrust*throttle/(isp*gSy)

pathCost = throttle* thrust/c

stateDynamics = twoBodyOdeMatrix + overallThrust
stateDynamics=stateDynamics.row_insert(6, sy.Matrix([mDot]))
stateVariables = [*symbolicElements.ToArray(), m]

simpleThrustCallbackHelper = OdeLambdifyHelper(t, stateVariables, stateDynamics, [mu, azi, elv, thrust, throttle, isp], subsDict)
simpleThrustCallback = simpleThrustCallbackHelper.CreateSimpleCallbackForSolveIvp()

satSolution = solve_ivp(simpleThrustCallback, [0.0, tfVal], [*initialElements.ToArray(), m0Val], args=[muVal, 1.5, 0.0, 1.0, thrustVal, ispVal], t_eval=np.linspace(0.0, tfVal,n), dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
satPath = ModifiedEquinoctialElementsHelpers.createSatPathFromIvpSolution(satSolution, muVal, "#00ffff")

#%%

tau = sy.Symbol(r'\tau', positive=True, real=True)
scalingFactors =  [Au, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
newSvs = scaledEquationOfMotionHolder.CreateVariablesWithBar(stateVariables, t)
scaledEquationsOfMotion = scaledEquationOfMotionHolder.ScaleStateVariablesAndTimeInFirstOrderOdes(stateVariables, stateDynamics, newSvs,scalingFactors, tau, tf, [azi, elv, throttle])
scaledSubsDict = scaledEquationsOfMotion.createCorrectedSubsDict(subsDict, stateVariables, t)

simpleThrustCallbackHelperScaled = OdeLambdifyHelper(tau, scaledEquationsOfMotion.newStateVariables, scaledEquationsOfMotion.scaledFirstOrderDynamics, [mu, *scaledEquationsOfMotion.otherSymbols, thrust, isp, tf], scaledSubsDict)

#%%
print("making pyomo model")

model = poenv.ConcreteModel()
model.t = podae.ContinuousSet(initialize=np.linspace(0.0, 1.0, n), domain=poenv.NonNegativeReals)
smaLow = 126.10e9/scalingFactors[0] # little less than earth
smaHigh = 327.0e9/scalingFactors[0] # little more than mars
lambdifyFunctionMap = {'sqrt': poenv.sqrt, 'sin': poenv.sin, 'cos':poenv.cos}
for k,v in lambdifyFunctionMap.items():
    simpleThrustCallbackHelperScaled.FunctionRedirectionDictionary[k] = v
listOfEomCallback = simpleThrustCallbackHelperScaled.CreateListOfEomCallbacks()

# some day, if I want to, using the add_component and a healthy amount of wrapping methods, we could
# fully automate the creation of a pyomo model from the symbolic problem statement.  But I would 
# rather explore more math and cases than build a well designed and tested library like that (for now).

class PyomoHelperFunctions :
    def __init__(self,model: poenv.ConcreteModel, domain):
        self.Model = model
        self.Domain = domain
        self.IndexToStateMap = {} #type: Dict[int, Any]
        self._nextStateIndex = 0
    
    def addStateElementToPyomo(self, name : str, lowerBound : float, upperBound:float, initialValue : float,  fixInitialValue = True) :
        model = self.Model
        model.add_component(name, poenv.Var(self.Domain, bounds=(lowerBound, upperBound), initialize=float(initialValue)))
        element = model.component(name)
        if fixInitialValue :
            element[0].fix(float(initialValue))
        self.IndexToStateMap[self._nextStateIndex] = lambda m, t : element[t]
        self._nextStateIndex =self._nextStateIndex+1
        return element

    # def addEquationOfMotionConstraint(self, name, callback, originalElement):
    #     model = self.Model
    #     model.add_component(name, podae.DerivativeVar(originalElement, wrt=model.t))
    #     derivativeElement = model.component(name)
    #     model.add_component(name+"Eom", poenv.Constraint(self.Domain, rule =callback))
    #     return model.component(name+"Eom")

    def addConstantSolveForParameter(self, name, lowerBound, upperBound, initialGuess):
        model = self.Model
        model.add_component(name, poenv.Var(bounds=(lowerBound, upperBound), initialize=float(initialGuess)))
        component = model.component(name)
        #component.fix(float(initialGuess))
        return component

    def addControlVariable(self, name, lowerBound, upperBound, initialValue) :
        model = self.Model
        theVar =  poenv.Var(self.Domain, bounds=(lowerBound, upperBound), initialize=initialValue)
        model.add_component(name,theVar)
        element = model.component(name)
        return element

pyomoHelper = PyomoHelperFunctions(model, model.t)
semiParam = pyomoHelper.addStateElementToPyomo("semiParam", smaLow, smaHigh, float(initialElements[0]/scalingFactors[0]))
fVar = pyomoHelper.addStateElementToPyomo("f", -0.7, 0.7, float(initialElements[1]))
gVar = pyomoHelper.addStateElementToPyomo("g", -0.7, 0.7, float(initialElements[2]))
hVar = pyomoHelper.addStateElementToPyomo("h", -0.7, 0.7, float(initialElements[3]))
kVar = pyomoHelper.addStateElementToPyomo("k", -0.7, 0.7, float(initialElements[4]))
lonVar = pyomoHelper.addStateElementToPyomo("lon", 0, 8*math.pi, float(initialElements[5]))
massVar = pyomoHelper.addStateElementToPyomo("mass", 0, float(m0Val), float(m0Val))
tfVar = pyomoHelper.addConstantSolveForParameter("tf", tfVal, tfVal, tfVal)
#model.tf = poenv.Var(bounds=(tfVal, tfVal), initialize=tfVal)
#model.tf.fix(float(tfVal))
azimuthControlVar = pyomoHelper.addControlVariable("controlAzimuth", -1*math.pi, math.pi, math.pi/2.0)
elevationControlVar = pyomoHelper.addControlVariable("controlElevation", -0.6, 0.6, 0.0) # although this can go from -90 to 90 deg, common sense suggests that a lower bounds would be appropriate for this problem.  If the optimizer stays at these limits, then increase them
throttleControlVar = pyomoHelper.addControlVariable("throttle", 0.0, 1.0, 1.0)

model.semiParamDot = podae.DerivativeVar(model.semiParam, wrt=model.t)
model.fDot = podae.DerivativeVar(model.f, wrt=model.t)
model.gDot = podae.DerivativeVar(model.g, wrt=model.t)
model.hDot = podae.DerivativeVar(model.h, wrt=model.t)
model.kDot = podae.DerivativeVar(model.k, wrt=model.t)
model.lonDot = podae.DerivativeVar(model.lon, wrt=model.t)
model.mDot = podae.DerivativeVar(model.mass, wrt=model.t)

indexToStateMap = {
0: lambda m, t : m.semiParam[t],
1: lambda m, t : m.f[t],
2: lambda m, t : m.g[t],
3: lambda m, t : m.h[t],
4: lambda m, t : m.k[t],
5: lambda m, t : m.lon[t],
6: lambda m, t : m.mass[t],
}

def mapPyomoStateToProblemState(m, t, expression) :    
    state = [m.semiParam[t], m.f[t], m.g[t],m.h[t], m.k[t], m.lon[t], m.mass[t]]
    args = [muVal, m.controlAzimuth[t], m.controlElevation[t], m.throttle[t], thrustVal, ispVal, m.tf]    
    ans = expression(t, state, *args)
    return ans

model.perEom = poenv.Constraint(model.t, rule =lambda m, t2: m.semiParamDot[t2] == mapPyomoStateToProblemState(m, t2, listOfEomCallback[0]))
model.fEom = poenv.Constraint(model.t, rule =lambda m, t2: m.fDot[t2] == mapPyomoStateToProblemState(m, t2, listOfEomCallback[1]))
model.gEom = poenv.Constraint(model.t, rule =lambda m, t2: m.gDot[t2] == mapPyomoStateToProblemState(m, t2, listOfEomCallback[2]))
model.hEom = poenv.Constraint(model.t, rule =lambda m, t2: m.hDot[t2] == mapPyomoStateToProblemState(m, t2, listOfEomCallback[3]))
model.kEom = poenv.Constraint(model.t, rule =lambda m, t2: m.kDot[t2] == mapPyomoStateToProblemState(m, t2, listOfEomCallback[4]))
model.lonEom = poenv.Constraint(model.t, rule =lambda m, t2: m.lonDot[t2] == mapPyomoStateToProblemState(m, t2, listOfEomCallback[5]))
model.massEom = poenv.Constraint(model.t, rule =lambda m, t2: m.mDot[t2] == mapPyomoStateToProblemState(m, t2, listOfEomCallback[6]))

model.bc1 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[0](mod1, 1.0) - float(finalElements.SemiParameter/scalingFactors[0]))
model.bc2 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[1](mod1, 1.0) - float(finalElements.EccentricityCosTermF))
model.bc3 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[2](mod1, 1.0) - float(finalElements.EccentricitySinTermG))
model.bc4 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[3](mod1, 1.0) - float(finalElements.InclinationCosTermH))
model.bc5 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[4](mod1, 1.0) - float(finalElements.InclinationSinTermK))
#model.bc6 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[5](mod1, 1.0) - float(finalElements.TrueLongitude + (4*math.pi)))
model.bc6 = poenv.Constraint(rule = lambda mod1 : 0 == poenv.sin(indexToStateMap[5](mod1, 1.0)) - math.sin(finalElements.TrueLongitude%(2*math.pi)))
model.bc7 = poenv.Constraint(rule = lambda mod1 : 0 == poenv.cos(indexToStateMap[5](mod1, 1.0)) - math.cos(finalElements.TrueLongitude%(2*math.pi)))

finalMassCallback = lambda m : m.mass[1.0]/150.0 # 150 has worked better than 100, 200 was not great...
model.massObjective = poenv.Objective(expr = finalMassCallback, sense=poenv.maximize)

sim = podae.Simulator(model, package='scipy')
model.var_input = poenv.Suffix(direction=poenv.Suffix.IMPORT)
model.var_input[model.controlAzimuth] = {0.0: math.pi/2.0}
model.var_input[model.controlElevation] = {0.0: 0.0}
model.var_input[model.throttle] = {0.0: 1.0}
tSim, profiles = sim.simulate(numpoints=n, varying_inputs=model.var_input, integrator='dop853')

#poenv.TransformationFactory('dae.finite_difference').apply_to(model, wrt=model.t, nfe=n, scheme='BACKWARD')
print("transforming pyomo")
poenv.TransformationFactory('dae.collocation').apply_to(model, wrt=model.t, nfe=n, ncp=3, scheme='LAGRANGE-RADAU')
#['LAGRANGE-RADAU', 'LAGRANGE-LEGENDRE']
print("initializing the pyomo model")
sim.initialize_model()

print("running the pyomo model")
solver = poenv.SolverFactory('cyipopt')
solver.config.options['tol'] = 1e-6
solver.config.options['max_iter'] = 2000

try :
    solver.solve(model, tee=True)
except Exception as ex:
    print("Whop whop" + str(ex))
#%%

pyomoVarDict = {semiParam: scaledEquationsOfMotion.newStateVariables[0], 
                fVar: scaledEquationsOfMotion.newStateVariables[1],
                gVar: scaledEquationsOfMotion.newStateVariables[2],
                hVar: scaledEquationsOfMotion.newStateVariables[3],
                kVar: scaledEquationsOfMotion.newStateVariables[4],
                lonVar: scaledEquationsOfMotion.newStateVariables[5],
                massVar: scaledEquationsOfMotion.newStateVariables[6],
                azimuthControlVar: azi,
                elevationControlVar: elv,
                throttleControlVar: throttle}

def extractPyomoSolution(model, pyomoVarToSymbolDict):
    tSpace =np.array( [t for t in model.t]) 
    # pSym = np.array([model.semiParam[t]() for t in model.t])
    # fSym = np.array([model.f[t]() for t in model.t])
    # gSym = np.array([model.g[t]() for t in model.t])
    # hSym = np.array([model.h[t]() for t in model.t])
    # kSym = np.array([model.k[t]() for t in model.t])
    # lonSym = np.array([model.lon[t]() for t in model.t])
    # massSym = np.array([model.mass[t]() for t in model.t])
    # controlAzimuth = np.array([model.controlAzimuth[t]() for t in model.t])
    # controlElevation = np.array([model.controlElevation[t]() for t in model.t])
    # throttle = np.array([model.throttle[t]() for t in model.t])
    ansAsDict = OrderedDict()

    for k,v in pyomoVarToSymbolDict.items():
        ansAsDict[v] = np.array([k[t]() for t in model.t])

    return [tSpace, ansAsDict]

stateSymbols = [stateVariables, [azi, elv], throttle]
[tauHistory, dictSolution] = extractPyomoSolution(model, pyomoVarDict)
#time = time*tfVal
timeDescaled, dictSolutionDescaled = scaledEquationsOfMotion.descaleStatesDict(tauHistory, dictSolution, [*scalingFactors, 1], tfVal )
equiElements = []
#%%
for i in range(0, len(timeDescaled)):    
    temp = ModifiedEquinoctialElements(dictSolutionDescaled[stateSymbols[0][0]][i], dictSolutionDescaled[stateSymbols[0][1]][i], dictSolutionDescaled[stateSymbols[0][2]][i], dictSolutionDescaled[stateSymbols[0][3]][i], dictSolutionDescaled[stateSymbols[0][4]][i], dictSolutionDescaled[stateSymbols[0][5]][i], muVal)
    #realEqui = scaleEquinoctialElements(temp, 1.0, 1.0)
    equiElements.append(temp)
#%%
simEqui = []
simOtherValues = {} #type: Dict[sy.Expr, List[float]]
simOtherValues[stateSymbols[0][6]] = []
# simOtherValues[stateSymbols[7]] = []
# simOtherValues[stateSymbols[8]] = []
# simOtherValues[stateSymbols[9]] = []
for i in range(0, len(tSim)) :
    temp = ModifiedEquinoctialElements(profiles[i][0]*scalingFactors[0], profiles[i][1], profiles[i][2], profiles[i][3], profiles[i][4], profiles[i][5], muVal)
    simOtherValues[stateSymbols[0][6]].append(profiles[i][6])
    # simOtherValues[stateSymbols[7]].append(profiles[i][7])
    # simOtherValues[stateSymbols[8]].append(profiles[i][8])
    # simOtherValues[stateSymbols[9]].append(profiles[i][9])
    simEqui.append(temp)

guessMotions = ModifiedEquinoctialElements.CreateEphemeris(simEqui)
simEphemeris = prim.EphemerisArrays()
simEphemeris.InitFromMotions(tSim*tfVal, guessMotions)
simPath = prim.PathPrimitive(simEphemeris)
simPath.color = "#00ff00"

try :
    motions = ModifiedEquinoctialElements.CreateEphemeris(equiElements)
    satEphemeris = prim.EphemerisArrays()
    satEphemeris.InitFromMotions(timeDescaled, motions)
    satPath = prim.PathPrimitive(satEphemeris)
    satPath.color = "#ff00ff"
except :
    print("Couldn't plot optimized path")

earthSolution = solve_ivp(twoBodyOdeCallback, [0.0, tfVal], initialElements.ToArray(), args=(muVal,), t_eval=np.linspace(0.0, tfVal,n), dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
earthPath = ModifiedEquinoctialElementsHelpers.createSatPathFromIvpSolution(earthSolution, muVal, "#00ff00")

marsSolution = solve_ivp(twoBodyOdeCallback, [tfVal, 0.0], finalElements.ToArray(), args=(muVal,), t_eval=np.linspace(tfVal, 0.0, n*2), dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
marsPath = ModifiedEquinoctialElementsHelpers.createSatPathFromIvpSolution(marsSolution, muVal, "#ff0000")

#%%

def convertThrustStartAndStopsToScatter3ds(startAndStopCartesians : List[Tuple[List[float], List[float]]], color : str, width : int):
    quiver = []
    for start, stop in startAndStopCartesians:
        smallDict = DataFrame({"x":np.array([start[0], stop[0]], dtype="float64"), "y":np.array([start[1], stop[1]], dtype="float64"), "z":np.array([start[2], stop[2]], dtype="float64")})
        thisLine = go.Scatter3d(x=smallDict["x"], y=smallDict["y"], z=smallDict["z"], mode="lines", line=dict(color=color, width=1))    
        quiver.append(thisLine)
    return quiver

thrustVectorRun = ModifiedEquinoctialElementsHelpers.getInertialThrustVectorFromDataDict(dictSolution[azi], dictSolution[elv], dictSolution[throttle], equiElements)    
thrustStartAndStops = ModifiedEquinoctialElementsHelpers.createScattersForThrustVectors(satPath.ephemeris, thrustVectorRun,  Au/5.0)
thrustPlotlyItemsRun = convertThrustStartAndStopsToScatter3ds(thrustStartAndStops, "#ff0000", 1)

fig = PlotAndAnimatePlanetsWithPlotly("Earth and Mars", [earthPath, marsPath, satPath], marsPath.ephemeris.T, thrustPlotlyItemsRun)
fig.update_layout()
fig.show()  

azimuthPlotData = prim.XAndYPlottableLineData(timeDescaled, dictSolution[stateSymbols[1][0]]*180.0/math.pi, "azimuth", '#0000ff', 2, 0)
elevationPlotData = prim.XAndYPlottableLineData(timeDescaled, dictSolution[stateSymbols[1][1]]*180.0/math.pi, "elevation", '#00ff00', 2, 0)

plot2DLines([azimuthPlotData, elevationPlotData], "Thrust angles (deg)")

throttle = prim.XAndYPlottableLineData(timeDescaled, dictSolution[stateSymbols[2]], "throttle", '#FF0000', 2)
plot2DLines([throttle], "Throttle (0 to 1)")