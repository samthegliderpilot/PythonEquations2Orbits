#%%
import __init__  #type: ignore
import sympy as sy
# import sys
# sys.path.append('..\\')
# sys.path.append('..\..\\')
from pyeq2orb.ForceModels.TwoBodyForce import CreateTwoBodyMotionMatrix, CreateTwoBodyListForModifiedEquinoctialElements
from pyeq2orb.ScaledSymbolicProblem import ScaledSymbolicProblem
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Coordinates.KeplerianModule import KeplerianElements
from pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule import ModifiedEquinoctialElements, CreateSymbolicElements
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from pyeq2orb.Numerical.LambdifyHelpers import LambdifyHelper, OdeLambdifyHelper, OdeLambdifyHelperWithBoundaryConditions
from typing import List, Dict, cast, Any
from matplotlib.figure import Figure #type: ignore
import scipyPaperPrinter as jh#type: ignore
import pyomo.environ as poenv#type: ignore
import pyomo.dae as podae#type: ignore
from matplotlib.figure import Figure
from pyeq2orb.NumericalProblemFromSymbolic import NumericalProblemFromSymbolicProblem
import numpy as np
import math as math
import plotly.express as px#type: ignore
from plotly.offline import iplot, init_notebook_mode#type: ignore
from plotly.graph_objs import Mesh3d#type: ignore
from scipy.integrate import solve_ivp#type: ignore
import plotly.graph_objects as go #type: ignore
from scipy.integrate import solve_ivp
import pyeq2orb.Graphics.Primitives as prim
from pyeq2orb.Graphics.Plotly2DModule import plot2DLines
from pyeq2orb.Graphics.PlotlyUtilities import PlotAndAnimatePlanetsWithPlotly
from pandas import DataFrame #type: ignore
import plotly.graph_objects as go
from collections import OrderedDict
from scipy.interpolate import splev, splrep #type: ignore
import pyeq2orb.Coordinates.OrbitFunctions as orb
from pyeq2orb.Numerical import OdeHelperModule
from collections import OrderedDict
from pyeq2orb import SafeSubs
from IPython.display import display
#import plotly.io as pio
#pio.renderers.default = "vscode"

# order in paper is perRad, f,g,h,k,l
class HowManyImpulses(SymbolicProblem) :
    def __init__(self):
        super().__init__()
        t = sy.Symbol('t')
        self._timeInitialSymbol = sy.Symbol('t_0', real=True, positive=True)
        self._timeFinalSymbol = sy.Symbol('t_f', real=True, positive=True)
        self._timeSymbol = t
        elements = CreateSymbolicElements(t)
        self._elements = elements
        self._mu = elements.GravitationalParameter
        g = sy.Symbol('g', real=True, positive=True) #9.8065
        f = CreateTwoBodyMotionMatrix(elements, True)
        B = elements.CreatePerturbationMatrix(True)
        azi = sy.Function(r'\theta', real=True)(t)
        elv = sy.Function(r'\phi', real=True)(t)
        self._azi = azi
        self._elv = elv

        ricToInertial = orb.CreateComplicatedRicToInertialMatrix(elements.ToMotionCartesian(True))
        alp = ricToInertial*Cartesian(sy.cos(azi)*sy.cos(elv), sy.sin(azi)*sy.cos(elv), sy.sin(elv))
        jh.showEquation("ric", alp)
        alp = sy.Matrix([[sy.cos(azi)*sy.cos(elv)], [sy.sin(azi)*sy.cos(elv)], [sy.sin(elv)]])
        thrust = sy.Symbol('T')
        m = sy.Function('m')(t)
        throttle = sy.Function('\delta', real=True)(t)
        #overallThrust = B*alp
        overallThrust = thrust*B*alp*(throttle)/(m) 
        stateDynamics = f + overallThrust
        isp = sy.Symbol("I_{sp}")
        c = isp * g

        elementsList = elements.ToArray()
        for i in range(0, len(elementsList)) :
            self.StateVariableDynamics.append(stateDynamics[i])
            self.StateVariables.append(elements[i])
        self.StateVariableDynamics.append(-1*thrust*throttle/(isp*g))
        self.StateVariables.append(m)
        self.ControlVariables.append(azi)
        self.ControlVariables.append(elv)
        self.ControlVariables.append(throttle)
        self._unIntegratedPathCost = throttle* thrust/c
        self._terminalCost = 0
        self.StateVariables.extend(SymbolicProblem.CreateCoVector(self.StateVariables, None, t))
        #self.StateVariableDynamics[self.CostateSymbols[0]] = 
        #self.Hamiltonian = self.CreateHamiltonian(self.CostateSymbols)

        self._gravity = g
        self._thrust = thrust
        self._mass = m
        self._throttle = throttle
        self._alphas = alp
        self._isp = isp
        self._mu = elements.GravitationalParameter
        
        #NEED TO DO BC's

    @property
    def modifiedEquinoctialElements(self) : 
        return self._elements


    @property
    def Thrust(self) :
        return self._thrust

    @property
    def Gravity(self) :
        return self._gravity

    @property
    def Mass(self) :
        return self._mass

    @property
    def Throttle(self) :
        return self._throttle

    @property
    def Azimuth(self) :
        return self._azi

    @property
    def Elevation(self) :
        return self._elv

    @property
    def Isp(self) :
        return self._isp

    @property
    def Mu(self) :
        return self._mu

    def AddFinalConditions(self, pF, fF, gF, hF, kF, lF) :
        elementsAtF = self.CreateVariablesAtTimeFinal(self.StateVariables)
        self.BoundaryConditions.append(elementsAtF[0] - pF)
        self.BoundaryConditions.append(elementsAtF[1] - fF)
        self.BoundaryConditions.append(elementsAtF[2] - gF)
        self.BoundaryConditions.append(elementsAtF[3] - hF)
        self.BoundaryConditions.append(elementsAtF[4] - kF)
        self.BoundaryConditions.append(elementsAtF[5] - lF)

    def flattenEquationsOfMotion(self) :
        auxValues = self.modifiedEquinoctialElements.AuxiliarySymbolsDict()
        for i in range(0, len(self.StateVariableDynamics)) :
            self.StateVariableDynamics[i] = SafeSubs(self.StateVariableDynamics[i], auxValues)

HowManyImpulses()
# Earth to Mars demo
tfVal = 793*86400.0
m0Val = 2000.0
isp = 3000.0
nRev = 2.0
thrustVal =  0.1997*1.2
g = 9.8065 
n = 303
tSpace = np.linspace(0.0, tfVal, n)

Au = 149597870700.0
AuSy = sy.Symbol('A_u')
muVal = 1.32712440042e20
r0 = Cartesian(58252488010.7, 135673782531.3, 2845058.1)
v0 = Cartesian(-27844.5, 11659.9, 0000.3)
initialElements = ModifiedEquinoctialElements.FromMotionCartesian(MotionCartesian(r0, v0), muVal)

gSy = sy.Symbol('g', real=True, positive=True)

rf = Cartesian(36216277800.4, -211692395522.5, -5325189049.9)
vf = Cartesian(24798.8, 6168.2, -480.0)
finalElements = ModifiedEquinoctialElements.FromMotionCartesian(MotionCartesian(rf, vf), muVal)


def GetEquiElementsOutOfIvpResults(ivpResults) :
    t = []
    equi = []
    yFromIntegrator = ivpResults.y 
    for i in range(0, len(yFromIntegrator[0])):
        temp = ModifiedEquinoctialElements(yFromIntegrator[0][i], yFromIntegrator[1][i], yFromIntegrator[2][i], yFromIntegrator[3][i], yFromIntegrator[4][i], yFromIntegrator[5][i], muVal)
        equi.append(temp)
        t.append(ivpResults.t[i])

    if t[0] > t[1] :
        t.reverse()
        equi.reverse()
    return (t, equi)

t = sy.Symbol('t', real=True)
symbolicElements = CreateSymbolicElements(t)

def makeSphere(x, y, z, radius, resolution=10):
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    #colors = ['#00ff00']*len(X)
    #size = [2]*len(X)
    return (X, Y, Z)#, colors, size)
Xs, Ys, Zs = makeSphere(0.0, 0.0, 0.0, 6378137.0)
sphereThing = Mesh3d({
                'x': Xs.flatten(),
                'y': Ys.flatten(),
                'z': Zs.flatten(),
                'alphahull': 0}, color='#0000ff')

def scaleEquinoctialElements(equiElements : ModifiedEquinoctialElements, distanceDivisor, timeDivisor) :
    newPer = equiElements.SemiParameter/distanceDivisor
    newMu = equiElements.GravitationalParameter * timeDivisor*timeDivisor/(distanceDivisor*distanceDivisor*distanceDivisor)
    return ModifiedEquinoctialElements(newPer, equiElements.EccentricityCosTermF, equiElements.EccentricitySinTermG, equiElements.InclinationCosTermH, equiElements.InclinationSinTermK, equiElements.TrueLongitude, newMu)

#initialElements = scaleEquinoctialElements(initialElements, Au, tfVal)
#finalElements = scaleEquinoctialElements(finalElements, Au, tfVal)

muVal = cast(float, initialElements.GravitationalParameter)
per0 = initialElements.SemiParameter
f0 = initialElements.EccentricityCosTermF
g0 = initialElements.EccentricitySinTermG
k0 = initialElements.InclinationCosTermH
h0 = initialElements.InclinationSinTermK
lon0 = initialElements.TrueLongitude
print("making base base problem")
baseProblem = HowManyImpulses()
newSvs = SymbolicProblem.CreateBarVariables(baseProblem.StateVariables, baseProblem.TimeSymbol)
baseProblem.SubstitutionDictionary[baseProblem.Mu] = cast(float, initialElements.GravitationalParameter)
baseProblem.SubstitutionDictionary[baseProblem.Isp] = isp
baseProblem.SubstitutionDictionary[baseProblem.Mass] = m0Val
baseProblem.SubstitutionDictionary[baseProblem.Thrust] = thrustVal
baseProblem.SubstitutionDictionary[AuSy] = Au
baseProblem.SubstitutionDictionary[gSy] = g

integrationSymbols = []
integrationSymbols.extend(baseProblem.StateVariables)

arguments = [baseProblem.Azimuth, baseProblem.Elevation, baseProblem.Throttle, baseProblem.TimeFinalSymbol]
#arguments = [baseProblem.Ux, baseProblem.Uy, baseProblem.Uz, baseProblem.Throttle, baseProblem.TimeFinalSymbol]
# for emK, emV in baseProblem.EquationsOfMotion.items() :
#     jh.showEquation(sy.diff(emK, t), emV)

lambdifyFunctionMap = {'sqrt': poenv.sqrt, 'sin': poenv.sin, 'cos':poenv.cos} #TODO: MOARE!!!!

trivialScalingDic = {} # type: Dict[sy.Symbol, SymbolOrNumber]
for sv in baseProblem.StateVariables :
    trivialScalingDic[sv]=1
trivialScalingDic[baseProblem.StateVariables[0]] = Au/10.0
trivialScalingDic[baseProblem.StateVariables[5]] = 1.0
print("making scaled problem")

for eom in baseProblem.EquationsOfMotionAsEquations:
    jh.showEquation(eom)
baseProblem.flattenEquationsOfMotion()
#%%
scaledProblem = baseProblem.ScaleProblem(baseProblem.StateVariables, trivialScalingDic, baseProblem.TimeFinalSymbol)

asNumericalProblem = OdeLambdifyHelperWithBoundaryConditions.CreateFromProblem(scaledProblem)
for (k,v) in lambdifyFunctionMap.items() :
    asNumericalProblem.FunctionRedirectionArray[k] =v
print("scaled and numerical problems made")



def getInertialThrustVectorFromDataDict(problem : SymbolicProblem, dataDict, muValue) -> List[Cartesian] :
    cartesians = []
    az = dataDict[problem.ControlVariables[0]]
    el = dataDict[problem.ControlVariables[1]]
    mag = dataDict[problem.ControlVariables[2]]
    for i in range(0, len(az)) :
        x = mag[i] * math.cos(az[i])*math.cos(el[i])
        y = mag[i] * math.sin(az[i])*math.cos(el[i])
        z = mag[i] * math.sin(el[i])      
        equiElements = ModifiedEquinoctialElements(dataDict[problem.StateVariables[0]][i], dataDict[problem.StateVariables[1]][i], dataDict[problem.StateVariables[2]][i], dataDict[problem.StateVariables[3]][i], dataDict[problem.StateVariables[4]][i], dataDict[problem.StateVariables[5]][i], muValue)      
        ricToInertial = orb.CreateComplicatedRicToInertialMatrix(equiElements.ToMotionCartesian())
        cartesians.append(ricToInertial*Cartesian(x,y,z))
    return cartesians

def convertIvpResultsToDataDict(problem : SymbolicProblem, ivpResults) -> dict :
    converted = OrderedDict()
    for i in range(0, len(problem.StateVariables)) :
        converted[problem.StateVariables[i]] = ivpResults.y[i]
    return converted

def addFixedAzElMagToDataDict(problem : SymbolicProblem, dataDict, initialAz, initialEl, initialMag):
    dataDict[problem.ControlVariables[0]] = []
    dataDict[problem.ControlVariables[1]] = []
    dataDict[problem.ControlVariables[2]] = []
    for i in range(0, len(dataDict[problem.StateVariables[0]])) :
        dataDict[problem.ControlVariables[0]].append(initialAz)
        dataDict[problem.ControlVariables[1]].append(initialEl)
        dataDict[problem.ControlVariables[2]].append(initialMag)

def createScattersForThrustVectors(ephemeris : prim.EphemerisArrays, inertialThrustVectors : List[Cartesian], color : str, scale : float) -> List[px.scatter_3d] :
    scats = []
    for i in range(0, len(ephemeris.T)) :
        hereX = ephemeris.X[i]
        hereY = ephemeris.Y[i]
        hereZ = ephemeris.Z[i]

        thereX = inertialThrustVectors[i].X*scale + hereX
        thereY = inertialThrustVectors[i].Y*scale + hereY
        thereZ = inertialThrustVectors[i].Z*scale + hereZ
        smallDict = DataFrame({"x":np.array([hereX, thereX], dtype="float64"), "y":np.array([hereY, thereY], dtype="float64"), "z":np.array([hereZ, thereZ], dtype="float64")})
        
        thisLine = go.Scatter3d(x=smallDict["x"], y=smallDict["y"], z=smallDict["z"], mode="lines", line=dict(color=color, width=1))
        scats.append(thisLine)
    #fig = go.Figure(data=scats)
    #fig.show()
    return scats


twoBodyMatrix = CreateTwoBodyListForModifiedEquinoctialElements(symbolicElements)
simpleTwoBodyLambdifyCreator = OdeLambdifyHelper(t, symbolicElements, twoBodyMatrix, [], {symbolicElements.GravitationalParameter: muVal})
odeCallback =simpleTwoBodyLambdifyCreator.CreateSimpleCallbackForSolveIvp()
print("propagating earth and mars")
earthSolution = solve_ivp(odeCallback, [0.0, tfVal], initialElements.ToArray(), args=tuple(), t_eval=np.linspace(0.0, tfVal,n), dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
marsSolution = solve_ivp(odeCallback, [tfVal, 0.0], finalElements.ToArray(), args=tuple(), t_eval=np.linspace(tfVal, 0.0, n*2), dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)

(tArray, equiElements) = GetEquiElementsOutOfIvpResults(earthSolution)
motions = ModifiedEquinoctialElements.CreateEphemeris(equiElements)
earthEphemeris = prim.EphemerisArrays()
earthEphemeris.InitFromMotions(tArray, motions)
earthPath = prim.PathPrimitive(earthEphemeris)
earthPath.color = "#0000ff"

(tArray, equiElements) = GetEquiElementsOutOfIvpResults(marsSolution)
marsMotions = ModifiedEquinoctialElements.CreateEphemeris(equiElements)
marsEphemeris = prim.EphemerisArrays()
marsEphemeris.InitFromMotions(tArray, marsMotions)
marsPath = prim.PathPrimitive(marsEphemeris)
marsPath.color = "#990011"

problemForOneOffPropagation = baseProblem

odeHelper = OdeHelperModule.OdeHelper(problemForOneOffPropagation.TimeSymbol)
initialStateSymbols = SafeSubs(problemForOneOffPropagation.StateVariables, {problemForOneOffPropagation.TimeSymbol: problemForOneOffPropagation.TimeInitialSymbol})
odeHelper.setStateElement(problemForOneOffPropagation.StateVariables[0], problemForOneOffPropagation.StateVariableDynamics[0], initialStateSymbols[0])
odeHelper.setStateElement(problemForOneOffPropagation.StateVariables[1], problemForOneOffPropagation.StateVariableDynamics[1], initialStateSymbols[1])
odeHelper.setStateElement(problemForOneOffPropagation.StateVariables[2], problemForOneOffPropagation.StateVariableDynamics[2], initialStateSymbols[2])
odeHelper.setStateElement(problemForOneOffPropagation.StateVariables[3], problemForOneOffPropagation.StateVariableDynamics[3], initialStateSymbols[3])
odeHelper.setStateElement(problemForOneOffPropagation.StateVariables[4], problemForOneOffPropagation.StateVariableDynamics[4], initialStateSymbols[4])
odeHelper.setStateElement(problemForOneOffPropagation.StateVariables[5], problemForOneOffPropagation.StateVariableDynamics[5], initialStateSymbols[5])
odeHelper.setStateElement(problemForOneOffPropagation.StateVariables[6], problemForOneOffPropagation.StateVariableDynamics[6], initialStateSymbols[6])
odeHelper.lambdifyParameterSymbols.append(baseProblem.Azimuth)
odeHelper.lambdifyParameterSymbols.append(baseProblem.Elevation)
odeHelper.lambdifyParameterSymbols.append(baseProblem.Throttle)
odeHelper.lambdifyParameterSymbols.append(baseProblem.TimeFinalSymbol)
odeHelper.constants = problemForOneOffPropagation.SubstitutionDictionary

callback = odeHelper.createLambdifiedCallback()
initialArray = initialElements.ToArray()
initialArray.append(2000)
fixedAz = math.pi/2.0
fixedEl = 0.0
fixedMag = 1.0
testSolution = solve_ivp(callback, (0.0, tfVal), initialArray, t_eval=np.linspace(0.0, tfVal, n), dense_output=True, method="DOP853", args=(fixedAz, fixedEl, fixedMag, tfVal))
equiElements = []
yFromIntegrator = testSolution.y 
for i in range(0, len(yFromIntegrator[0])):
    temp = ModifiedEquinoctialElements(yFromIntegrator[0][i], yFromIntegrator[1][i], yFromIntegrator[2][i], yFromIntegrator[3][i], yFromIntegrator[4][i], yFromIntegrator[5][i], muVal)
    equiElements.append(temp)
guessMotions = ModifiedEquinoctialElements.CreateEphemeris(equiElements)
simEphemeris = prim.EphemerisArrays()
simEphemeris.InitFromMotions(tSpace, guessMotions)
simPath = prim.PathPrimitive(simEphemeris)
simPath.color = "#00ff00"

print("making pyomo model")

model = poenv.ConcreteModel()
model.t = podae.ContinuousSet(initialize=np.linspace(0.0, 1.0, n), domain=poenv.NonNegativeReals)
smaLow = 126.10e9/trivialScalingDic[baseProblem.StateVariables[0]] # little less than earth
smaHigh = 327.0e9/trivialScalingDic[baseProblem.StateVariables[0]] # little more than mars

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

    def addControlVariable(self, name, lowerBound, upperBound) :
        model = self.Model
        model.add_component(name, poenv.Var(self.Domain, bounds=(lowerBound, upperBound)))
        element = model.component(name)
        return element
    

pyomoHelper = PyomoHelperFunctions(model, model.t)
perRad = pyomoHelper.addStateElementToPyomo("perRad", smaLow, smaHigh, float(per0/trivialScalingDic[baseProblem.StateVariables[0]]))
fVar = pyomoHelper.addStateElementToPyomo("f", -0.7, 0.7, float(f0))
gVar = pyomoHelper.addStateElementToPyomo("g", -0.7, 0.7, float(g0))
hVar = pyomoHelper.addStateElementToPyomo("h", -0.7, 0.7, float(h0))
kVar = pyomoHelper.addStateElementToPyomo("k", -0.7, 0.7, float(k0))
lonVar = pyomoHelper.addStateElementToPyomo("lon", 0, 8*math.pi, float(lon0))
massVar = pyomoHelper.addStateElementToPyomo("mass", 0, float(m0Val), float(m0Val))
tfVar = pyomoHelper.addConstantSolveForParameter("tf", tfVal, tfVal, tfVal)
#model.tf = poenv.Var(bounds=(tfVal, tfVal), initialize=tfVal)
#model.tf.fix(float(tfVal))
azimuthControlVar = pyomoHelper.addControlVariable("controlAzimuth", -1*math.pi, math.pi)
elevationControlVar = pyomoHelper.addControlVariable("controlElevation", -0.6, 0.6) # although this can go from -90 to 90 deg, common sense suggests that a lower bounds would be appropriate for this problem.  If the optimizer stays at these limits, then increase them
throttleControlVar = pyomoHelper.addControlVariable("throttle", 0.0, 1.0)

model.perDot = podae.DerivativeVar(model.perRad, wrt=model.t)
model.fDot = podae.DerivativeVar(model.f, wrt=model.t)
model.gDot = podae.DerivativeVar(model.g, wrt=model.t)
model.hDot = podae.DerivativeVar(model.h, wrt=model.t)
model.kDot = podae.DerivativeVar(model.k, wrt=model.t)
model.lonDot = podae.DerivativeVar(model.lon, wrt=model.t)
model.mDot = podae.DerivativeVar(model.mass, wrt=model.t)

indexToStateMap = {
0: lambda m, t : m.perRad[t],
1: lambda m, t : m.f[t],
2: lambda m, t : m.g[t],
3: lambda m, t : m.h[t],
4: lambda m, t : m.k[t],
5: lambda m, t : m.lon[t],
6: lambda m, t : m.mass[t],
}
#%%
def mapPyomoStateToProblemState(m, t, expression) :    
    state = [t, [m.perRad[t], m.f[t], m.g[t],m.h[t], m.k[t], m.lon[t], m.mass[t], m.controlAzimuth[t], m.controlElevation[t], m.throttle[t], m.tf]]    
    return expression(*state)



display(asNumericalProblem.CheckAllParametersArePresent())
display(asNumericalProblem.LambdifyArguments)

listOfEomCallback = asNumericalProblem.CreateListOfStateVariableCallbacksTimeFirst()

model.perEom = poenv.Constraint(model.t, rule =lambda m, t2: m.perDot[t2] == mapPyomoStateToProblemState(m, t2, listOfEomCallback[0]))
model.fEom = poenv.Constraint(model.t, rule =lambda m, t2: m.fDot[t2] == mapPyomoStateToProblemState(m, t2, listOfEomCallback[1]))
model.gEom = poenv.Constraint(model.t, rule =lambda m, t2: m.gDot[t2] == mapPyomoStateToProblemState(m, t2, listOfEomCallback[2]))
model.hEom = poenv.Constraint(model.t, rule =lambda m, t2: m.hDot[t2] == mapPyomoStateToProblemState(m, t2, listOfEomCallback[3]))
model.kEom = poenv.Constraint(model.t, rule =lambda m, t2: m.kDot[t2] == mapPyomoStateToProblemState(m, t2, listOfEomCallback[4]))
model.lonEom = poenv.Constraint(model.t, rule =lambda m, t2: m.lonDot[t2] == mapPyomoStateToProblemState(m, t2, listOfEomCallback[5]))
model.massEom = poenv.Constraint(model.t, rule =lambda m, t2: m.mDot[t2] == mapPyomoStateToProblemState(m, t2, listOfEomCallback[6]))

model.bc1 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[0](mod1, 1.0) - float(finalElements.SemiParameter/trivialScalingDic[baseProblem.StateVariables[0]]))
model.bc2 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[1](mod1, 1.0) - float(finalElements.EccentricityCosTermF))
model.bc3 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[2](mod1, 1.0) - float(finalElements.EccentricitySinTermG))
model.bc4 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[3](mod1, 1.0) - float(finalElements.InclinationCosTermH))
model.bc5 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[4](mod1, 1.0) - float(finalElements.InclinationSinTermK))
#model.bc6 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[5](mod1, 1.0) - float(finalElements.TrueLongitude + (4*math.pi)))
model.bc6 = poenv.Constraint(rule = lambda mod1 : 0 == poenv.sin(indexToStateMap[5](mod1, 1.0)) - math.sin(finalElements.TrueLongitude%(2*math.pi)))
model.bc7 = poenv.Constraint(rule = lambda mod1 : 0 == poenv.cos(indexToStateMap[5](mod1, 1.0)) - math.cos(finalElements.TrueLongitude%(2*math.pi)))

finalMassCallback = lambda m : m.mass[1.0]/100.0
model.massObjective = poenv.Objective(expr = finalMassCallback, sense=poenv.maximize)

print("simulating the pyomo model")

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
solver.config.options['max_iter'] = 3000

try :
    solver.solve(model, tee=True)
except Exception as ex:
    print("Whop whop" + str(ex))
    #print(lastState)

def extractPyomoSolution(model, stateSymbols):
    tSpace =np.array( [t for t in model.t]) 
    pSym = np.array([model.perRad[t]() for t in model.t])
    fSym = np.array([model.f[t]() for t in model.t])
    gSym = np.array([model.g[t]() for t in model.t])
    hSym = np.array([model.h[t]() for t in model.t])
    kSym = np.array([model.k[t]() for t in model.t])
    lonSym = np.array([model.lon[t]() for t in model.t])
    massSym = np.array([model.mass[t]() for t in model.t])
    controlAzimuth = np.array([model.controlAzimuth[t]() for t in model.t])
    controlElevation = np.array([model.controlElevation[t]() for t in model.t])
    throttle = np.array([model.throttle[t]() for t in model.t])
    ansAsDict = OrderedDict()
    ansAsDict[stateSymbols[0]]= pSym
    ansAsDict[stateSymbols[1]]= fSym
    ansAsDict[stateSymbols[2]]= gSym
    ansAsDict[stateSymbols[3]]= hSym
    ansAsDict[stateSymbols[4]]= kSym
    ansAsDict[stateSymbols[5]]= lonSym
    ansAsDict[stateSymbols[6]]= massSym
    ansAsDict[stateSymbols[7]]= controlAzimuth
    ansAsDict[stateSymbols[8]]= controlElevation
    ansAsDict[stateSymbols[9]]= throttle

    return [tSpace, ansAsDict]

stateSymbols = [*baseProblem.StateVariables, *baseProblem.ControlVariables, baseProblem.Throttle]
[time, dictSolution] = extractPyomoSolution(model, stateSymbols)
time = time*tfVal
dictSolution = scaledProblem.DescaleResults(dictSolution)
equiElements = []
for i in range(0, len(time)):    
    temp = ModifiedEquinoctialElements(dictSolution[stateSymbols[0]][i]*trivialScalingDic[baseProblem.StateVariables[0]], dictSolution[stateSymbols[1]][i], dictSolution[stateSymbols[2]][i], dictSolution[stateSymbols[3]][i], dictSolution[stateSymbols[4]][i], dictSolution[stateSymbols[5]][i], muVal)
    #realEqui = scaleEquinoctialElements(temp, 1.0, 1.0)
    equiElements.append(temp)

simEqui = []
simOtherValues = {} #type: Dict[sy.Expr, List[float]]
simOtherValues[stateSymbols[6]] = []
# simOtherValues[stateSymbols[7]] = []
# simOtherValues[stateSymbols[8]] = []
# simOtherValues[stateSymbols[9]] = []
for i in range(0, len(tSim)) :
    temp = ModifiedEquinoctialElements(profiles[i][0]*trivialScalingDic[baseProblem.StateVariables[0]], profiles[i][1], profiles[i][2], profiles[i][3], profiles[i][4], profiles[i][5], muVal)
    simOtherValues[stateSymbols[6]].append(profiles[i][6])
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
    satEphemeris.InitFromMotions(time, motions)
    satPath = prim.PathPrimitive(satEphemeris)
    satPath.color = "#ff00ff"
except :
    print("Couldn't plot optimized path")




thrustVectorRun = getInertialThrustVectorFromDataDict(baseProblem, dictSolution, muVal)
thrustPlotlyItemsRun = createScattersForThrustVectors(satPath.ephemeris, thrustVectorRun, "#ff0000", Au/10.0)
fig = PlotAndAnimatePlanetsWithPlotly("some title", [earthPath, marsPath, simPath, satPath], time, thrustPlotlyItemsRun)
fig.update_layout()#autosize=False, width=800, height=600)
fig.show()  
azimuthPlotData = prim.XAndYPlottableLineData(time, dictSolution[stateSymbols[7]]*180.0/math.pi, "azimuth", '#0000ff', 2, 0)
elevationPlotData = prim.XAndYPlottableLineData(time, dictSolution[stateSymbols[8]]*180.0/math.pi, "elevation", '#00ff00', 2, 0)

#azimuthPlotDataSim = prim.XAndYPlottableLineData(time, np.array(simOtherValues[stateSymbols[7]])*180.0/math.pi, "azimuth_sim", '#ff00ff', 2, 0)
#elevationPlotDataSim = prim.XAndYPlottableLineData(time, np.array(simOtherValues[stateSymbols[8]])*180.0/math.pi, "elevation_sim", '#ffff00', 2, 0)

plot2DLines([azimuthPlotData, elevationPlotData], "Thrust angles (deg)")

throttle = prim.XAndYPlottableLineData(time, dictSolution[stateSymbols[9]], "throttle", '#FF0000', 2)
plot2DLines([throttle], "Throttle (0 to 1)")


# %%
