#%%
import sympy as sy
import os
import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0]))) # need to import 2 directories up (so pyeq2orb is a subfolder)

from pyeq2orb.ForceModels.TwoBodyForce import CreateTwoBodyMotionMatrix, CreateTwoBodyList
from pyeq2orb.ScaledSymbolicProblem import ScaledSymbolicProblem
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Coordinates.KeplerianModule import KeplerianElements
from pyeq2orb.Coordinates.EquinoctialElements import EquinoctialElements, CreateSymbolicElements
from pyeq2orb.Numerical.LambdifyModule import LambdifyHelper
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
from typing import List, Dict
from matplotlib.figure import Figure
import JupyterHelper as jh
import pyomo.environ as poenv
import pyomo.dae as podae
from matplotlib.figure import Figure
from pyeq2orb.NumericalProblemFromSymbolic import NumericalProblemFromSymbolicProblem
import numpy as np
import math as math
import plotly.express as px
from plotly.offline import iplot, init_notebook_mode
from plotly.graph_objs import Mesh3d
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from pyeq2orb.Numerical.LambdifyModule import LambdifyHelper
from scipy.integrate import solve_ivp
import pyeq2orb.Graphics.Primitives as prim
from pandas import DataFrame
import plotly.graph_objects as go
from collections import OrderedDict
from scipy.interpolate import splev, splrep

#import plotly.io as pio
#pio.renderers.default = "vscode"

def CreateThrustMatrix(eqElements : EquinoctialElements) ->sy.Matrix :
    mu = eqElements.GravitationalParameter
    pEq = eqElements.PeriapsisRadius
    kEq = eqElements.InclinationSinTermK
    hEq = eqElements.InclinationCosTermH
    fEq = eqElements.EccentricityCosTermF
    gEq = eqElements.EccentricitySinTermG
    lEq = eqElements.TrueLongitude
    w = 1+fEq*sy.cos(lEq)+gEq*sy.sin(lEq)
    #s2 = sy.Symbol('s^2')#(heq, keq) # note this is not s but s^2!!! This is a useful cheat
    s2 = 1+hEq**2+kEq**2
    sqrtpOverMu=sy.sqrt(pEq/mu)
    B = sy.Matrix([[0, (2*pEq/w)*sqrtpOverMu, 0],
                [sqrtpOverMu*sy.sin(lEq), sqrtpOverMu*(1/w)*((w+1)*sy.cos(lEq)+fEq), -1*sqrtpOverMu*(gEq/w)*(hEq*sy.sin(lEq)-kEq*sy.cos(lEq))],
                [-1*sqrtpOverMu*sy.cos(lEq), sqrtpOverMu*((w+1)*sy.sin(lEq)+gEq), sqrtpOverMu*(fEq/w)*(hEq*sy.sin(lEq)-kEq*sy.cos(lEq))],
                [0,0,sqrtpOverMu*(s2*sy.cos(lEq)/(2*w))],
                [0,0,sqrtpOverMu*(s2*sy.sin(lEq)/(2*w))],
                [0,0,sqrtpOverMu*(hEq*sy.sin(lEq)-kEq*sy.cos(lEq))]])
    return B

# order in paper is perRad, f,g,h,k,l
class HowManyImpulses(SymbolicProblem) :
    def __init__(self):
        super().__init__()
        t = sy.Symbol('t')
        self._timeInitialSymbol = sy.Symbol('t_0', real=True, positive=True)
        self._timeFinalSymbol = sy.Symbol('t_f', real=True, positive=True)
        self._timeSymbol = t
        elements = CreateSymbolicElements(t)
        self._mu = elements.GravitationalParameter
        g = sy.Symbol('g', real=True, positive=True) #9.8065
        f = CreateTwoBodyMotionMatrix(elements)
        B = elements.CreatePerturbationMatrix()
        #azi = sy.Function(r'\theta', real=True)(t)
        #elv = sy.Function(r'\phi', real=True)(t)
        #self._azi = azi
        #self._elv = elv

        ux = sy.Function('u_x', real=True)(t)
        uy = sy.Function('u_y', real=True)(t)
        uz = sy.Function('u_z', real=True)(t)
        self._ux = ux
        self._uy = uy
        self._uz = uz


        #alp = sy.Matrix([[sy.cos(azi)*sy.cos(elv)], [sy.sin(azi)*sy.cos(elv)], [sy.sin(elv)]])
        alp = sy.Matrix([[ux], [uy], [uz]])
        thrust = sy.Symbol('T')
        m = sy.Function('m')(t)
        throttle = sy.Function('\delta', real=True)(t)
        overallThrust = thrust*B*alp*throttle/(m) 
        eoms = f + overallThrust
        isp = sy.Symbol("I_{sp}")
        c = isp * g

        elementsList = elements.ToArray()
        for i in range(0, len(elementsList)) :
            self.EquationsOfMotion[elementsList[i]] = eoms[i]
            self.StateVariables.append(elementsList[i])
        self.EquationsOfMotion[m]=-1*thrustVal/(isp*g)
        self.StateVariables.append(m)
        #self.StateVariables.append(self._timeFinalSymbol)
        #self.ControlVariables.append(azi)
        #self.ControlVariables.append(elv)
        self.ControlVariables.append(ux)
        self.ControlVariables.append(uy)
        self.ControlVariables.append(uz)


        self.ControlVariables.append(throttle)
        self._unIntegratedPathCost = throttle* thrust/c
        self._terminalCost = 0
        self.CostateSymbols.extend(SymbolicProblem.CreateCoVector(self.IntegrationSymbols, None, t))
        #self.EquationsOfMotion[self.CostateSymbols[0]] = 
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

    # @property
    # def Azimuth(self) :
    #     return self._azi

    # @property
    # def Elevation(self) :
    #     return self._elv

    @property
    def Ux(self) :
        return self._ux

    @property
    def Uy(self) :
        return self._uy
    
    @property    
    def Uz(self) :
        return self._uz

    @property
    def Isp(self) :
        return self._isp

    @property
    def Mu(self) :
        return self._mu

    def AddStandardResultsToFigure(self, figure: Figure, t: List[float], dictionaryOfValueArraysKeyedOffState: Dict[object, List[float]], label: str) -> None:
        pass

    def AddFinalConditions(self, pF, fF, gF, hF, kF, lF) :
        elementsAtF = self.CreateVariablesAtTimeFinal(self.StateVariables)
        self.BoundaryConditions.append(elementsAtF[0] - pF)
        self.BoundaryConditions.append(elementsAtF[1] - fF)
        self.BoundaryConditions.append(elementsAtF[2] - gF)
        self.BoundaryConditions.append(elementsAtF[3] - hF)
        self.BoundaryConditions.append(elementsAtF[4] - kF)
        self.BoundaryConditions.append(elementsAtF[5] - lF)

# Earth to Mars demo
tfVal = 793*86400.0
m0Val = 2000.0
isp = 3000.0
nRev = 2.0
thrustVal =  0.1997
g = 9.8065 
n = 120
tSpace = np.linspace(0.0, tfVal, n)

Au = 149597870700.0
AuSy = sy.Symbol('A_u')
muVal = 1.32712440042e20
r0 = Cartesian(58252488010.7, 135673782531.3, 2845058.1)
v0 = Cartesian(-27844.5, 11659.9, 0000.3)
initialElements = EquinoctialElements.FromMotionCartesian(MotionCartesian(r0, v0), muVal)

gSy = sy.Symbol('g', real=True, positive=True)

rf = Cartesian(36216277800.4, -211692395522.5, -5325189049.9)
vf = Cartesian(24798.8, 6168.2, -480.0)
finalElements = EquinoctialElements.FromMotionCartesian(MotionCartesian(rf, vf), muVal)

t = sy.Symbol('t', real=True)
symbolicElements = CreateSymbolicElements(t)
twoBodyMatrix = CreateTwoBodyList(symbolicElements)
simpleTwoBodyLambidfyCreator = LambdifyHelper(t, symbolicElements.ToArray(), twoBodyMatrix, [], {symbolicElements.GravitationalParameter: muVal})
odeCallback =simpleTwoBodyLambidfyCreator.CreateSimpleCallbackForSolveIvp()
print("propagating earth and mars")
earthSolution = solve_ivp(odeCallback, [0.0, tfVal], initialElements.ToArray(), args=tuple(), t_eval=np.linspace(0.0, tfVal,n), dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
marsSolution = solve_ivp(odeCallback, [tfVal, 0.0], finalElements.ToArray(), args=tuple(), t_eval=np.linspace(tfVal, 0.0, n*2), dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)

def GetEquiElementsOutOfIvpResults(ivpResults) :
    t = []
    equi = []
    yFromIntegrator = ivpResults.y 
    for i in range(0, len(yFromIntegrator[0])):
        temp = EquinoctialElements(yFromIntegrator[0][i], yFromIntegrator[1][i], yFromIntegrator[2][i], yFromIntegrator[3][i], yFromIntegrator[4][i], yFromIntegrator[5][i], muVal)
        equi.append(temp)
        t.append(ivpResults.t[i])

    if t[0] > t[1] :
        t.reverse()
        equi.reverse()
    return (t, equi)

import pyeq2orb.Graphics.Primitives as prim

(tArray, equiElements) = GetEquiElementsOutOfIvpResults(earthSolution)
motions = EquinoctialElements.CreateEphemeris(equiElements)
earthEphemeris = prim.EphemerisArrays()
earthEphemeris.InitFromMotions(tArray, motions)
earthPath = prim.PathPrimitive(earthEphemeris)
earthPath.color = "#0000ff"

(tArray, equiElements) = GetEquiElementsOutOfIvpResults(marsSolution)
marsMotions = EquinoctialElements.CreateEphemeris(equiElements)
marsEphemeris = prim.EphemerisArrays()
marsEphemeris.InitFromMotions(tArray, marsMotions)
marsPath = prim.PathPrimitive(marsEphemeris)
marsPath.color = "#ff0000"

dataArray = []

def makesphere(x, y, z, radius, resolution=10):
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    #colors = ['#00ff00']*len(X)
    #size = [2]*len(X)
    return (X, Y, Z)#, colors, size)
Xs, Ys, Zs = makesphere(0.0, 0.0, 0.0, 6378137.0)
sphereThing = Mesh3d({
                'x': Xs.flatten(),
                'y': Ys.flatten(),
                'z': Zs.flatten(),
                'alphahull': 0}, color='#0000ff')



def scaleEquinoctialElements(equiElements : EquinoctialElements, distanceDivisor, timeDivisor) :
    newPer = equiElements.PeriapsisRadius/distanceDivisor
    newMu = equiElements.GravitationalParameter * timeDivisor*timeDivisor/(distanceDivisor*distanceDivisor*distanceDivisor)
    return EquinoctialElements(newPer, equiElements.EccentricityCosTermF, equiElements.EccentricitySinTermG, equiElements.InclinationCosTermH, equiElements.InclinationSinTermK, equiElements.TrueLongitude, newMu)

#initialElements = scaleEquinoctialElements(initialElements, Au, tfVal)
#finalElements = scaleEquinoctialElements(finalElements, Au, tfVal)

muVal = initialElements.GravitationalParameter
per0 = initialElements.PeriapsisRadius
g0 = initialElements.EccentricitySinTermG
f0 = initialElements.EccentricityCosTermF
k0 = initialElements.InclinationSinTermK
h0 = initialElements.InclinationCosTermH
lon0 = initialElements.TrueLongitude
print("making base base problem")
baseProblem = HowManyImpulses()
newSvs = ScaledSymbolicProblem.CreateBarVariables(baseProblem.StateVariables, baseProblem.TimeSymbol)
baseProblem.SubstitutionDictionary[baseProblem.Mu] = initialElements.GravitationalParameter
baseProblem.SubstitutionDictionary[baseProblem.Isp] = isp
baseProblem.SubstitutionDictionary[baseProblem.Mass] = m0Val
baseProblem.SubstitutionDictionary[baseProblem.Thrust] = thrustVal
baseProblem.SubstitutionDictionary[AuSy] = Au
baseProblem.SubstitutionDictionary[gSy] = g

integrationSymbols = []
integrationSymbols.extend(baseProblem.StateVariables)

#arguments = [baseProblem.Azimuth, baseProblem.Elevation, baseProblem.Throttle, baseProblem.TimeFinalSymbol]
arguments = [baseProblem.Ux, baseProblem.Uy, baseProblem.Uz, baseProblem.Throttle, baseProblem.TimeFinalSymbol]
for emK, emV in baseProblem.EquationsOfMotion.items() :
    jh.showEquation(sy.diff(emK, t), emV)

lambdifyHelper = LambdifyHelper(baseProblem.TimeSymbol, integrationSymbols, baseProblem.EquationsOfMotion.values(), arguments, baseProblem.SubstitutionDictionary)
odeIntEomCallback = lambdifyHelper.CreateSimpleCallbackForSolveIvp()
fullInitialState = []
fullInitialState.extend(initialElements.ToArray())
fullInitialState.append(m0Val)

lambdiafyFunctionMap = {'sqrt': poenv.sqrt, 'sin': poenv.sin, 'cos':poenv.cos} #TODO: MOOOORE!!!!

trivialScalingDic = {}
for sv in baseProblem.StateVariables :
    trivialScalingDic[sv]=1
trivialScalingDic[baseProblem.StateVariables[0]] = Au
print("making scaled problem")
scaledProblem = ScaledSymbolicProblem(baseProblem, baseProblem.StateVariables, trivialScalingDic, True)
asNumericalProblem = NumericalProblemFromSymbolicProblem(scaledProblem, lambdiafyFunctionMap)
print("scaled and numerical problems made")

def PlotAndAnimatePlanetsWithPlotly(title : str, wanderers : List[prim.PathPrimitive], tArray : List[float], thrustVect : List[go.Scatter3d]) :
    lines = []
    maxValue = -1

    #animation arrays
    xArrays = []
    yArrays = []
    zArrays=[]
    for planet in wanderers :
        #colors = np.full(len(planet.ephemeris.T), planet.color)
        dataDict = DataFrame({"x":planet.ephemeris.X, "y":planet.ephemeris.Y, "z": planet.ephemeris.Z })
        thisLine = go.Scatter3d(x=dataDict["x"], y=dataDict["y"], z=dataDict["z"], mode="lines", line=dict(color=planet.color, width=5))
        
        lines.append(thisLine)
        
        thisMax = planet.ephemeris.GetMaximumValue()
        if thisMax > maxValue :
            maxVal = thisMax
        
        # for the animation, we can only have 1 scatter_3d and we need to shuffle all of the 
        # points for all of the planets to be at the same time 
        xForAni = splev(tArray, splrep(planet.ephemeris.T, planet.ephemeris.X))
        yForAni = splev(tArray, splrep(planet.ephemeris.T, planet.ephemeris.Y))
        zForAni = splev(tArray, splrep(planet.ephemeris.T, planet.ephemeris.Z))
        xArrays.append(xForAni)
        yArrays.append(yForAni)
        zArrays.append(zForAni)


    dataDictionary = {"x":[], "y":[], "z":[], "t":[], "color":[], "size":[]}
    t = dataDictionary["t"]
    k = 0
    for step in tArray :
        p = 0
        for cnt in wanderers:
            t.append(step/86400)
            dataDictionary["x"].append(xArrays[p][k])
            dataDictionary["y"].append(yArrays[p][k])
            dataDictionary["z"].append(zArrays[p][k])
            dataDictionary["color"].append(cnt.color)
            dataDictionary["size"].append(7)
            p=p+1
        k=k+1
    
    fig = px.scatter_3d(dataDictionary, title=title, x="x", y="y", z="z", animation_frame="t", color="color", size="size")    

    # make the scaling item
    scalingMarker = go.Scatter3d(name="",
        visible=True,
        showlegend=False,
        opacity=0, # full transparent
        hoverinfo='none',
        x=[0,maxVal],
        y=[0,maxVal],
        z=[0,maxVal])
        
    fig.add_trace(scalingMarker)
    for item in lines :
        fig.add_trace(item)
    if thrustVect != None :
        for thrust in thrustVect :
            fig.add_trace(thrust)
    fig.update_layout(autosize=False, width=1200, height=800)
    fig.show()

def getInertialThrustVectorFromIvpResults(integrationResults, initialValues : List[float]) -> List[Cartesian]:
    cartesians = []
    if len(integrationResults.y) <=7 :
        # use default values
        x = initialValues[2] * math.cos(initialValues[0])*math.cos(initialValues[1])
        y = initialValues[2] * math.sin(initialValues[0])*math.cos(initialValues[1])
        z = initialValues[2] * math.sin(initialValues[1])
        cart = Cartesian(x, y, z)
        for t in integrationResults.t :
            cartesians.append(cart)
    else :
        # assume indices are
        # 7 : azimuth
        # 8 : elevation
        # 9 : magnitude
        cartesians = getInertialThrustVectorFromAzimuthElevationMagnitudeArrays(integrationResults.y[7],integrationResults.y[8],integrationResults.y[9] )
    return cartesians

def getInertialThrustVectorFromAzimuthElevationMagnitudeArrays(az : List[float], el : List[float], mag : List[float]) -> List[Cartesian]:
    cartesians = []
    for i in range(0, len(az)) :
        x = mag[i] * math.cos(az[i])*math.cos(el[i])
        y = mag[i] * math.sin(az[i])*math.cos(el[i])
        z = mag[i] * math.sin(el[i])            
        cartesians.append(Cartesian(x,y,z))
    return cartesians

def createScattersForThrustVecters(ephemeris : prim.EphemerisArrays, inertialThrustVectors : List[Cartesian], color : str, scale : float) -> List[px.scatter_3d] :
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

if False :
    #testSolution = solve_ivp(odeIntEomCallback, [0.0, 1.0*tfVal], fullInitialState, args=tuple([0.0, 0.0, 1.0, tfVal]), t_eval=np.linspace(0.0, tfVal,n), dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
    testSolution = solve_ivp(odeIntEomCallback, [0.0, 1.0*tfVal], fullInitialState, args=tuple([0.0, 1.0, 0.0, 1.0, tfVal]), t_eval=np.linspace(0.0, tfVal,n), dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
    equiElements = []
    yFromIntegrator = testSolution.y 
    for i in range(0, len(yFromIntegrator[0])):
        temp = EquinoctialElements(yFromIntegrator[0][i], yFromIntegrator[1][i], yFromIntegrator[2][i], yFromIntegrator[3][i], yFromIntegrator[4][i], yFromIntegrator[5][i], muVal)
        equiElements.append(temp)
    guessMotions = EquinoctialElements.CreateEphemeris(equiElements)
    simEphem = prim.EphemerisArrays()
    simEphem.InitFromMotions(tSpace, guessMotions)
    simPath = prim.PathPrimitive(simEphem)
    simPath.color = "#00ff00"

    thrustVect = getInertialThrustVectorFromIvpResults(testSolution, [0.0, 0.0, 1.0])
    thrustPlotlyItems = createScattersForThrustVecters(simPath.ephemeris, thrustVect, "#ff0000", Au/10.0)
    PlotAndAnimatePlanetsWithPlotly("Integration sample", [earthPath, marsPath, simPath], tSpace, thrustPlotlyItems)


print("making pyomo model")

model = poenv.ConcreteModel()
model.t = podae.ContinuousSet(initialize=np.linspace(0.0, 1.0, n), domain=poenv.NonNegativeReals)
smaLow = 146.10e9/trivialScalingDic[baseProblem.StateVariables[0]] # little less than earth
smaHigh = 267.0e9/trivialScalingDic[baseProblem.StateVariables[0]] # little more than mars
model.perRad = poenv.Var(model.t, bounds=(smaLow, smaHigh), initialize=float(per0/trivialScalingDic[baseProblem.StateVariables[0]]))
model.f = poenv.Var(model.t, bounds=(-0.3, 0.3), initialize=float(f0))
model.g = poenv.Var(model.t, bounds=(-0.3, 0.3), initialize=float(g0))
model.h  = poenv.Var(model.t, bounds=(-0.3, 0.3), initialize=float(h0))
model.k = poenv.Var(model.t, bounds=(-0.3, 0.3), initialize=float(k0))
model.lon = poenv.Var(model.t, bounds=(0, 6*math.pi), initialize=float(lon0))
model.mass = poenv.Var(model.t, bounds=(0.0, m0Val), initialize=(m0Val))

model.perRad[0].fix(float(per0/trivialScalingDic[baseProblem.StateVariables[0]]))
model.f[0].fix(float(f0))
model.g[0].fix(float(g0))
model.h[0].fix(float(h0))
model.k[0].fix(float(k0))
model.lon[0].fix(float(lon0))
model.mass[0].fix(float(m0Val))

model.tf = poenv.Var(bounds=(tfVal, tfVal), initialize=float(tfVal))
#model.controlAzimuth = poenv.Var(model.t, bounds=(-1*math.pi, math.pi))
#model.controlElevation = poenv.Var(model.t, bounds=(-0.6, 0.6))  # although this can go from -90 to 90 deg, common sense suggests that a lower bounds would be approprate for this problem.  If the optimizer stays at these limits, then increase them
model.ux = poenv.Var(model.t, bounds=(-1.0, 1.0))
model.uy = poenv.Var(model.t, bounds=(-1.0, 1.0))
model.uz = poenv.Var(model.t, bounds=(-1.0, 1.0))
model.throttle = poenv.Var(model.t, bounds=(0.0, 1.0))

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

def finalConditionsCallback(m, t, i) :
    return indexToStateMap[i](m, t)

lastState = None
def mapPyomoStateToProblemState(m, t, expre) :
    global lastState
    #state = [t, m.perRad[t], m.f[t], m.g[t],m.h[t], m.k[t], m.lon[t], m.mass[t], m.controlAzimuth[t], m.controlElevation[t], m.throttle[t], m.tf]
    state = [t, m.perRad[t], m.f[t], m.g[t],m.h[t], m.k[t], m.lon[t], m.mass[t], m.ux[t], m.uy[t], m.uz[t], m.throttle[t], m.tf]
    lastState = state
    return expre(state)

model.perEom = poenv.Constraint(model.t, rule =lambda m, t2: m.perDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 0)))
model.fEom = poenv.Constraint(model.t, rule =lambda m, t2: m.fDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 1)))
model.gEom = poenv.Constraint(model.t, rule =lambda m, t2: m.gDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 2)))
model.hEom = poenv.Constraint(model.t, rule =lambda m, t2: m.hDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 3)))
model.kEom = poenv.Constraint(model.t, rule =lambda m, t2: m.kDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 4)))
model.lonEom = poenv.Constraint(model.t, rule =lambda m, t2: m.lonDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 5)))
model.massEom = poenv.Constraint(model.t, rule =lambda m, t2: m.mDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 6)))

model.bc1 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[0](mod1, 1.0) - float(finalElements.PeriapsisRadius/trivialScalingDic[baseProblem.StateVariables[0]]))
model.bc2 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[1](mod1, 1.0) - float(finalElements.EccentricityCosTermF))
model.bc3 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[2](mod1, 1.0) - float(finalElements.EccentricitySinTermG))
model.bc4 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[3](mod1, 1.0) - float(finalElements.InclinationCosTermH))
model.bc5 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[4](mod1, 1.0) - float(finalElements.InclinationSinTermK))
model.bc6 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[5](mod1, 1.0) - float(finalElements.TrueLongitude))

finalMassCallback = lambda m : m.mass[1.0]
model.massObjective = poenv.Objective(expr = finalMassCallback, sense=poenv.maximize)

model.var_input = poenv.Suffix(direction=poenv.Suffix.LOCAL)
#model.var_input[model.controlAzimuth] = {0: 0.0}
#model.var_input[model.controlElevation] = {0: 0.0}
model.var_input[model.ux] = {0: 0.0}
model.var_input[model.uy] = {0: 1.0}
model.var_input[model.uz] = {0: 0.0}
model.var_input[model.throttle] = {0: 1.0}
model.var_input[model.tf] = {0: tfVal}
print("siming the pyomo model")
sim = podae.Simulator(model, package='scipy')
tsim, profiles = sim.simulate(numpoints=n, varying_inputs=model.var_input, integrator='dop853', initcon=np.array([per0/trivialScalingDic[baseProblem.StateVariables[0]], f0, g0, h0, k0, lon0, m0Val], dtype=float))

#poenv.TransformationFactory('dae.finite_difference').apply_to(model, wrt=model.t, nfe=n, scheme='BACKWARD')
print("transforming pyomo")
poenv.TransformationFactory('dae.collocation').apply_to(model, wrt=model.t, nfe=n,ncp=3, scheme='LAGRANGE-RADAU')
#['LAGRANGE-RADAU', 'LAGRANGE-LEGENDRE']
print("initing the pyomo model")
sim.initialize_model()
print("running the pyomo model")
solver = poenv.SolverFactory('cyipopt')
try :
    solver.solve(model, tee=True)
except Exception as ex:
    print("Whop whop" + str(ex))
    print(lastState)

from collections import OrderedDict
def extractPyomoSolution(model, stateSymbols):
    tSpace =np.array( [t for t in model.t]) 
    pSym = np.array([model.perRad[t]() for t in model.t])
    fSym = np.array([model.f[t]() for t in model.t])
    gSym = np.array([model.g[t]() for t in model.t])
    hSym = np.array([model.h[t]() for t in model.t])
    kSym = np.array([model.k[t]() for t in model.t])
    lonSym = np.array([model.lon[t]() for t in model.t])
    massSym = np.array([model.mass[t]() for t in model.t])
    #controlAzimuth = np.array([model.controlAzimuth[t]() for t in model.t])
    #controlElevation = np.array([model.controlElevation[t]() for t in model.t])
    ux = np.array([model.ux[t]() for t in model.t])
    uy = np.array([model.uy[t]() for t in model.t])
    uz = np.array([model.uz[t]() for t in model.t])
    throttle = np.array([model.throttle[t]() for t in model.t])
    ansAsDict = OrderedDict()
    ansAsDict[stateSymbols[0]]= pSym
    ansAsDict[stateSymbols[1]]= fSym
    ansAsDict[stateSymbols[2]]= gSym
    ansAsDict[stateSymbols[3]]= hSym
    ansAsDict[stateSymbols[4]]= kSym
    ansAsDict[stateSymbols[5]]= lonSym
    ansAsDict[stateSymbols[6]]= massSym
    ansAsDict[stateSymbols[7]]= ux
    ansAsDict[stateSymbols[8]]= uy
    ansAsDict[stateSymbols[9]]= uz
    ansAsDict[stateSymbols[10]]= throttle
    #ansAsDict[stateSymbols[7]]= controlAzimuth
    #ansAsDict[stateSymbols[8]]= controlElevation
    #ansAsDict[stateSymbols[9]]= throttle

    return [tSpace, ansAsDict]

stateSymbols = [*baseProblem.StateVariables, *baseProblem.ControlVariables, baseProblem.Throttle]
[time, dictSolution] = extractPyomoSolution(model, stateSymbols)
time = time*tfVal
dictSolution = scaledProblem.DescaleResults(dictSolution)
equiElements = []
for i in range(0, len(time)):    
    temp = EquinoctialElements(dictSolution[stateSymbols[0]][i]*trivialScalingDic[baseProblem.StateVariables[0]], dictSolution[stateSymbols[1]][i], dictSolution[stateSymbols[2]][i], dictSolution[stateSymbols[3]][i], dictSolution[stateSymbols[4]][i], dictSolution[stateSymbols[5]][i], muVal)
    #realEqui = scaleEquinoctialElements(temp, 1.0, 1.0)
    equiElements.append(temp)

simEqui = []
for i in range(0, len(tsim)) :
    temp = EquinoctialElements(profiles[i][0]*trivialScalingDic[baseProblem.StateVariables[0]], profiles[i][1], profiles[i][2], profiles[i][3], profiles[i][4], profiles[i][5], muVal)
    simEqui.append(temp)

guessMotions = EquinoctialElements.CreateEphemeris(simEqui)
simEphem = prim.EphemerisArrays()
simEphem.InitFromMotions(tsim*tfVal, guessMotions)
simPath = prim.PathPrimitive(simEphem)
simPath.color = "#00ff00"

#dataArray.append(CreatePlotlyLineDataObject(earthPath))
#dataArray.append(CreatePlotlyLineDataObject(marsPath))
try :
    motions = EquinoctialElements.CreateEphemeris(equiElements)
    satEphem = prim.EphemerisArrays()
    satEphem.InitFromMotions(time, motions)
    satPath = prim.PathPrimitive(satEphem)
    satPath.color = "#ff00ff"
    #dataArray.append(CreatePlotlyLineDataObject(satPath))
except :
    print("Couldn't plot optimized path")
thrustVectRun = getInertialThrustVectorFromAzimuthElevationMagnitudeArrays(dictSolution[stateSymbols[7]], dictSolution[stateSymbols[8]], dictSolution[stateSymbols[9]])
thrustPlotlyItemsRun = createScattersForThrustVecters(satPath.ephemeris, thrustVectRun, "#ff0000", Au/10.0)
PlotAndAnimatePlanetsWithPlotly("some title", [earthPath, marsPath, simPath, satPath], time, thrustPlotlyItemsRun)