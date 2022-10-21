#%%
import sympy as sy
import sys
sys.path.append(r'C:\src\PythonEquations2Orbits') # and this line is needed for running like a normal python script
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
from plotly.offline import download_plotlyjs, plot,iplot
from plotly.offline import iplot, init_notebook_mode
from plotly.graph_objs import Mesh3d
from plotly.graph_objs.layout.shape import Line
from scipy.integrate import solve_ivp
from pyeq2orb.Numerical.LambdifyModule import LambdifyHelper
from scipy.integrate import solve_ivp
import pyeq2orb.Graphics.Primitives as prim
import pyeq2orb.Graphics.PlotlyUtilities as plotlyUtil

import plotly.graph_objects as go
from collections import OrderedDict
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
        alp = sy.Matrix([[sy.Function(r'\alpha_x', real=True)(t)],[sy.Function(r'\alpha_y', real=True)(t)],[sy.Function(r'\alpha_z', real=True)(t)]])
        thrust = sy.Symbol('T')
        m = sy.Function('m')(t)
        throttle = sy.Function('\delta', real=True)(t)
        overallThrust = thrust*B*alp*throttle/(m*alp.norm())
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
        for i in range(0, len(alp)) :
            self.ControlVariables.append(alp[i])

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
        #self._mu = elements.GravitationalParameter

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

    @property
    def Alphas(self) :
        return self._alphas

    @property
    def Isp(self) :
        return self._isp

    @property
    def Mu(self) :
        return self._mu

    def AddStandardResultsToFigure(self, figure: Figure, t: List[float], dictionaryOfValueArraysKeyedOffState: Dict[object, List[float]], label: str) -> None:
        pass

    def AddFinalConditions(self, smaF, fF, gF, hF, kF, lF) :
        elementsAtF = self.CreateVariablesAtTimeFinal(self.StateVariables)
        self.BoundaryConditions.append(elementsAtF[0] - smaF)
        self.BoundaryConditions.append(elementsAtF[1] - fF)
        self.BoundaryConditions.append(elementsAtF[2] - gF)
        self.BoundaryConditions.append(elementsAtF[3] - hF)
        self.BoundaryConditions.append(elementsAtF[4] - kF)
        self.BoundaryConditions.append(elementsAtF[5] - lF)

# Earth to Mars demo
Au = 149597870700
AuSy = sy.Symbol('A_u')
muVal = 1.32712440042e20
r0 = Cartesian(58252488010.7, 135673782531.3, 2845058.1)
v0 = Cartesian(-27844.5, 11659.9, 0000.3)
initialElements = EquinoctialElements.FromMotionCartesian(MotionCartesian(r0, v0), muVal)

gSy = sy.Symbol('g', real=True, positive=True)
tfVal = 793*86400
rf = Cartesian(36216277800.4, -211692395522.5, -5325189049.9)
vf = Cartesian(24798.8, 6168.2, -480.0)
finalElements = EquinoctialElements.FromMotionCartesian(MotionCartesian(rf, vf), muVal)

t = sy.Symbol('t', real=True)
symbolicElements = CreateSymbolicElements(t)
twoBodyMatrix = CreateTwoBodyList(symbolicElements)
simpleTwoBodyLambidfyCreator = LambdifyHelper(t, symbolicElements.ToArray(), twoBodyMatrix, [], {symbolicElements.GravitationalParameter: muVal})
odeCallback =simpleTwoBodyLambidfyCreator.CreateSimpleCallbackForSolveIvp()

earthSolution = solve_ivp(odeCallback, [0.0, tfVal], initialElements.ToArray(), args=tuple(), t_eval=np.linspace(0.0, tfVal,900), dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
marsSolution = solve_ivp(odeCallback, [tfVal, 0.0], finalElements.ToArray(), args=tuple(), t_eval=np.linspace(tfVal, 0.0, 900), dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)

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


# def makesphere(x, y, z, radius, resolution=10):
#     """Return the coordinates for plotting a sphere centered at (x,y,z)"""
#     u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
#     X = radius * np.cos(u)*np.sin(v) + x
#     Y = radius * np.sin(u)*np.sin(v) + y
#     Z = radius * np.cos(v) + z
#     #colors = ['#00ff00']*len(X)
#     #size = [2]*len(X)
#     return (X, Y, Z)#, colors, size)
# Xs, Ys, Zs = makesphere(0.0, 0.0, 0.0, 6378137.0)
# sphereThing = Mesh3d({
#                 'x': Xs.flatten(),
#                 'y': Ys.flatten(),
#                 'z': Zs.flatten(),
#                 'alphahull': 0}, color='#0000ff')

def scaleEquinoctialElements(equiElements : EquinoctialElements, distanceDivisor, timeDivisor) :
    newPer = equiElements.PeriapsisRadius/distanceDivisor
    newMu = equiElements.GravitationalParameter * timeDivisor*timeDivisor/(distanceDivisor*distanceDivisor*distanceDivisor)
    return EquinoctialElements(newPer, equiElements.EccentricityCosTermF, equiElements.EccentricitySinTermG, equiElements.InclinationCosTermH, equiElements.InclinationSinTermK, equiElements.TrueLongitude, newMu)

initialElements = scaleEquinoctialElements(initialElements, Au, 1.0)
finalElements = scaleEquinoctialElements(finalElements, Au, 1.0)
muVal = initialElements.GravitationalParameter
per0 = initialElements.PeriapsisRadius
g0 = initialElements.EccentricitySinTermG
f0 = initialElements.EccentricityCosTermF
k0 = initialElements.InclinationSinTermK
h0 = initialElements.InclinationCosTermH
lon0 = initialElements.TrueLongitude

m0Val = 2000
isp = 3000
nRev = 2
thrustVal =  0.1996 / Au
g = 9.8065 / Au
n=300
tSpace = np.linspace(0.0, 1.0, n)

baseProblem = HowManyImpulses()
newSvs = ScaledSymbolicProblem.CreateBarVariables(baseProblem.StateVariables, baseProblem.TimeSymbol)
baseProblem.SubstitutionDictionary[baseProblem.Mu] = initialElements.GravitationalParameter
baseProblem.SubstitutionDictionary[baseProblem.Isp] = isp
baseProblem.SubstitutionDictionary[baseProblem.Mass] = m0Val
baseProblem.SubstitutionDictionary[baseProblem.Thrust] = thrustVal
baseProblem.SubstitutionDictionary[AuSy] = Au
baseProblem.SubstitutionDictionary[gSy] = g

integrationSymbols = baseProblem.StateVariables
#integrationSymbols.append(baseProblem.Mass)
#integrationSymbols.extend(baseProblem.Alphas)
#integrationSymbols.append(baseProblem.Throttle)

#baseProblem.EquationsOfMotion[baseProblem.Alphas[0]]=0.0
#baseProblem.EquationsOfMotion[baseProblem.Alphas[1]]=0.0
#baseProblem.EquationsOfMotion[baseProblem.Alphas[2]]=0.0
#baseProblem.EquationsOfMotion[baseProblem.Throttle]=0.0
#scaledProblem.EquationsOfMotion[baseProblem.TimeFinalSymbol]=tfVal


arguments = [*baseProblem.Alphas, baseProblem.Throttle, baseProblem.TimeFinalSymbol]

for emK, emV in baseProblem.EquationsOfMotion.items() :
    jh.showEquation(sy.diff(emK, t), emV)

lambdifyHelper = LambdifyHelper(baseProblem.TimeSymbol, integrationSymbols, baseProblem.EquationsOfMotion.values(), arguments, baseProblem.SubstitutionDictionary)
odeIntEomCallback = lambdifyHelper.CreateSimpleCallbackForSolveIvp()
lambdiafyFunctionMap = {'sqrt': poenv.sqrt, 'sin': poenv.sin, 'cos':poenv.cos} #TODO: MOOOORE!!!!

trivialScalingDic = {}
for sv in baseProblem.StateVariables :
    trivialScalingDic[sv]=1
print("Creating scaled symbolic problem")
scaledProblem = ScaledSymbolicProblem(baseProblem, baseProblem.StateVariables, trivialScalingDic, True)
print("Creating numerical symbolic problem")
asNumericalProblem = NumericalProblemFromSymbolicProblem(scaledProblem, lambdiafyFunctionMap)

# asManualNumericalProblem = NumericalProblemFromSymbolicProblem(scaledProblem, {})
# print("lambdifying")
# lambdifyHelper2 = LambdifyHelper(baseProblem.TimeSymbol, integrationSymbols, baseProblem.EquationsOfMotion.values(), arguments, baseProblem.SubstitutionDictionary)
# odeIntEomCallback2 = lambdifyHelper2.CreateSimpleCallbackForSolveIvp()
# print("integrating")

# testSolution = solve_ivp(odeIntEomCallback2, [0.0, tfVal], [*initialElements.ToArray(), 2000.0], args=tuple([0.0, 1.0, 0.0, 1.0, tfVal]), t_eval=np.linspace(0.0, tfVal,900), dense_output=True)#, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)


# (tArray, equiElements) = GetEquiElementsOutOfIvpResults(earthSolution)
# motions = EquinoctialElements.CreateEphemeris(equiElements)
# earthEphemeris = prim.EphemerisArrays()
# earthEphemeris.InitFromMotions(tArray, motions)
# earthPlanet = prim.PlanetPrimitive(earthEphemeris, 3, 1, '#0000ff', 6378137, "Earth")

# (tArray, equiElements) = GetEquiElementsOutOfIvpResults(marsSolution)
# marsMotions = EquinoctialElements.CreateEphemeris(equiElements)
# marsEphemeris = prim.EphemerisArrays()
# marsEphemeris.InitFromMotions(tArray, marsMotions)
# marsPlanet = prim.PlanetPrimitive(marsEphemeris, 3, 1, '#ff0000', 4000000, "Mars")

# (tArray, satEqui) = GetEquiElementsOutOfIvpResults(testSolution)
# satEquiElements = []
# for i in range(0, len(tArray)):
#     temp = satEqui[i]
#     realEqui = scaleEquinoctialElements(temp, 1.0/Au, 1.0/tfVal)
#     satEquiElements.append(realEqui)
# satMotions = EquinoctialElements.CreateEphemeris(satEquiElements)
# satEphemeris = prim.EphemerisArrays()
# satEphemeris.InitFromMotions(tArray, satMotions)
# satPlanet = prim.PlanetPrimitive(satEphemeris, 3, 1, '#ff00ff', 10, "Satellite")

# plotlyHelper = plotlyUtil.PlotlyDataAndFramesAccumulator()
# allPrims = [earthPlanet, marsPlanet, satPlanet]
# plotlyHelper.AddLinePrimitives(allPrims)
# plotlyHelper.AddMarkerPrimitives(tArray, allPrims)
# plotlyHelper.AddScalingPoints(allPrims)

# overallFig = go.Figure(data=plotlyHelper.data, frames=plotlyHelper.frames)
# for item in plotlyHelper.data:
#     overallFig.add_trace(item)
# overallFig.update_layout(updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(redraw=True,fromcurrent=True, mode='immediate'))])])])
# overallFig.show()
#%%
model = poenv.ConcreteModel()
model.t = podae.ContinuousSet(initialize=tSpace, domain=poenv.NonNegativeReals)
smaLow = 146.10e9/Au # little less than earth
smaHigh = 229.0e9/Au # little more than mars
model.perRad = poenv.Var(model.t, bounds=(smaLow, smaHigh), initialize=float(per0))
model.f = poenv.Var(model.t, bounds=(-1.0, 1.0), initialize=float(f0))
model.g = poenv.Var(model.t, bounds=(-1.0, 1.0), initialize=float(g0))
model.h  = poenv.Var(model.t, bounds=(-2.0, 2.0), initialize=float(h0))
model.k = poenv.Var(model.t, bounds=(-2.0, 2.0), initialize=float(k0))
model.lon = poenv.Var(model.t, bounds=(0, 4*math.pi), initialize=float(lon0))
model.mass = poenv.Var(model.t, bounds=(0.0, m0Val), initialize=float(m0Val))

model.perRad[0].fix(float(initialElements.PeriapsisRadius))
model.f[0].fix(float(initialElements.EccentricityCosTermF))
model.g[0].fix(float(initialElements.EccentricitySinTermG))
model.h[0].fix(float(initialElements.InclinationCosTermH))
model.k[0].fix(float(initialElements.InclinationSinTermK))
model.lon[0].fix(float(initialElements.TrueLongitude))
model.mass[0].fix(float(m0Val))

model.controlX = poenv.Var(model.t, bounds=(-1.0, 1.0))
model.controlY = poenv.Var(model.t, bounds=(-1.0, 1.0))
model.controlZ = poenv.Var(model.t, bounds=(-1.0, 1.0))
model.throttle = poenv.Var(model.t, bounds=(0.9, 1.0))
model.tf = poenv.Var(bounds=(tfVal-10000, tfVal+10000), initialize=float(tfVal))

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

def mapPyomoStateToProblemState(m, t, expre) :
    return expre([t, m.perRad[t], m.f[t], m.g[t],m.h[t], m.k[t], m.lon[t], m.mass[t], m.controlX[t], m.controlY[t], m.controlZ[t], m.throttle[t], m.tf])

model.perEom = poenv.Constraint(model.t, rule =lambda m, t2: m.perDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 0)))
model.fEom = poenv.Constraint(model.t, rule =lambda m, t2: m.fDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 1)))
model.gEom = poenv.Constraint(model.t, rule =lambda m, t2: m.gDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 2)))
model.hEom = poenv.Constraint(model.t, rule =lambda m, t2: m.hDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 3)))
model.kEom = poenv.Constraint(model.t, rule =lambda m, t2: m.kDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 4)))
model.lonEom = poenv.Constraint(model.t, rule =lambda m, t2: m.lonDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 5)))
model.massEom = poenv.Constraint(model.t, rule =lambda m, t2: m.mDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 6)))

model.bc1 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[0](mod1, 1.0) - float(finalElements.PeriapsisRadius))
model.bc2 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[1](mod1, 1.0) - float(finalElements.EccentricityCosTermF))
model.bc3 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[2](mod1, 1.0) - float(finalElements.EccentricitySinTermG))
model.bc4 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[3](mod1, 1.0) - float(finalElements.InclinationCosTermH))
model.bc5 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[4](mod1, 1.0) - float(finalElements.InclinationSinTermK))
model.bc6 = poenv.Constraint(rule = lambda mod1 : 0 == indexToStateMap[5](mod1, 1.0) - float(finalElements.TrueLongitude))

finalMassCallback = lambda m : m.mass[1.0]
model.massObjective = poenv.Objective(expr = finalMassCallback, sense=poenv.maximize)

model.var_input = poenv.Suffix(direction=poenv.Suffix.LOCAL)
model.var_input[model.tf] = {0: tfVal}
model.var_input[model.controlX] = {0: 0.0}
model.var_input[model.controlY] = {0: 1.0}
model.var_input[model.controlZ] = {0: 0.0}
model.var_input[model.throttle] = {0: 1.0}

sim = podae.Simulator(model, package='scipy')
tsim, profiles = sim.simulate(numpoints=n, varying_inputs=model.var_input, integrator='dop853', initcon=np.array([per0, f0, g0, h0, k0, lon0, m0Val], dtype=float))

#poenv.TransformationFactory('dae.finite_difference').apply_to(model, wrt=model.t, nfe=n, scheme='BACKWARD')
poenv.TransformationFactory('dae.collocation').apply_to(model, wrt=model.t, nfe=n,ncp=3, scheme='LAGRANGE-RADAU')
#['LAGRANGE-RADAU', 'LAGRANGE-LEGENDRE']
sim.initialize_model()
solver = poenv.SolverFactory('cyipopt')
#solver.solve(model, tee=True)

def extractPyomoSolution(model, stateSymbols):
    tSpace =np.array( [t for t in model.t]) * model.tf.value
    pSym = np.array([model.perRad[t]() for t in model.t])
    fSym = np.array([model.f[t]() for t in model.t])
    gSym = np.array([model.g[t]() for t in model.t])
    hSym = np.array([model.h[t]() for t in model.t])
    kSym = np.array([model.k[t]() for t in model.t])
    lonSym = np.array([model.lon[t]() for t in model.t])
    controlX = np.array([model.controlX[t]() for t in model.t])
    controlY = np.array([model.controlY[t]() for t in model.t])
    controlZ = np.array([model.controlZ[t]() for t in model.t])
    throttle = np.array([model.throttle[t]() for t in model.t])
    # print("control 0 = " + str(controls[0]))
    # plt.title("Thrust Angle")
    # plt.plot(tSpace/86400, controls*180.0/math.pi, label="Thrust Angle (deg)")
    # plt.tight_layout()
    # plt.grid(alpha=0.5)
    # plt.legend(framealpha=1, shadow=True)
    # plt.show()    
    ansAsDict = OrderedDict()
    ansAsDict[stateSymbols[0]]= pSym
    ansAsDict[stateSymbols[1]]= fSym
    ansAsDict[stateSymbols[2]]= gSym
    ansAsDict[stateSymbols[3]]= hSym
    ansAsDict[stateSymbols[4]]= kSym
    ansAsDict[stateSymbols[5]]= lonSym
    ansAsDict[stateSymbols[6]]= controlX
    ansAsDict[stateSymbols[7]]= controlY
    ansAsDict[stateSymbols[8]]= controlZ
    ansAsDict[stateSymbols[9]]= throttle

    return [tSpace, ansAsDict]

stateSymbols = [*baseProblem.StateVariables, *baseProblem.ControlVariables, baseProblem.Throttle]
[time, dictSolution] = extractPyomoSolution(model, stateSymbols)

satEquiElements = []
for i in range(0, len(time)):
    temp = EquinoctialElements(dictSolution[stateSymbols[0]][i], dictSolution[stateSymbols[1]][i], dictSolution[stateSymbols[2]][i], dictSolution[stateSymbols[3]][i], dictSolution[stateSymbols[4]][i], dictSolution[stateSymbols[5]][i], muVal)
    realEqui = scaleEquinoctialElements(temp, 1.0/Au, 1.0/tfVal)
    satEquiElements.append(realEqui)


(tArray, equiElements) = GetEquiElementsOutOfIvpResults(earthSolution)
motions = EquinoctialElements.CreateEphemeris(equiElements)
earthEphemeris = prim.EphemerisArrays()
earthEphemeris.InitFromMotions(tArray, motions)
earthPlanet = prim.PlanetPrimitive(earthEphemeris, 3, 1, '#0000ff', 6378137, "Earth")

(tArray, equiElements) = GetEquiElementsOutOfIvpResults(marsSolution)
marsMotions = EquinoctialElements.CreateEphemeris(equiElements)
marsEphemeris = prim.EphemerisArrays()
marsEphemeris.InitFromMotions(tArray, marsMotions)
marsPlanet = prim.PlanetPrimitive(marsEphemeris, 3, 1, '#ff0000', 4000000, "Mars")

class sim :
    def __init__(self, y, t) :
        self.y = y
        self.t = t
array = np.array(profiles)
transposed_array = array.T
transposed_list_of_lists = transposed_array.tolist()
(tArray, satElements) =GetEquiElementsOutOfIvpResults(sim(transposed_list_of_lists, tsim))
satEquiElements = []
for i in range(0, len(tArray)):
    temp = satElements[i]
    realEqui = scaleEquinoctialElements(temp, 1.0/Au, 1.0)
    satEquiElements.append(realEqui)
satMotions = EquinoctialElements.CreateEphemeris(satEquiElements)
satEphemeris = prim.EphemerisArrays()
satEphemeris.InitFromMotions(tsim, satMotions)
satPlanet = prim.PlanetPrimitive(satEphemeris, 3, 1, '#ff00ff', 10.0, "Satellite")

plotlyHelper = plotlyUtil.PlotlyDataAndFramesAccumulator()
allPrims = [earthPlanet, marsPlanet, satPlanet]
plotlyHelper.AddLinePrimitives(allPrims)
plotlyHelper.AddMarkerPrimitives(tArray, allPrims)
plotlyHelper.AddScalingPoints(allPrims)

overallFig = go.Figure(data=plotlyHelper.data, frames=plotlyHelper.frames)
for item in plotlyHelper.data:
    overallFig.add_trace(item)
overallFig.update_layout(updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(redraw=True,fromcurrent=True, mode='immediate'))])])])
overallFig.show()