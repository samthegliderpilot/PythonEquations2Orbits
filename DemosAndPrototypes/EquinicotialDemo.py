#%%
import sympy as sy
import sys
sys.path.append(r'C:\src\PythonEquations2Orbits') # and this line is needed for running like a normal python script
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Coordinates.EquinoctialElements import EquinoctialElements, CreateSymbolicElements
import EquinicotialDemo as ed
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

def CreateTwoBodyMotionMatrix(eqElements : EquinoctialElements) ->sy.Matrix :
    mu = eqElements.GravitationalParameter
    peq = eqElements.InclinationSinTermP
    qeq = eqElements.InclinationCosTermQ
    keq = eqElements.EccentricityCosTermK
    heq = eqElements.EccentricitySinTermH
    leq = eqElements.TrueLongitude   
    #wsy = sy.Symbol("w")#(feq, geq, leq)
    w = 1+keq*sy.cos(leq)+heq*sy.sin(leq)
    #s2 = sy.Symbol('s^2')#(heq, keq) # note this is not s but s^2!!! This is a useful cheat
    #s2Func = 1+heq**2+keq**2

    f = sy.Matrix([[0],[0],[0],[0],[0],[sy.sqrt(mu*peq)*(w/peq)**2]])
    return f

def CreateThrustMatrix(eqElements : EquinoctialElements) ->sy.Matrix :
    mu = eqElements.GravitationalParameter
    peq = eqElements.InclinationSinTermP
    qeq = eqElements.InclinationCosTermQ
    keq = eqElements.EccentricityCosTermK
    heq = eqElements.EccentricitySinTermH
    leq = eqElements.TrueLongitude    
    w = 1+keq*sy.cos(leq)+heq*sy.sin(leq)
    s2 = sy.Symbol('s^2')#(heq, keq) # note this is not s but s^2!!! This is a useful cheat
    s2Func = 1+qeq**2+peq**2
    sqrtpOverMu=sy.sqrt(peq/mu)
    B = sy.Matrix([[0, (2*peq/w)*sqrtpOverMu, 0],
                [sqrtpOverMu*sy.sin(leq), sqrtpOverMu*(1/w)*((w+1)*sy.cos(leq)+keq), -1*sqrtpOverMu*(heq/w)*(heq*sy.sin(leq)-keq*sy.cos(leq))],
                [-1*sqrtpOverMu*sy.cos(leq), sqrtpOverMu*(1/w)*((w+1)*sy.sin(leq)+heq), sqrtpOverMu*(keq/w)*(heq*sy.sin(leq)-keq*sy.cos(leq))],
                [0,0,sqrtpOverMu*(s2*sy.cos(leq)/(2*w))],
                [0,0,sqrtpOverMu*(s2*sy.sin(leq)/(2*w))],
                [0,0,sqrtpOverMu*(1/w)*(heq*sy.sin(leq)-keq*sy.cos(leq))]])
    return B
# order in paper is sma, f,g,h,k,l
# sma is sma
# f  is 
class HowManyImpulses(SymbolicProblem) :
    def __init__(self):
        super().__init__()
        t = sy.Symbol('t')
        self._timeInitialSymbol = sy.Symbol('t_0')
        self._timeFinalSymbol = sy.Symbol('t_f')
        self._timeSymbol = t
        elements = CreateSymbolicElements(t)
        g = sy.Symbol('g') #9.8065
        f = ed.CreateTwoBodyMotionMatrix(elements)
        B = ed.CreateThrustMatrix(elements)        
        alp = sy.Matrix([[sy.Symbol(r'alpha_x', real=True)],[sy.Symbol(r'alpha_y', real=True)],[sy.Symbol(r'alpha_z', real=True)]])
        thrust = sy.Symbol('T')
        m = sy.Symbol('m')
        throttle = sy.Symbol('\delta')
        overallThrust = thrust*B*alp*throttle/m
        eoms = f + overallThrust
        c = sy.Symbol("I_{xp}") * g

        elementsList = elements.ToArray()
        for i in range(0, len(elementsList)) :
            self.EquationsOfMotion[elementsList[i]] = eoms[i]
            self._integrationSymbols.append(elementsList[i])
            self.StateVariables.append(elementsList[i])

        for i in range(0, len(alp)) :
            self.ControlVariables.append(alp[i])

        self._unIntegratedPathCost = throttle* thrust/c
        self._terminalCost = 0
        self.CostateSymbols.extend(SymbolicProblem.CreateCoVector(elementsList, None, t))
        self._integrationSymbols.extend(self.CostateSymbols)
        self.Hamiltonian = self.CreateHamiltonian(self.CostateSymbols)

        #NEED TO DO BC's

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
muVal =1.32712440042e20

r0 = Cartesian(58252488010.7, 135673782.5313, 2845058.1)
v0 = Cartesian(-27844.5, 11659.9, 0000.3)

initialElements = EquinoctialElements.FromMotionCartesian(MotionCartesian(r0, v0), muVal)
sma0 = initialElements.SemiMajorAxis
f0 = initialElements.EccentricitySinTermH
g0 = initialElements.EccentricityCosTermK
p0 = initialElements.InclinationSinTermP
q0 = initialElements.InclinationCosTermQ

rf = Cartesian(36216277800.4, -211692395522.5, -5325189049.9)
vf = Cartesian(24798.8, 6168.2, -480.0)
tf = 793*86400
m0Val = 2000
isp = 3000
nRev = 2


problem = HowManyImpulses()
jh.showEquation("H", problem.Hamiltonian)

lambdiafyFunctionMap = {'sqrt': poenv.sqrt, 'sin': poenv.sin, 'cos':poenv.cos} #TODO: MOOOORE!!!!

asNumericalProblem = NumericalProblemFromSymbolicProblem(problem, lambdiafyFunctionMap)

n=200
tSpace = np.linspace(0.0, 1.0, n)

model = poenv.ConcreteModel()
model.t = podae.ContinuousSet(initialize=tSpace, domain=poenv.NonNegativeReals)
smaLow = 146.10e9 # little less than earth
smaHigh = 229.0e9 # little more than mars
model.sma = poenv.Var(model.t, bounds=(smaLow, smaHigh), initialize=float(sma0))
model.f = poenv.Var(model.t, bounds=(-1.0, 1.0), initialize=float(f0))
model.g = poenv.Var(model.t, bounds=(-1.0, 1.0), initialize=float(g0))
model.p  = poenv.Var(model.t, bounds=(-1.0, 1.0), initialize=float(p0))
model.q = poenv.Var(model.t, bounds=(-1.0, 1.0), initialize=float(q0))
model.lon = poenv.Var(model.t, bounds=(0, 4*math.pi), initialize=float(l0))
model.tf = poenv.Var(bounds=(tf-2, tf+2), initialize=float(tf))

model.controlX = poenv.Var(model.t, bounds=(-1.0, 1.0))
model.controlY = poenv.Var(model.t, bounds=(-1.0, 1.0))
model.controlZ = poenv.Var(model.t, bounds=(-1.0, 1.0))


model.sma[0].fix(float(sma0))
model.f[0].fix(float(f0))
model.g[0].fix(float(g0))
model.p[0].fix(float(q0))
model.q[0].fix(float(q0))
model.lon[0].fix(float(lon0))

model.smaDot = podae.DerivativeVar(model.sma, wrt=model.t)
model.fDot = podae.DerivativeVar(model.f, wrt=model.t)
model.gDot = podae.DerivativeVar(model.g, wrt=model.t)
model.pDot = podae.DerivativeVar(model.p, wrt=model.t)
model.qDot = podae.DerivativeVar(model.q, wrt=model.t)
model.lonDot = podae.DerivativeVar(model.lon, wrt=model.t)

def mapPyomoStateToProblemState(m, t, expre) :
    return expre([t, m.sma[t], m.f[t], m.g[t],m.p[t], m.q[t], m.lon[t], m.controlX[t], m.controlY[t], m.controlZ[t], m.tf])

model.smaEom = poenv.Constraint(model.t, rule =lambda m, t2: m.rDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 0)))
model.fEom = poenv.Constraint(model.t, rule =lambda m, t2: m.uDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 1)))
model.gEom = poenv.Constraint(model.t, rule =lambda m, t2: m.vDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 2)))
model.gEom = poenv.Constraint(model.t, rule =lambda m, t2: m.vDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 2)))
model.gEom = poenv.Constraint(model.t, rule =lambda m, t2: m.vDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 2)))
model.lonEom = poenv.Constraint(model.t, rule =lambda m, t2: m.lonDot[t2] == mapPyomoStateToProblemState(m, t2, lambda state : asNumericalProblem.SingleEquationOfMotionWithTInState(state, 3)))

model.bc1 = poenv.Constraint(rule = lambda mod1 : 0 == mapPyomoStateToProblemState(mod1, 1.0, asNumericalProblem.BoundaryConditionCallbacks[0]))
model.bc2 = poenv.Constraint(rule = lambda mod1 : 0 == mapPyomoStateToProblemState(mod1, 1.0, asNumericalProblem.BoundaryConditionCallbacks[1]))
model.bc3 = poenv.Constraint(rule = lambda mod1 : 0 == mapPyomoStateToProblemState(mod1, 1.0, asNumericalProblem.BoundaryConditionCallbacks[2]))
model.bc4 = poenv.Constraint(rule = lambda mod1 : 0 == mapPyomoStateToProblemState(mod1, 1.0, asNumericalProblem.BoundaryConditionCallbacks[3]))
model.bc5 = poenv.Constraint(rule = lambda mod1 : 0 == mapPyomoStateToProblemState(mod1, 1.0, asNumericalProblem.BoundaryConditionCallbacks[4]))
model.bc6 = poenv.Constraint(rule = lambda mod1 : 0 == mapPyomoStateToProblemState(mod1, 1.0, asNumericalProblem.BoundaryConditionCallbacks[5]))

def singlePyomoArrayToTerminalCostCallback(m, t, expr) :
    return expr(m.tf, [t, m.sma[t], m.f[t], m.g[t],m.p[t], m.q[t], m.lon[t], m.control[t], m.controlX[t], m.controlY[t], m.controlZ[t]])
#finalRadiusCallback = lambda m : singlePyomoArrayToTerminalCostCallback(m, 1.0, asNumericalProblem.TerminalCost)
#model.radiusObjective = poenv.Objective(expr = finalRadiusCallback, sense=poenv.maximize)

model.var_input = poenv.Suffix(direction=poenv.Suffix.LOCAL)
model.var_input[model.control] = {0: 0.03}
model.var_input[model.tf] = {0: tf}

sim = podae.Simulator(model, package='scipy') 
tsim, profiles = sim.simulate(numpoints=n, varying_inputs=model.var_input, integrator='dop853', initcon=np.array([sma0, f0, g0, p0, q0, lon0], dtype=float))

#poenv.TransformationFactory('dae.finite_difference').apply_to(model, wrt=model.t, nfe=n, scheme='BACKWARD')
poenv.TransformationFactory('dae.collocation').apply_to(model, wrt=model.t, nfe=n,ncp=3, scheme='LAGRANGE-RADAU')
#['LAGRANGE-RADAU', 'LAGRANGE-LEGENDRE']
sim.initialize_model()
solver = poenv.SolverFactory('cyipopt')
solver.solve(model, tee=True)

#%%
import plotly.express as px

#fig = px.line(df, x="year", y="lifeExp", title='Life expectancy in Canada')
fig.show()