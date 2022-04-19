#%%
import sys
from typing import OrderedDict
from attr import asdict
sys.path.append("..") # treating this as a jupyter-like cell requires adding one directory up
sys.path.append("../PythonOptimizationWithNlp") # and this line is needed for running like a normal python script
# these two appends do not conflict with eachother

import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
import math

from PythonOptimizationWithNlp.SymbolicOptimizerProblem import PlanerLeoToGeoSymbolicEquationsSy, SymbolicProblem
from PythonOptimizationWithNlp.Symbolics import ScalingHelper
from PythonOptimizationWithNlp.Symbolics.Vectors import Vector

import JupyterHelper as jh

problem = PlanerLeoToGeoSymbolicEquationsSy()

jh.t = problem.Ts
jh.printMarkdown("# Two Dimensional LEO to GEO Transfer with Time Varying Mass")
jh.printMarkdown('The two dimensional LEO to GEO transfer has been a very useful problem to study in optimal control.  Having found solutions using both direct and indirect methods, it is complicated enough to learn insights to the various solving methods, while at the same time being tractable enough to not get stuck.  This script will show several ways to solve the problem as well as comparing the different solutions.')

jh.printMarkdown('To starts with, we state the problem.  There are many different ways to form the problem, but for a given final time, we will maximize the final radius:')
jh.showEquation(problem.CreateCostFunctionAsEquation())

jh.printMarkdown('Subject to the equations of motion:')
for eq in problem.CreateEquationOfMotionsAsEquations() :
    jh.showEquation(eq)
jh.printMarkdown(r'Where $\alpha$ is the control value to derive.')
jh.printMarkdown('Additionally, there is a full set of initial conditions for the four state variables, and all of the final conditions will be specified except for the final longitude.')

g = 9.80665
mu = 3.986004418e14  
thrust = 20.0
isp = 6000.0
m0 = 1500.0

r0 = 6678000.0
u0 = 0.0
v0 = sy.sqrt(mu/r0) # circular
lon0 = 0.0

jh.printMarkdown('### Indirect Method')
jh.printMarkdown('The indirect method will have us find co-state equations of motion whose initial conditions we have to find.  Also there will be an additional set of optimality conditions that must be satisfied.  We start by creating those costate variables.')

lambdas = IndirectHelpers.CreateCoVector(problem.StateVariables, r'\lambda', problem.TimeSymbol)
hamiltonian = IndirectHelpers.CreateHamiltonian(problem, lambdas)
jh.showEquation('H', hamiltonian)

jh.printMarkdown('We can derive the control law by taking the derivative of the Hamiltonion with respect to the control variables and setting it equal to 0.  That must hold along the optimal trajectory.  I am not sure of a general way to take advantage of this; the few other problems I\'d solved have handled this fact opertunistically based on the details of the problem.')
dHdu = IndirectHelpers.CreateHamiltonianControlConditions(problem, hamiltonian).doit()[0]
jh.showEquation("\\frac{dH}{du}", dHdu)
dHduEq = sy.Eq(0, dHdu).expand().simplify()
jh.showEquation(dHduEq)
controlSolved = sy.solve(dHduEq, problem.ControlVariables[0])[0].trigsimp().simplify()
jh.showEquation(problem.ControlVariables[0], controlSolved)
jh.printMarkdown('We need to substitute this into the state\'s and costate\'s equations of motions.')
finalEquationsOfMotion = []
for x in problem.StateVariables :
    finalEquationsOfMotion.append(problem.EquationsOfMotion[x].subs(problem.ControlVariables[0], controlSolved))

jh.printMarkdown(r'Now we make our optimal $\dot{\lambda}$ differential equations.')
#hamlt = hamlt.subs(thetaSubs)
lambdaDotExpressions = IndirectHelpers.CreateLambdaDotCondition(problem, hamiltonian, lambdas).doit()
jh.printMarkdown(r'The $\dot{\lambda}$ equations are:')
for i in range(0, len(lambdas)) :
    jh.showEquation(sy.diff(lambdas[i], problem.TimeSymbol), lambdaDotExpressions[i])
    finalEquationsOfMotion.append(lambdaDotExpressions[i].subs(problem.ControlVariables[0], controlSolved))

initialVales = [0]*8
initialVales[0] = r0
initialVales[1] = u0
initialVales[2] = v0
initialVales[3] = lon0
jh.printMarkdown('With our equations of motion, now we need to find a solution to our 2 point boundary value problem.')
jh.printMarkdown(r'We need an guess for the initial values of the costates before we hand it off to a numerical solver.  For this, we need some intuition.  Starting with the last of the equations, we set $\lambda_{\theta}(0)=0$ as its derivative is 0.')
initialVales[7] = 0.0

jh.printMarkdown('To increase the radius of an orbit, we want most of the thrust to be adding to the circular velocity.  Inspecting the equation for $\dot{v}$, we can apply some intuition to get a reasonable first guess.')
jh.showEquation(r'\dot{v}', finalEquationsOfMotion[2], cleanEqu=False)#TODO: figure out why clean cleans too much...
jh.printMarkdown(r'For initial-guess purposes, we want to maximize $\dot{v}$, to the expense of everything else.  We want the cosine of the first term to be 1, so we want its argument to be 0.  To that end, we want $\lambda_v$ to be 1 and $\lambda_u$ to be near 0.  Not 0 exactly because that will cause numerical problems later.')
initialVales[5] = 0.0001
initialVales[6] = 1.0

jh.printMarkdown(r'Inspecting $\dot{u}$ and $\dot{\lambda_{u}}$, we want $\dot{\lambda_{u}}$ to be 0 so that $\dot{u}$ is minimized initially.  Putting in our values for the other 2 lambdas and initial values, and solving it for $\lambda_{r_0}$ gives us:')
lambdaUDot = finalEquationsOfMotion[5].subs(lambdas[1], initialVales[5]).subs(lambdas[2], initialVales[6]).subs(problem.StateVariables[0], initialVales[0]).subs(problem.StateVariables[1], initialVales[1]).subs(problem.StateVariables[2], initialVales[2])
lambdaR0Value = sy.solve(sy.Eq(lambdaUDot, 0), lambdas[0])[0]
jh.showEquation(r'\lambda_{r_{0}}', lambdaR0Value)
initialVales[4] = float(lambdaR0Value) # later on, arrays will care that this MUST be a float

xfBc = [None, 0.0, None, None]

jh.printMarkdown("Finally, we need to include the transversality condition.")
# hail mary try to solve the differential equations
# eqForDSolveAttempt = []
# for i in range(0, 4) :
#     eqForDSolveAttempt.append(sy.Derivative(problem.StateVariables[i], problem.TimeSymbol).doit()- finalEquationsOfMotion[i].simplify())

# for i in range(0, 4) :
#     eqForDSolveAttempt.append(sy.Derivative(lambdas[i], problem.TimeSymbol).doit()- finalEquationsOfMotion[i+4].simplify())

# ans = sy.dsolve(eqForDSolveAttempt)
# display(ans)
transversalityCondition = IndirectHelpers.YetAnotherAttemptAtTheTransversalityCondition(problem, sy.Symbol('H'), lambdas, 0.0)
#TODO: Throw if wrong number, expect 2
print(transversalityCondition)
for exp in transversalityCondition :
    jh.showEquation(0, exp)
constantsSubsDict = {}
problem.AppendConstantsToSubsDict(constantsSubsDict, mu, g, thrust, m0, isp)
eoms = []
for ep in finalEquationsOfMotion :
    eoms.append(ep.subs(constantsSubsDict))

stateForEom = [problem.TimeSymbol]
stateForEom.append(problem.StateVariables)
stateForEom.append(lambdas)

integratableEoms = sy.lambdify(stateForEom, eoms)
stateForEnd = []
for i in range(0, len(problem.StateVariables)) :
    stateForEnd.append(problem.StateVariables[i].subs(problem.TimeSymbol, problem.TimeFinal))
stateForEnd.extend(lambdas)
evaluatableTransversalityConditions = []
for excond in transversalityCondition:
    evalTransCond = sy.lambdify(stateForEnd, excond.subs(constantsSubsDict))
    evaluatableTransversalityConditions.append(evalTransCond)

from scipy.optimize import fsolve
from scipy.integrate import odeint
tfVal  = 3600*3.97152*24 
tArray = np.linspace(0, tfVal, 1200)

def integrateFromInitialValues(z0) :
    integratableCb = lambda z,t : integratableEoms(t, z[0:4], z[4:8])
    return odeint(integratableCb, z0, tArray)

def callbackForFsolve(lambdaGuesses) :
    z0 = []
    z0.extend(initialVales[0:4])
    z0.extend(lambdaGuesses)
    ans = integrateFromInitialValues(z0)
    finalVOffCircular = ans[:,2][-1] - math.sqrt(mu/ans[:,0][-1])

    finalState = [ans[:,0][-1],ans[:,1][-1],ans[:,2][-1],ans[:,3][-1],ans[:,4][-1],ans[:,5][-1],ans[:,6][-1],ans[:,7][-1]]
    
    finalAnswers = []
    finalAnswers.append(ans[:,1][-1])
    finalAnswers.append(finalVOffCircular)
    for transCondition in evaluatableTransversalityConditions: 
        finalAnswers.append(transCondition(*finalState))
    return finalAnswers

fSolveSol = fsolve(callbackForFsolve, initialVales[4:8], epsfcn=0.001, factor=0.1, full_output=True)
print(fSolveSol)
finalInitialValues = []
finalInitialValues.extend(initialVales[0:4])
finalInitialValues.extend(fSolveSol[0])
sol = integrateFromInitialValues(finalInitialValues)
asDict = OrderedDict()
asDict[problem.StateVariables[0]] = sol[:,0]
asDict[problem.StateVariables[1]] = sol[:,1]
asDict[problem.StateVariables[2]] = sol[:,2]
asDict[problem.StateVariables[3]] = sol[:,3]
asDict[lambdas[0]] = sol[:,4]
asDict[lambdas[1]] = sol[:,5]
asDict[lambdas[2]] = sol[:,6]
asDict[lambdas[3]] = sol[:,7]
problem.PlotSolution(tArray, asDict, "Test")

jh.showEquation(problem.StateVariables[0].subs(problem.TimeSymbol, problem.TimeFinal), sol[:,0][-1])
jh.showEquation(problem.StateVariables[1].subs(problem.TimeSymbol, problem.TimeFinal), sol[:,1][-1])
jh.showEquation(problem.StateVariables[2].subs(problem.TimeSymbol, problem.TimeFinal), sol[:,2][-1])
jh.showEquation(problem.StateVariables[3].subs(problem.TimeSymbol, problem.TimeFinal), sol[:,3][-1])

