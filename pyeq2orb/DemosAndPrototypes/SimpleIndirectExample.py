#%%
from pyeq2orb.ProblemBase import Problem, ProblemVariable
from pyeq2orb import SafeSubs
from pyeq2orb.NumericalOptimizerProblem import NumericalOptimizerProblemBase
import sympy as sy
from IPython.display import display
from scipyPaperPrinter import printMarkdown, showEquation
t = sy.Symbol('t', real=True, positive=True)
t0 = sy.Symbol('t_0', real=True, positive=True)
tf = sy.Symbol('t_f', real=True, positive=True)
x = sy.Function('x', real=True)
y = sy.Function('y', real=True)
vx = sy.Function('v_x', real=True)
vy = sy.Function('v_y', real=True)
F = sy.Symbol('F', real=True, positive=True)
m = sy.Function('m', real=True, positive=True)
g = sy.Symbol('g', real=True, positive=True)

alpha = sy.Function(r'\alpha', real=True)
rCb = sy.Symbol('r_{e}', real=True, positive=True)
rc = sy.Symbol('r_c', real=True, positive=True)
vxf = sy.Symbol('v_{x_f}', real=True, positive=True)
mDotS = sy.Symbol('\dot{m}', real=True)
m0 = sy.Symbol("m_0", real=True, positive=True)

xDot = vx(t)
yDot = vy(t)
vxDot = F*sy.cos(alpha(t))/m(t)
vyDot = F*sy.sin(alpha(t))/m(t) - g
mDot = t*mDotS

display(vxDot)

bc1 = y(tf)-rc+rCb#=0
bc2 = vxf-vx(tf)#=0
bc3 = vy(tf) #=0
bc4 = m0-m0


######## Standard problem setup
pr_t = t
pr_y = [x(t), y(t), vx(t), vy(t), m(t)] 
pr_eom = [xDot, yDot, vxDot, vyDot, mDot]
pr_initialGuesses = [0, 0, 0, 0, m0]
pr_bcs = [bc1, bc2, bc3, bc4]
pr_j = tf
pr_sense = -1 # minimize
pr_controls = [alpha(t)]
pr_subsDict = {vxf:1.6270, rCb:1740.0, rc:1852.000, g:0.00162, F:10000*g, mDotS:-0.001, m0:2000.0} #type: ignore

problem = Problem()
for (k,v) in pr_subsDict.items():
    problem.SubstitutionDictionary[k] = v

for i in range(0, len(pr_y)):
    problem.AddStateVariable(ProblemVariable(pr_y[i], pr_eom[i]))

problem.ControlVariables.append(pr_controls[0])

problem.BoundaryConditions.extend(pr_bcs)
problem.TimeFinalSymbol = tf
problem.TimeInitialSymbol = t0
problem.TimeSymbol = t

tau = sy.Symbol(r'\tau')
tau0 = sy.Symbol(r'\tau_0')
tauf = sy.Symbol(r'\tau_f')
problem = problem.ScaleTime(tau, tau0, tauf, tau*tf)


######## Indirect specific solving
lambdas = Problem.CreateCoVector(pr_y, None, tau)
hamiltonian = problem.CreateHamiltonian(lambdas)
showEquation("H", hamiltonian)
lambdasEom = Problem.CreateLambdaDotEquationsStatic(hamiltonian, tau, problem.StateVariables, lambdas)
for i in range(0, len(pr_y)):
    problem.AddCostateVariable(ProblemVariable(lambdas[i], lambdasEom[i]))

controlExpressions = problem.CreateControlExpressionsFromHamiltonian(hamiltonian, problem.ControlVariables)
i=0
for (k,v) in controlExpressions.items(): #type: ignore
    problem.SubstitutionDictionary[k] = v #type: ignore
    showEquation(pr_controls[i], v)
    i=i+1

i=0
for eom in problem.EquationsOfMotionAsEquations:
    display(eom)
    display(problem.CostateDynamicsEquations[i])
    i=i+1

for bc in problem.BoundaryConditions:
    display(sy.Eq(0, bc))

## pick xversality
lambdas_f = SafeSubs(lambdas, {tau:tauf})
lambdas_0 = SafeSubs(lambdas, {tau:tau0})
xversality = problem.TransversalityConditionInTheDifferentialForm(hamiltonian, sy.Symbol('dt_f'), lambdas_f)
for i in range(0, len(xversality)):
    problem.BoundaryConditions.append(xversality[i])

###### Indirect Problem is fully setup, time to start doing numerical things.....
#%%
import numpy as np
initialGuess = [0,0,0,0,2000,  0,-3000.0,0.001,3000.0,0]
tArray = np.linspace(0.0, 1.0, 400)

from pyeq2orb.Numerical.LambdifyHelpers import OdeLambdifyHelperWithBoundaryConditions

numerical = OdeLambdifyHelperWithBoundaryConditions.CreateFromProblem(problem)
numerical.SymbolsToSolveForWithBoundaryConditions.clear()
numerical.SymbolsToSolveForWithBoundaryConditions.extend(lambdas_0[:4])
numerical.SymbolsToSolveForWithBoundaryConditions.append(tf)
numerical.ApplySubstitutionDictionaryToExpressions()
ivpCallback = numerical.CreateSimpleCallbackForSolveIvp()
def solve_ivp_wrapper(t, y, args):
    anAns = solve_ivp(ivpCallback, [t[0], t[-1]], y, dense_output=True, args=args, method='LSODA')
    return anAns
bcCallback = numerical.createCallbackToSolveForBoundaryConditionsBetter(solve_ivp_wrapper, [lambdas_0[0], lambdas_0[1], lambdas_0[2], lambdas_0[3], tf], tArray, initialGuess, (480,))

from scipy.integrate import solve_ivp #type: ignore
from scipy.optimize import fsolve  #type: ignore

fSolveInitialGuess = initialGuess[5:-1]
fSolveInitialGuess.append(480)
fsolveAns = fsolve(bcCallback, fSolveInitialGuess, full_output=True,  factor=0.2,epsfcn=0.001 )
print(fsolveAns)
finalInitialState = initialGuess[:5]
finalInitialState.extend(fsolveAns[0][:5])
tfReal = fsolveAns[0][-1]
anAns = solve_ivp(ivpCallback, [0,1], finalInitialState, dense_output=True, args=(fsolveAns[0][-1],), method='LSODA')
print(anAns)

from pyeq2orb.Graphics.Plotly2DModule import plot2DLines
import pyeq2orb.Graphics.Primitives as prim

def plotAThing(title, label1, t1, dataset1):
    plot2DLines([prim.XAndYPlottableLineData(t1, dataset1, label1, '#0000ff', 2, 0)], title)

titles = ["x", "y", "vx", "vy", "m"]
i=0
for title in titles:
    plotAThing(titles[i], titles[i], anAns.t*tfReal, anAns.y[i])
    i=i+1
i=0

plotAThing("XY Path", "path", anAns.y[0], anAns.y[0])

print(fsolveAns)



