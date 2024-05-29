#%%
from pyeq2orb.ProblemBase import Problem, ProblemVariable
from pyeq2orb import SafeSubs
from pyeq2orb.NumericalOptimizerProblem import NumericalOptimizerProblemBase
import sympy as sy
from IPython.display import display
from scipyPaperPrinter import printMarkdown, showEquation
t = sy.Symbol('t')
t0 = sy.Symbol('t_0')
tf = sy.Symbol('t_f')
x = sy.Function('x')
y = sy.Function('y')
vx = sy.Function('v_x')
vy = sy.Function('v_y')
F = sy.Symbol('F')
m = sy.Function('m')
g = sy.Symbol('g')

alpha = sy.Function(r'\alpha')
rCb = sy.Symbol('r_{e}')
rc = sy.Symbol('r_c')
vxf = sy.Symbol('v_{x_f}')
mDotS = sy.Symbol('\dot{m}')
m0 = sy.Symbol("m_0")

xDot = vx(t)
yDot = vy(t)
vxDot = F*sy.cos(alpha(t))/m(t)
vyDot = F*sy.sin(alpha(t))/m(t) - g
mDot = t*mDotS

display(vxDot)

bc1 = y(tf)-rc+rCb#=0
bc2 = vxf-vx(tf)#=0
bc3 = vy(tf) #=0
bc4 = m(0)-m0


######## Standard problem setup
pr_t = t
pr_y = [x(t), y(t), vx(t), vy(t), m(t)] 
pr_eom = [xDot, yDot, vxDot, vyDot, mDot]
pr_initialGuesses = [0, 0, 0, 0, m0]
pr_bcs = [bc1, bc2, bc3, bc4]
pr_j = tf
pr_sense = -1 # minimize
pr_controls = [alpha(t)]
pr_subsDict = {vxf:1627.0, rc:185200.0, g:1.62, F:3*g, mDotS:-0.001, m0:2000.0} #type: ignore

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
lambdas = Problem.CreateBarVariables(pr_y, tau)
hamiltonian = problem.CreateHamiltonian(lambdas)
showEquation("H", hamiltonian)
lambdasEom = Problem.CreateLambdaDotEquationsStatic(hamiltonian, tau, problem.StateVariables, lambdas)
for i in range(0, len(pr_y)):
    problem.AddCostateVariable(ProblemVariable(lambdas[i], lambdasEom[i]))

controlExpressions = problem.CreateControlExpressionsFromHamiltonian(hamiltonian, problem.ControlVariables)

for (k,v) in controlExpressions.items(): #type: ignore
    problem.SubstitutionDictionary[k] = v #type: ignore

i=0
for eom in problem.EquationsOfMotionAsEquations:
    display(eom)
    display(problem.CostateDynamicsEquations[i])
    i=i+1

for bc in problem.BoundaryConditions:
    display(sy.Eq(0, bc))

## pick xversality
lambdas_f = SafeSubs(lambdas, {tau:tauf})
xversality = problem.TransversalityConditionInTheDifferentialForm(hamiltonian, sy.Symbol('dt_f'), lambdas_f)
for i in range(0, len(xversality)):
    problem.BoundaryConditions.append(xversality[i])

###### Indirect Problem is fully setup, time to start doing numerical things.....
import numpy as np
initialGuess = [0,0,0,0,2000,-0.001,0.001,0.001,0.001,0]
tArray = np.linspace(0.0, 1.0, 400)

from pyeq2orb.Numerical.LambdifyHelpers import OdeLambdifyHelperWithBoundaryConditions

numerical = OdeLambdifyHelperWithBoundaryConditions.CreateFromProblem(problem)
numerical.ApplySubstitutionDictionaryToExpressions()
ivpCallback = numerical.CreateSimpleCallbackForSolveIvp()
bcCallback = numerical.createCallbackToSolveForBoundaryConditions(ivpCallback, tArray, initialGuess)

from scipy.integrate import solve_ivp #type: ignore
from scipy.optimize import fsolve  #type: ignore
#%%
fSolveInitialGuess = initialGuess[5:]
fSolveInitialGuess.append(480)
fsolveAns = fsolve(bcCallback, fSolveInitialGuess, full_output=True )

anAns = solve_ivp(ivpCallback, [0,1], initialGuess, dense_output=True, args=(400,), method='LSODA')
print(anAns)

from pyeq2orb.Graphics.Plotly2DModule import plot2DLines
import pyeq2orb.Graphics.Primitives as prim

def plotAThing(title, label1, t1, dataset1):
    plot2DLines([prim.XAndYPlottableLineData(t1, dataset1, label1, '#0000ff', 2, 0)], title)

titles = ["x", "y", "vx", "vy", "m"]
i=0
for title in titles:
    plotAThing(titles[i], titles[i], anAns.t, anAns.y[i])
    i=i+1
i=0



