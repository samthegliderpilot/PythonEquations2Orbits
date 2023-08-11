#%%
import __init__ #type: ignore
from IPython.display import display
from scipy.integrate import solve_ivp #type: ignore
import matplotlib.pyplot as plt#type: ignore
import numpy as np
import sympy as sy
import plotly.express as px#type: ignore
from pandas import DataFrame #type: ignore
import math
from scipy.optimize import fsolve#type: ignore
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
from pyeq2orb.ScaledSymbolicProblem import ScaledSymbolicProblem
from pyeq2orb.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem
from pyeq2orb.Numerical import ScipyCallbackCreators


import sympy as sy
from typing import List, Dict, Callable, Optional, Any
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
from pyeq2orb.Utilities.Typing import SymbolOrNumber

class LambdifyHelper :
    """As I've worked with more and more problems, trying to find the line of responsibility for the library to assist
    with lambdify'ing expressions and keeping things flexible for the script writer has been hard.  To that end, this 
    LambdifyHelper is grouping the lambdify arguments, the expressions to lambdify, and a substitution dictionary.

    This (and derived) types are not meant to be that complicated.  When working with systems of equations, you would 
    need to keep track of the order of expressions, have lists of symbols and expressions... these types are mostly 
    providing a consistant structure of that data without needing to learn the nuiances of this type in addition to 
    whatever solver you are using.  Keep this and derived types simple.  The goal isn't to do things for the 
    script writer, it is to make it easy for the script writer to do things.

    There are derived types to assist in more specialized lambdifying.
    """
    def __init__(self, lambdifyArguments : List[sy.Expr], expressionsToLambdify : List[sy.Expr], substitutionDictionary : Dict) :
        """Initialize a new instance.

        Args:
            stateVariablesList (List[sy.Expr]): A list of the state variables.  If this is empty (and will be filled in later) note that the order may matter.
            expressionsToLambdify (List[sy.expr]): The expressions to lambdify.  Even if there is only 1 item, it should still be in a list.  The order of items 
            in this list may need to align with the order of stateVariablesList.
            otherArgsList (List[sy.Expr]): If there are other arguments that what ever uses the lambdified callback needs that shouldn't be part of the state, they are here.  This can be None.
            substitutionDictionary (Dict): If there are constants that need to be substituted into the expressions to lambdify, include them here.
        """
        if lambdifyArguments == None :
            lambdifyArguments = []
        if expressionsToLambdify == None :
            expressionsToLambdify = []
        if substitutionDictionary == None:
            substitutionDictionary = {}

        self._lambdifyArguments = lambdifyArguments
        self._expressionsToGetLambdified = expressionsToLambdify
        i=0
        for exp in self._expressionsToGetLambdified :
            if isinstance(exp, sy.Eq) :
                self._expressionsToGetLambdified[i] = exp.rhs
            i=i+1
        self._substitutionDictionary = substitutionDictionary

    @property
    def LambdifyArguments(self) -> List[sy.Expr]:
        """The list of state variables that are used to lambdify an expression.  This list is never 
        None.  It is likely the lambdified expressions will care about the order of the values in 
        this list.

        Returns:
            List[sy.Expr]: The list of state variables, often sy.Symbol's or sy.Functions of the Time symbol.
        """
        return self._lambdifyArguments

    @property
    def ExpressionsToLambdify(self) -> List[sy.Expr]:
        """The expressions that will get lambdified.  This list is never None, and the order of these 
        expressions may need to align to the order of the StateVariableListOrdered (for example, [ignoring 
        the time parameter] doing multi-variable numerical integration, the first equation of motion in this
        list must be the derivative of the first value in StateVariableListOrdered).

        Returns:
           List[sy.Expr]: The expressions to get lambdified.
        """
        return self._expressionsToGetLambdified

    @property 
    def SubstitutionDictionary(self) -> Dict :
        """Constants that should be substituted into the expressions to lambdify before lambdify'ing them.
        This is never None, but it may be empty.

        Returns:
            Dict: Constants that should be substituted into the expressions to lambdify.
        """
        return self._substitutionDictionary

    def Lambdify(self):
        return sy.lambdify(self.LambdifyArguments, SymbolicProblem.SafeSubs(self.ExpressionsToLambdify, self.SubstitutionDictionary))

    @staticmethod
    def CreateLambdifiedExpressions(stateExpressionList : List[sy.Expr], expressionsToLambdify : List[sy.Expr], constantsSubstitutionDictionary : Dict[sy.Expr, float]) ->sy.Expr :
        """ A helper function to create a lambdified callback of some expressions while also substituting in constant values into the expressions. One common problem that 
        might come up is if the constantsSubstitutionDictionary contains an independent variable of one of the symbols in the state (for example, if one of your state 
        variables is x(t) and you put a constant value of t into the constantsSubstitutionDictionary, this turns x(t) into x(2) and things get confusing quickly). Generally
        you shouldn't really want to do this and should re-structure your code to avoid this.

        Args:
            boundaryConditionState (List[sy.Expr]): The state (independent variables) for the returned lambdified expressions.
            expressionsToLambdify (List[sy.Expr]): The expressions to lambdify
            constantsSubstitutionDictionary (Dict[sy.Expr, float]): Constant values to bake into the expressionsToLambdify ahead of time

        Returns:
            _type_: A callback that numerically evaluates the expressionsToLambdify.
        """
        lambdifiedExpressions = []
        for exp in expressionsToLambdify :
            bc = SymbolicProblem.SafeSubs(exp, constantsSubstitutionDictionary)
            lambdifiedExpressions.append(bc)
        return sy.lambdify(stateExpressionList, lambdifiedExpressions)    
    
    def GetExpressionToLambdifyInMatrixForm(self) -> sy.Matrix:
        return sy.Matrix(self.ExpressionsToLambdify)
       
    def AddMoreExpressions(self, newSvs, newExpressions):
        for i in range(0, len(newSvs)) :
            self.LambdifyArguments.append(newSvs[i])
            self.ExpressionsToLambdify.append(newExpressions[i])

    def LambdifyArgumentsInMatrixForm(self) -> sy.Matrix :
        return sy.Matrix([item for sublist in self.LambdifyArguments for item in sublist]) #type: ignore

    def ApplySubstitutionDictionaryToExpressions(self):
        self._expressionsToGetLambdified = SymbolicProblem.SafeSubs(self._expressionsToGetLambdified, self.SubstitutionDictionary)

class OdeLambdifyHelper(LambdifyHelper):
    def __init__(self, time, equationsOfMotion, otherArgsList : List[sy.Expr], substitutionDictionary : Dict) :
        self._nonTimeStateVariables = [] 
        self._equationsOfMotion = equationsOfMotion
        self._time = time
        expressionsOfMotion = []
        for eom in self._equationsOfMotion :
            expressionsOfMotion.append(eom.rhs)
            self._nonTimeStateVariables.append(eom.lhs.expr)

        if otherArgsList == None :
            otherArgsList = []

        self._otherArgs = otherArgsList
        LambdifyHelper.__init__(self, [time, self._nonTimeStateVariables], expressionsOfMotion, substitutionDictionary)

    @property
    def Time(self) -> sy.Expr:
        """Gets the time symbol.  This may be None if that makes sense for the expressions getting lambdified.

        Returns:
            sy.Expr: The time symbol.
        """
        return self._time
    
    @Time.setter
    def Time(self, timeValue : sy.Expr) :
        """Sets the time symbol.  This may be None if that makes sense for the expressions getting lambdified.

        Args:
            timeValue (sy.Expr): The time symbol to set.
        """
        self._time = timeValue


    def CreateSimpleCallbackForSolveIvp(self) -> Callable : 
        """Creates a lambdified expression of the equations of motion in ExpressionsToLambdify.

        Args:
            ExpressionsToLambdify. Defaults to None which will use the CreateDefaultState function.

        Returns:
            Callable: A callback to use in scipy.ode.solveivp functions (and odeint if you put time first).
        """
        # if odeState == None :
        #     odeState = self.CreateDefaultState()
        equationsOfMotion = self.ExpressionsToLambdify
        eomList = []
        for thisEom in equationsOfMotion :
            # eom's could be constant equations.  Check, add if it doesn't have subs
            if(hasattr(thisEom, "subs")) :
                thisEom = thisEom.subs(self.SubstitutionDictionary) 
            eomList.append(thisEom)   
        odeArgs = self.LambdifyArguments
        if odeArgs[0] != self.Time :   
            odeArgs = [self.Time, odeArgs]
        eomCallback = sy.lambdify(odeArgs, eomList, modules=['scipy'])

        # don't need the next wrapper if there are no other args
        if self.OtherArguments == None or len(self.OtherArguments) == 0 :            
            return eomCallback

        # but if there are other arguments, handle that
        def callbackFunc(t, y, *args) :
            return eomCallback(t, y, args)
        return callbackFunc        

    def CreateSimpleCallbackForOdeint(self) -> Callable : 
        """Creates a lambdified expression of the (assumed) equations of motion in ExpressionsToLambdify.

        Args:
            ExpressionsToLambdify. Defaults to None which will use the CreateDefaultState function.

        Returns:
            Callable: A callback to use in scipy.integrate.odeint functions where time comes after y.
        """        
        originalCallback = self.CreateSimpleCallbackForSolveIvp()
        # don't need the next wrapper if there are no other args
        if self.OtherArguments == None or len(self.OtherArguments) == 0 :            
            def switchTimeOrderCallback(y, t) :
                return originalCallback(t, y)
            return switchTimeOrderCallback
        #else...        
        def switchTimeOrderCallback2(y, t, *args) :
            return originalCallback(t, y, *args)
        return switchTimeOrderCallback2   
    
    @property
    def NonTimeLambdifyArguments(self) :
        return self.LambdifyArguments[1]

    def AddMoreEquationsOfMotion(self, newEoms : List[sy.Eq]):
        for i in range(0, len(newEoms)) :
            self.NonTimeLambdifyArguments.append(newEoms[i].lhs.expr)
            self.ExpressionsToLambdify.append(newEoms[i].rhs)
            self._equationsOfMotion.append(newEoms[i])

    def NonTimeArgumentsArgumentsInMatrixForm(self) -> sy.Matrix :
        return sy.Matrix(self.NonTimeLambdifyArguments)
    
    @property 
    def OtherArguments(self) -> List[sy.Expr] :
        """If there are other arguments that need to be passed to the lambdified expression that are not 
        part of the state, those arguments are specified here.  This list is never None but may be empty.

        Returns:
            List[sy.Expr]: The list of other arguments.
        """
        return self._otherArgs

    @property
    def EquationsOfMotion(self) -> List[sy.Eq] :
        return self._equationsOfMotion


class OdeLambdifyHelperWithBoundaryConditions(OdeLambdifyHelper):
    def __init__(self, time : sy.Symbol, t0: sy.Symbol, tf: sy.Symbol, equationsOfMotion : List[sy.Eq], boundaryConditionEquations : List[sy.Eq], otherArgsList : List[sy.Expr], substitutionDictionary : Dict) :
        OdeLambdifyHelper.__init__(self, time, equationsOfMotion, otherArgsList, substitutionDictionary)
        self._t0 = t0
        self._tf = tf
        self._boundaryConditions = boundaryConditionEquations
        self._symbolsToSolveForWithBcs = []

    @property
    def t0(self) :
        return self._t0
    
    @t0.setter
    def setT0(self, value) :
        self._t0 = value

    @property
    def tf(self) :
        return self._tf
    
    @tf.setter
    def setTf(self, value) :
        self._tf = value

    @property
    def BoundaryConditionExpressions(self) -> List[sy.Expr] :
        return self._boundaryConditions # must equal 0
    
    @property
    def SymbolsToSolveForWithBoundaryConditions(self) -> List[sy.Symbol] :
        return self._symbolsToSolveForWithBcs # these need to be in terms of t0 or tf

    def CreateCallbackForBoundaryConditionsWithFullState(self) :
        stateForBoundaryConditions = [] #type: List[sy.Expr]
        stateForBoundaryConditions.append(self.t0)
        stateForBoundaryConditions.extend(SymbolicProblem.SafeSubs(self.NonTimeLambdifyArguments, {self.Time: self.t0}))
        stateForBoundaryConditions.append(self.tf)
        stateForBoundaryConditions.extend(SymbolicProblem.SafeSubs(self.NonTimeLambdifyArguments, {self.Time: self.tf}))
        stateForBoundaryConditions.extend(self.OtherArguments) #even if things are repeated, that is ok
                
        #stateForBoundaryConditions.extend(fSolveOnlyParameters)

        boundaryConditionEvaluationCallbacks = LambdifyHelper.CreateLambdifiedExpressions(stateForBoundaryConditions, self.BoundaryConditionExpressions, self.SubstitutionDictionary)    
        return [stateForBoundaryConditions,boundaryConditionEvaluationCallbacks]
    
    
    def createCallbackToSolveForBoundaryConditions(self, solveIvpCallback, tArray, preSolveInitialGuessForIntegrator : List[float]) :
        [stateForBoundaryConditions,boundaryConditionEvaluationCallbacks] = self.CreateCallbackForBoundaryConditionsWithFullState()
        mapForIntegrator = [] #type: List[int]
        mapForBcs = [] #type: List[int]
        for i in range(0, len(self.SymbolsToSolveForWithBoundaryConditions)) :
            mapForBcs.append(stateForBoundaryConditions.index(self.SymbolsToSolveForWithBoundaryConditions[i]))
            try :
                indexForIntegrator = self.NonTimeLambdifyArguments.index(SymbolicProblem.SafeSubs(self.SymbolsToSolveForWithBoundaryConditions[i], {self.t0: self.Time})) #TODO: Do I need to do TF?
                mapForIntegrator.append(indexForIntegrator)
            except ValueError:
                pass
              
        
        preSolveInitialGuessForIntegrator = preSolveInitialGuessForIntegrator.copy()

        def callbackForFsolve(bcSolverState) :
            for j in range(0, len(mapForIntegrator)) :
                preSolveInitialGuessForIntegrator[mapForIntegrator[j]] = bcSolverState[j]
            
            ans = solveIvpCallback(preSolveInitialGuessForIntegrator)
            finalState = []
            finalState.append(tArray[0])
            finalState.extend(ScipyCallbackCreators.GetInitialStateFromIntegratorResults(ans))
            finalState.append(tArray[-1])
            finalState.extend(ScipyCallbackCreators.GetFinalStateFromIntegratorResults(ans))
            # for j in range(0, len(mapForBcs)) :
            #     finalState[mapForBcs[j]] = bcSolverState[j]
            
            finalAnswers = []
            finalAnswers.extend(boundaryConditionEvaluationCallbacks(*finalState))
            print(finalAnswers)
            return finalAnswers    
        return callbackForFsolve

def plotSolution(helper, solution):

    xyz = np.zeros((len(tArray), 3))
    for i in range(0, len(solution[helper.NonTimeLambdifyArguments[0]])) :
        r = solution[helper.NonTimeLambdifyArguments[0]][i]
        theta = solution[helper.NonTimeLambdifyArguments[3]][i]
        x = r*math.cos(theta)
        y = r*math.sin(theta)
        xyz[i,0] = x
        xyz[i,1] = y
        xyz[i,2] = 0


    df = DataFrame(xyz)

    xf = np.array(xyz[:,0])
    yf = np.array(xyz[:,1])
    zf = np.array(xyz[:,2])
    df = DataFrame({"x": xf, "y":yf, "z":zf})
    fig = px.line_3d(df, x="x", y="y", z="z")
    fig.show()

from pyeq2orb.Utilities.SolutionDictionaryFunctions import GetValueFromStateDictionaryAtIndex
import scipyPaperPrinter as jh#type: ignore
from typing import cast
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
tArray = np.linspace(0.0, tfOrg, 1200)
#if scaleTime:
#    tfVal = 1.0
#    tArray = np.linspace(0.0, 1.0, 1200)
    

# your choice of the nu vector here controls which transversality condition we use
nus = [sy.Symbol('B_{u_f}'), sy.Symbol('B_{v_f}')]
#nus = []

mus = sy.Symbol(r'\mu', real=True, positive=True)
thrusts = sy.Symbol('T', real=True, positive=True)
m0s = sy.Symbol('m_0', real=True, positive=True)

gs = sy.Symbol('g', real=True, positive=True)

ispS = sy.Symbol('I_{sp}', real=True, positive=True)
ts = sy.Symbol('t', real=True)
t0s = sy.Symbol('t_0', real=True)
tfs = sy.Symbol('t_f', real=True, positive=True)

rs = sy.Function('r', real=True, positive=True)(ts)
us = sy.Function('u', real=True, nonnegative=True)(ts)
vs = sy.Function('v', real=True, nonnegative=True)(ts)
lonS =  sy.Function(r'\theta', real=True, nonnegative=True)(ts)

alps = sy.Function(r'\alpha', real=True)(ts)

bc1 = us.subs(ts, tfs)
bc2 = vs.subs(ts, tfs)-sy.sqrt(mu/rs.subs(ts, tfs))

terminalCost = rs.subs(ts, tfs) # maximization problem



mFlowRate = -1*thrusts/(ispS*gs)
mEq = m0s+ts*mFlowRate

rEom = us
uEom = vs*vs/rs - mus/(rs*rs) + thrusts*sy.sin(alps)/mEq
vEom = -vs*us/rs + thrusts*sy.cos(alps)/mEq
lonEom = vs/rs


rEquation = sy.Eq(rs.diff(ts), rEom)
uEquation = sy.Eq(us.diff(ts), uEom)
vEquation = sy.Eq(vs.diff(ts), vEom)
lonEquation = sy.Eq(lonS.diff(ts), lonEom)



helper = OdeLambdifyHelperWithBoundaryConditions(ts, t0s, tfs, [rEquation, uEquation, vEquation, lonEquation], [bc1, bc2], [], {gs:g, ispS:isp, m0s: m0, thrusts:thrust, mus:mu})

lmds = SymbolicProblem.CreateCoVector(helper.NonTimeLambdifyArguments, None, ts)
hamlt = SymbolicProblem.CreateHamiltonianStatic(helper.NonTimeLambdifyArguments, ts, helper.GetExpressionToLambdifyInMatrixForm(), 0, lmds)
lambdaEoms = SymbolicProblem.CreateLambdaDotEquationsStatic(hamlt, ts, helper.NonTimeArgumentsArgumentsInMatrixForm(), lmds)
helper.AddMoreEquationsOfMotion(lambdaEoms)
dHdu = SymbolicProblem.CreateHamiltonianControlExpressionsStatic(hamlt, alps)
controlSolved = sy.solve(dHdu, alps)[0] 
helper.SubstitutionDictionary[alps] =  controlSolved

#TODO: Get the transversality conditions more generally
bc3 = 1-lmds[0].subs(ts, tfs)+0.5*lmds[2].subs(ts, tfs)*sy.sqrt(1/(rs.subs(ts, tfs)**3))
bc4 = lmds[3].subs(ts, tfs)
newBcs = [bc3, bc4]
helper.BoundaryConditionExpressions.extend(newBcs)
helper.SymbolsToSolveForWithBoundaryConditions.extend(SymbolicProblem.SafeSubs(lmds, {ts: t0s}))

del helper.BoundaryConditionExpressions[-1]
del helper.NonTimeLambdifyArguments[-1]
del helper.EquationsOfMotion[-1]
del helper.ExpressionsToLambdify[-1]
del helper.SymbolsToSolveForWithBoundaryConditions[-1]
helper.SubstitutionDictionary[lmds[3]] =0
helper.SubstitutionDictionary[lmds[3].subs(ts, tfs)]=0
helper.SubstitutionDictionary[lmds[3].subs(ts, t0s)]=0


ipvCallback = helper.CreateSimpleCallbackForSolveIvp()
def realIpvCallback(initialState) :
    solution = solve_ivp(ipvCallback, [tArray[0], tArray[-1]], initialState, t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
    solutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(helper.NonTimeLambdifyArguments, solution)
    return solution





initialaFSolveState = [0.0011569091762708, 0.00010000000130634948, 1.0]

justBcCb = helper.CreateCallbackForBoundaryConditionsWithFullState()
display(justBcCb[1](0, 6000, 7000, 8000, 0, *initialaFSolveState, 5000000, 42164000, -2000, -3000, 17, 10000, 20000, 30000))

solverCb = helper.createCallbackToSolveForBoundaryConditions(realIpvCallback, tArray, [r0, u0, v0, lon0, *initialaFSolveState])

#display(solverCb([1.0, 0.001, 0.001, 0.0]))
#display(helper.GetExpressionToLambdifyInMatrixForm())
#print(ipvCallback(0, [r0, u0, v0, lon0, 1.0, 0.001, 0.001, 0.0]))
solution = solve_ivp(ipvCallback, [tArray[0], tArray[-1]], [r0, u0, v0, lon0, *initialaFSolveState], t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
solutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(helper.NonTimeLambdifyArguments, solution)
plotSolution(helper, solutionDictionary)

fSolveSol = fsolve(solverCb, initialaFSolveState, epsfcn=0.000001, full_output=True)
display(fSolveSol)
fSolveSolution = solve_ivp(ipvCallback, [tArray[0], tArray[-1]], [r0, u0, v0, lon0, *fSolveSol[0]], t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
fSolveSolutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(helper.NonTimeLambdifyArguments, fSolveSolution)


plotSolution(helper, fSolveSolutionDictionary)

#%%
# these are options to switch to try different things
scaleElements = True
scaleTime = scaleElements and True

baseProblem = ContinuousThrustCircularOrbitTransferProblem()
initialStateValues = baseProblem.CreateVariablesAtTime0(baseProblem.StateVariables)
problem = cast(SymbolicProblem, baseProblem)

if scaleElements :
    newSvs = ScaledSymbolicProblem.CreateBarVariables(problem.StateVariables, problem.TimeSymbol) 
    problem = ScaledSymbolicProblem(baseProblem, newSvs, {problem.StateVariables[0]: initialStateValues[0], 
                                                          problem.StateVariables[1]: initialStateValues[2], 
                                                          problem.StateVariables[2]: initialStateValues[2], 
                                                          problem.StateVariables[3]: 1} , scaleTime) # note the int here for the scaling, not a float
stateAtTf = SymbolicProblem.SafeSubs(problem.StateVariables, {problem.TimeSymbol: problem.TimeFinalSymbol})
# make the time array
tArray = np.linspace(0.0, tfOrg, 1200)
if scaleTime:
    tfVal = 1.0
    tArray = np.linspace(0.0, 1.0, 1200)
jh.t = problem._timeSymbol # needed for cleaner printed equations

# register constants
constantsSubsDict = problem.SubstitutionDictionary
constantsSubsDict[baseProblem.Isp] = isp
constantsSubsDict[baseProblem.MassInitial] = m0
constantsSubsDict[baseProblem.Gravity] = g
constantsSubsDict[baseProblem.Mu]= mu
constantsSubsDict[baseProblem.Thrust] = thrust

# register initial state values
constantsSubsDict.update(zip(initialStateValues, [r0, u0, v0, lon0]))
if scaleElements :
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
    
# this next block does most of the problem, pretty standard optimal control actions
problem.Lambdas.extend(problem.CreateCoVector(problem.StateVariables, r'\lambda', problem.TimeSymbol))
lambdas = problem.Lambdas
hamiltonian = problem.CreateHamiltonian(lambdas)
lambdaDotExpressions = problem.CreateLambdaDotCondition(hamiltonian)
dHdu = problem.CreateHamiltonianControlExpressions(hamiltonian)[0]
controlSolved = sy.solve(dHdu, problem.ControlVariables[0])[0] # something that may be different for other problems is when there are multiple control variables

# you are in control of the order of integration variables and what equations of motion get evaluated, start updating the problem
# NOTE that this call adds the lambdas to the integration state
problem.EquationsOfMotion.update(zip(lambdas, lambdaDotExpressions))
SymbolicProblem.SafeSubs(problem.EquationsOfMotion, {problem.ControlVariables[0]: controlSolved})
# the trig simplification needs the deep=True for this problem to make the equations even cleaner
for (key, value) in problem.EquationsOfMotion.items() :
    problem.EquationsOfMotion[key] = value.trigsimp(deep=True).simplify() # some simplification to make numerical code more stable later, and that is why this code forces us to do things somewhat manually.  There are often special things like this that we ought to do that you can't really automate.

## Start with the boundary conditions
if scaleTime : # add BC if we are working with the final time (kind of silly for this example, but we need an equal number of in's and out's for fsolve later)
    problem.BoundaryConditions.append(baseProblem.TimeFinalSymbol-tfOrg)

# make the transversality conditions
if len(nus) != 0:
    transversalityCondition = problem.TransversalityConditionsByAugmentation(nus)
else:
    transversalityCondition = problem.TransversalityConditionInTheDifferentialForm(hamiltonian, sy.Symbol(r'dt_f'))
# and add them to the problem
problem.BoundaryConditions.extend(transversalityCondition)

initialFSolveStateGuess = ContinuousThrustCircularOrbitTransferProblem.CreateInitialLambdaGuessForLeoToGeo(problem, controlSolved)

# lambda_lon is always 0, so do that cleanup
del problem.EquationsOfMotion[lambdas[3]]
problem.BoundaryConditions.remove(transversalityCondition[-1])
lmdTheta = lambdas.pop()
problem.IntegrationSymbols.pop()
constantsSubsDict[lmdTheta]=0
constantsSubsDict[lmdTheta.subs(problem.TimeSymbol, problem.TimeFinalSymbol)]=0
constantsSubsDict[lmdTheta.subs(problem.TimeSymbol, problem.TimeInitialSymbol)]=0

# start the conversion to a numerical problem
if scaleTime :
    initialFSolveStateGuess.append(tfOrg)

otherArgs = []
if scaleTime :
    otherArgs.append(baseProblem.TimeFinalSymbol)
if len(nus) > 0 :
    otherArgs.extend(nus)
stateAndLambdas = []
stateAndLambdas.extend(problem.StateVariables)
stateAndLambdas.extend(lambdas)
odeState = [problem.TimeSymbol, stateAndLambdas, otherArgs]

def safeSubs(exprs, toBeSubbed):
    tbr = []
    for eom in exprs :
        if hasattr(eom, "subs"):
            tbr.append(eom.subs(toBeSubbed))
        else :
            tbr.append(eom)    
    return tbr

class OdeHelper :
    lambdifyStateFlattenOption = "flatten"
    lambdifyStateGroupedAllOption = "group"
    lambdifyStateGroupedAllButParametersOption = "groupFlattenParameters"

    lambdifyStateOrderOptionTimeFirst = "Time,StateVariables,MissingInitialValues,Parameters"
    lambdifyStateOrderOptionTimeMiddle = "StateVariables,Time,MissingInitialValues,Parameters"
    def __init__(self, t) :
        self.equationsOfMotion = []
        self.initialSymbols = []
        self.stateFunctionSymbols = []
        self.t = t
        self.constants = {}
        self.lambdifyParameterSymbols = []

    def setStateElement(self, sympyFunctionSymbol, symbolicEom, initialSymbol) :
        self.equationsOfMotion.append(symbolicEom)
        self.stateFunctionSymbols.append(sympyFunctionSymbol)
        self.initialSymbols.append(initialSymbol)

    def makeStateForLambdifiedFunction(self, groupOrFlatten=lambdifyStateGroupedAllButParametersOption, orderOption=lambdifyStateOrderOptionTimeFirst):
        arrayForLmd = []
        if orderOption == OdeHelper.lambdifyStateOrderOptionTimeFirst :
            arrayForLmd.append(self.t)
        stateArray = []    
        for svf in self.stateFunctionSymbols :
            stateArray.append(svf)
        if groupOrFlatten != OdeHelper.lambdifyStateFlattenOption :
            arrayForLmd.append(stateArray)    
        else :
            arrayForLmd.extend(stateArray)
        if orderOption == OdeHelper.lambdifyStateOrderOptionTimeMiddle :
            arrayForLmd.append(self.t)

        if len(self.lambdifyParameterSymbols) != 0 :
            if groupOrFlatten == OdeHelper.lambdifyStateGroupedAllButParametersOption or groupOrFlatten == OdeHelper.lambdifyStateFlattenOption:
                arrayForLmd.extend(self.lambdifyParameterSymbols)
            elif groupOrFlatten == OdeHelper.lambdifyStateGroupedAllOption :
                arrayForLmd.append(self.lambdifyParameterSymbols)
        return arrayForLmd

    def _createParameterOptionalWrapperOfLambdifyCallback(self, baseLambdifyCallback) :
        def callbackWrapper(a, b, *args) :
            if len(self.lambdifyParameterSymbols) == 0 :
                return baseLambdifyCallback(a, b)
            else :
                return baseLambdifyCallback(a, b, *args)
        return callbackWrapper

    def createLambdifiedCallback(self, groupOrFlatten=lambdifyStateGroupedAllButParametersOption, orderOption=lambdifyStateOrderOptionTimeFirst) :
        arrayForLmd=self.makeStateForLambdifiedFunction(groupOrFlatten, orderOption)
        subbedEom = safeSubs(self.equationsOfMotion, self.constants)
        baseLambdifyCallback = sy.lambdify(arrayForLmd, subbedEom, 'numpy')
        return self._createParameterOptionalWrapperOfLambdifyCallback(baseLambdifyCallback)

thisOdeHelper = OdeHelper(problem.TimeSymbol)
for key, value in problem.EquationsOfMotion.items() :
    thisOdeHelper.setStateElement(key, value, key.subs(problem.TimeSymbol, problem.TimeInitialSymbol) )

if scaleTime:
    thisOdeHelper.lambdifyParameterSymbols.append(baseProblem.TimeFinalSymbol)

if len(nus) != 0:
    thisOdeHelper.lambdifyParameterSymbols.append(problem.CostateSymbols[1])
    thisOdeHelper.lambdifyParameterSymbols.append(problem.CostateSymbols[2])

thisOdeHelper.constants = problem.SubstitutionDictionary
display(thisOdeHelper.makeStateForLambdifiedFunction())
odeIntEomCallback = thisOdeHelper.createLambdifiedCallback()

if len(nus) > 0 :
    # run a test solution to get a better guess for the final nu values, this is a good technique, but 
    # it is still a custom-to-this-problem piece of code because it is still initial-guess work
    initialFSolveStateGuess.append(initialFSolveStateGuess[1])
    initialFSolveStateGuess.append(initialFSolveStateGuess[2])  
    argsForOde = []
    if scaleTime :
        argsForOde.append(tfOrg)
    argsForOde.append(initialFSolveStateGuess[1])
    argsForOde.append(initialFSolveStateGuess[2])  
    print("solving ivp for final adjoined variable guess")
    testSolution = solve_ivp(odeIntEomCallback, [tArray[0], tArray[-1]], [r0, u0, v0, lon0, *initialFSolveStateGuess[0:3]], args=tuple(argsForOde), t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)  
    #testSolution = odeint(odeIntEomCallback, [r0, u0, v0, lon0, *initialFSolveStateGuess[0:3]], tArray, args=tuple(argsForOde))
    finalValues = ScipyCallbackCreators.GetFinalStateFromIntegratorResults(testSolution)
    initialFSolveStateGuess[-2] = finalValues[5]
    initialFSolveStateGuess[-1] = finalValues[6]

print(initialFSolveStateGuess)


stateForBoundaryConditions = []
stateForBoundaryConditions.extend(SymbolicProblem.SafeSubs(problem.IntegrationSymbols, {problem.TimeSymbol: problem.TimeInitialSymbol}))
stateForBoundaryConditions.extend(SymbolicProblem.SafeSubs(problem.IntegrationSymbols, {problem.TimeSymbol: problem.TimeFinalSymbol}))
stateForBoundaryConditions.extend(lambdas)
stateForBoundaryConditions.extend(otherArgs)

fSolveCallback = ContinuousThrustCircularOrbitTransferProblem.createSolveIvpSingleShootingCallbackForFSolve(problem, problem.IntegrationSymbols, [r0, u0, v0, lon0], tArray, odeIntEomCallback, problem.BoundaryConditions, SymbolicProblem.SafeSubs(lambdas, {problem.TimeSymbol: problem.TimeInitialSymbol}), otherArgs)
fSolveSol = fsolve(fSolveCallback, initialFSolveStateGuess, epsfcn=0.000001, full_output=True) # just to speed things up and see how the initial one works
print(fSolveSol)

# final run with answer
solution = solve_ivp(odeIntEomCallback, [tArray[0], tArray[-1]], [r0, u0, v0, lon0, *fSolveSol[0][0:3]], args=tuple(fSolveSol[0][3:len(fSolveSol[0])]), t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
#solution = odeint(odeIntEomCallback, [r0, u0, v0, lon0, *fSolveSol[0][0:3]], tArray, args=tuple(fSolveSol[0][3:len(fSolveSol[0])]))
#solution = odeint(odeIntEomCallback, [r0, u0, v0, lon0, 26.0, 1.0, 27.0], tArray, args=(tfOrg,))
solutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(problem.IntegrationSymbols, solution)
unscaledResults = solutionDictionary
unscaledTArray = tArray
unscaledResults = problem.DescaleResults(solutionDictionary)
if scaleTime:
    unscaledTArray=tfOrg*tArray

if scaleElements:    
    finalState = GetValueFromStateDictionaryAtIndex(solutionDictionary, -1)
    jh.showEquation(stateAtTf[0], finalState[problem.StateVariables[0]])
    jh.showEquation(stateAtTf[1], finalState[problem.StateVariables[1]])
    jh.showEquation(stateAtTf[2], finalState[problem.StateVariables[2]])
    jh.showEquation(stateAtTf[3], (finalState[problem.StateVariables[3]]%(2*math.pi))*180.0/(2*math.pi))

baseProblem.PlotSolution(tArray*tfOrg, unscaledResults, "Test")
jh.showEquation(baseProblem.StateVariables[0].subs(problem.TimeSymbol, problem.TimeFinalSymbol), unscaledResults[baseProblem.StateVariables[0]][-1])
jh.showEquation(baseProblem.StateVariables[1].subs(problem.TimeSymbol, problem.TimeFinalSymbol), unscaledResults[baseProblem.StateVariables[1]][-1])
jh.showEquation(baseProblem.StateVariables[2].subs(problem.TimeSymbol, problem.TimeFinalSymbol), unscaledResults[baseProblem.StateVariables[2]][-1])
jh.showEquation(baseProblem.StateVariables[3].subs(problem.TimeSymbol, problem.TimeFinalSymbol), (unscaledResults[baseProblem.StateVariables[3]][-1]%(2*math.pi))*180.0/(2*math.pi))

[hamiltonVals, dhduValues, d2hdu2Values] = problem.EvaluateHamiltonianAndItsFirstTwoDerivatives(solutionDictionary, tArray, hamiltonian, {problem.ControlVariables[0]: controlSolved}, {baseProblem.TimeFinalSymbol: tfOrg})
plt.title("Hamiltonian and its derivatives")
plt.plot(tArray/86400, hamiltonVals, label="Hamiltonian")
plt.plot(tArray/86400, dhduValues, label=r'$\frac{dH}{du}$')
plt.plot(tArray/86400, d2hdu2Values, label=r'$\frac{d^2H}{du^2}$')

plt.tight_layout()
plt.grid(alpha=0.5)
plt.legend(framealpha=1, shadow=True)
plt.show()   

xyz = np.zeros((len(tArray), 3))
for i in range(0, len(unscaledResults[baseProblem.StateVariables[0]])) :
    r = unscaledResults[baseProblem.StateVariables[0]][i]
    theta = unscaledResults[baseProblem.StateVariables[3]][i]
    x = r*math.cos(theta)
    y = r*math.sin(theta)
    xyz[i,0] = x
    xyz[i,1] = y
    xyz[i,2] = 0


df = DataFrame(xyz)

xf = np.array(xyz[:,0])
yf = np.array(xyz[:,1])
zf = np.array(xyz[:,2])
df = DataFrame({"x": xf, "y":yf, "z":zf})
fig = px.line_3d(df, x="x", y="y", z="z")
fig.show()
