import __init__ #type: ignore
import sympy as sy
#from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
from pyeq2orb.Numerical import ScipyCallbackCreators

import sympy as sy
from typing import Optional, List, Dict, Callable, cast, Any, Union, Tuple
from pyeq2orb.ProblemBase import Problem
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from pyeq2orb.Symbolics.SymbolicUtilities import SafeSubs


class LambdifyHelper :
    """As I've worked with more and more problems, trying to find the line of responsibility for the library to assist
    with lambdify'ing expressions and keeping things flexible for the script writer has been hard.  To that end, this 
    LambdifyHelper is grouping the lambdify arguments, the expressions to lambdify, and a substitution dictionary.

    This (and derived) types are not meant to be that complicated.  When working with systems of equations, you would 
    need to keep track of the order of expressions, have lists of symbols and expressions... these types are mostly 
    providing a consistent structure of that data without needing to learn the nuisances of this type in addition to 
    whatever solver you are using.  Keep this and derived types simple.  The goal isn't to do things for the 
    script writer, it is to make it easy for the script writer to do things.

    There are derived types to assist in more specialized lambdify'ing.
    """
    def __init__(self, lambdifyArguments : List[Union[sy.Symbol, List[sy.Symbol]]], expressionsToLambdify : List[sy.Expr], substitutionDictionary : Dict) :
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

        self._functionRedirectionArray={} #type: Dict[str, Callable]

        self._lambdifyArguments = lambdifyArguments
        self._expressionsToGetLambdified = expressionsToLambdify
        i=0
        for exp in self._expressionsToGetLambdified :
            if isinstance(exp, sy.Eq) :
                self._expressionsToGetLambdified[i] = exp.rhs #TODO: Throw here at some point...
            i=i+1
        self._substitutionDictionary = substitutionDictionary

    @property
    def LambdifyArguments(self) -> List[Union[sy.Symbol, List[sy.Symbol]]]:
        """The list of state variables that are used to lambdify an expression.  This list is never 
        None.  It is likely the lambdified expressions will care about the order of the values in 
        this list.

        Returns:
            List[Union[sy.Symbol, List[sy.Symbol]]]: The list of state variables, often sy.Symbol's or sy.Functions of the Time symbol.
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
    def FunctionRedirectionArray(self) -> Dict[str, Callable]:
        return self._functionRedirectionArray

    @property 
    def SubstitutionDictionary(self) -> Dict :
        """Constants that should be substituted into the expressions to lambdify before lambdify'ing them.
        This is never None, but it may be empty.

        Returns:
            Dict: Constants that should be substituted into the expressions to lambdify.
        """
        return self._substitutionDictionary

    def Lambdify(self):
        return sy.lambdify(self.LambdifyArguments, SafeSubs(self.ExpressionsToLambdify, self.SubstitutionDictionary))

    @staticmethod
    def CreateLambdifiedExpressions(stateExpressionList : List[sy.Expr], expressionsToLambdify : List[sy.Expr], constantsSubstitutionDictionary : Dict[sy.Expr, float], functionRedirectionArray : Optional[List[Callable]] = None) ->Callable[..., float] :
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
            bc = SafeSubs(exp, constantsSubstitutionDictionary)
            lambdifiedExpressions.append(bc)
        
        return sy.lambdify(stateExpressionList, lambdifiedExpressions, functionRedirectionArray)    
    
    def GetExpressionToLambdifyInMatrixForm(self) -> sy.Matrix:
        return sy.Matrix(self.ExpressionsToLambdify)
       
    def AddMoreExpressions(self, newSvs, newExpressions):
        for i in range(0, len(newSvs)) :
            self.LambdifyArguments.append(newSvs[i])
            self.ExpressionsToLambdify.append(newExpressions[i])

    def LambdifyArgumentsInMatrixForm(self) -> sy.Matrix :
        return sy.Matrix([item for sublist in self.LambdifyArguments for item in sublist]) #type: ignore

    def ApplySubstitutionDictionaryToExpressions(self):
        self._expressionsToGetLambdified = SafeSubs(self._expressionsToGetLambdified, self.SubstitutionDictionary)


    def CheckAllParametersArePresent(self) ->List[sy.Basic] :
        missingArgs = []
        for exper in self.ExpressionsToLambdify :
            allArgs =  [x for x in exper.atoms(sy.Function) if hasattr(x, 'name')]
            allArgs.extend(exper.atoms(sy.Symbol))
            for arg in allArgs:
                found =  arg in self.SubstitutionDictionary.keys() #or arg in self.OtherArguments
                for lmdArg in self.LambdifyArguments :
                    found = found or lmdArg == arg or (hasattr(lmdArg, "__len__")) and arg in cast(List[sy.Symbol], lmdArg)
                if not found and not arg in missingArgs:
                    missingArgs.append(arg)
        return missingArgs         




class OdeLambdifyHelper(LambdifyHelper):
    def __init__(self, time, stateVariables, firstOrderDynamicExpressions, otherArgsList : List[sy.Symbol], substitutionDictionary : Dict) :
        self._nonTimeStateVariables = stateVariables
        self._firstOrderStateDynamics = firstOrderDynamicExpressions
        self._time = time

        if otherArgsList == None :
            otherArgsList = []

        self._otherArgs = otherArgsList
        LambdifyHelper.__init__(self, [time, self._nonTimeStateVariables], self._firstOrderStateDynamics, substitutionDictionary)

    @property
    def Time(self) -> sy.Symbol:
        """Gets the time symbol.  This may be None if that makes sense for the expressions getting lambdified.

        Returns:
            sy.Expr: The time symbol.
        """
        return self._time
    
    @Time.setter
    def Time(self, timeValue : sy.Symbol) :
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
                thisEom = SafeSubs(thisEom, self.SubstitutionDictionary).doit(deep=True)  
            eomList.append(thisEom)   
        # for thisEom in equationsOfMotion :
        #     # eom's could be constant equations.  Check, add if it doesn't have subs
        #     if(hasattr(thisEom, "subs")) :
        #         thisEom = SafeSubs(thisEom, self.SubstitutionDictionary).doit(deep=True)  
        odeArgs = self.LambdifyArguments
        if odeArgs[0] != self.Time :   
            odeArgs = [self.Time, cast(Union[sy.Symbol, List[sy.Symbol]], List(odeArgs))]
        if self.OtherArguments != None and len(self.OtherArguments) >0 :
            odeArgs.append(self.OtherArguments)
        
        eomCallback = sy.lambdify(odeArgs, eomList, modules=['scipy'], cse=True)
        #TODO: This cant call lambdify directly, it must call base class

        # don't need the next wrapper if there are no other args
        if self.OtherArguments == None or len(self.OtherArguments) == 0 :            
            return eomCallback

        # but if there are other arguments, handle that
        def callbackFunc(t, y, *args) :
            return eomCallback(t, y,args)
        return callbackFunc        

    def CreateListOfStateVariableCallbacksTimeFirst(self) ->List[Callable]:
        # if odeState == None :
        #     odeState = self.CreateDefaultState()
        equationsOfMotion = self.ExpressionsToLambdify
        eomList = []
        odeArgs = self.LambdifyArguments
        if odeArgs[0] != self.Time :   
            odeArgs = [self.Time, cast(Union[sy.Symbol, List[sy.Symbol]], odeArgs)]
        if self.OtherArguments != None and len(self.OtherArguments) >0 :
            odeArgs.append(self.OtherArguments)        
        for thisEom in equationsOfMotion :
            # eom's could be constant equations.  Check, add if it doesn't have subs
            if(hasattr(thisEom, "subs")) :
                thisEom = SafeSubs(thisEom, self.SubstitutionDictionary).doit(deep=True)  
            #eomList.append(thisEom)   
        # for thisEom in equationsOfMotion :
        #     # eom's could be constant equations.  Check, add if it doesn't have subs
        #     if(hasattr(thisEom, "subs")) :
        #         thisEom = SafeSubs(thisEom, self.SubstitutionDictionary).doit(deep=True)  
            modules = ['scipy']
            if self.FunctionRedirectionArray != None and len(self.FunctionRedirectionArray) >0 : 
                moduels = self.FunctionRedirectionArray
            eomCallback = sy.lambdify(odeArgs, thisEom, modules=moduels, cse=True)
        # don't need the next wrapper if there are no other args
            if not (self.OtherArguments == None or len(self.OtherArguments) == 0) :            
                # if there are other arguments, handle that
                def callbackFunc(t, y, *args) :
                    return eomCallback(t, y,args)
                eomCallback = callbackFunc
            eomList.append(eomCallback)
        return eomList        


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
        def switchTimeOrderCallback2(y, t, args) :
            return originalCallback(t, y, args)
        return switchTimeOrderCallback2   
    
    @property
    def NonTimeLambdifyArguments(self) :
        return self.LambdifyArguments[1]

    def AddStateVariable(self, stateVariable : sy.Symbol, firstOrderStateVariableDynamic : sy.Expr):
        self.NonTimeLambdifyArguments.append(stateVariable)
        self.ExpressionsToLambdify.append(firstOrderStateVariableDynamic)
        self._firstOrderStateDynamics.append(firstOrderStateVariableDynamic)

    def AddStateVariables(self, stateVariables : List[sy.Symbol], stateVariableDynamics : List[sy.Expr]) :
        for i in range(0, len(stateVariables)) :
            self.AddStateVariable(stateVariables[i], stateVariableDynamics[i])

    def NonTimeArgumentsArgumentsInMatrixForm(self) -> sy.Matrix :
        return sy.Matrix(self.NonTimeLambdifyArguments)
    
    @property 
    def OtherArguments(self) -> List[sy.Symbol] :
        """If there are other arguments that need to be passed to the lambdified expression that are not 
        part of the state, those arguments are specified here.  This list is never None but may be empty.

        Returns:
            List[sy.Symbol]: The list of other arguments.
        """
        return self._otherArgs

    @property
    def EquationsOfMotion(self) -> List[sy.Expr] :
        return self._firstOrderStateDynamics


# the line between this and a problem is VERY fuzzy.  There is little keeping us from making the relevant code here 
# just be member items on Problem, BUT, I want to keep separate the responsibility between managing and helping 
# make the content of a Problem, and lambdifying it
class OdeLambdifyHelperWithBoundaryConditions(OdeLambdifyHelper):
    
    def __init__(self, time : sy.Symbol, t0: sy.Symbol, tf: sy.Symbol, stateVariables : List[sy.Symbol], dynamicExpressions : List[sy.Expr], boundaryConditionEquations : List[sy.Expr], otherArgsList : List[sy.Symbol], substitutionDictionary : Dict) :
        OdeLambdifyHelper.__init__(self, time, stateVariables, dynamicExpressions, otherArgsList, substitutionDictionary)
        self._t0 = t0 #type: sy.Symbol
        self._tf = tf #type: sy.Symbol
        self._boundaryConditions = boundaryConditionEquations
        self._symbolsToSolveForWithBcs = [] #type: List[sy.Symbol]

    @staticmethod
    def CreateFromProblem(problem : Problem) :
        stateAndControl = [*problem.StateVariables, *problem.ControlVariables]
        return OdeLambdifyHelperWithBoundaryConditions(problem.TimeSymbol, problem.TimeInitialSymbol, problem.TimeFinalSymbol, stateAndControl, problem.StateVariableDynamic, problem.BoundaryConditions, [], problem.SubstitutionDictionary)

    @property
    def t0(self) -> sy.Symbol :
        return self._t0
    
    @t0.setter
    def t0(self, value:sy.Symbol) :
        self._t0 = value

    @property
    def tf(self) -> sy.Symbol:
        return self._tf
    
    @tf.setter
    def tf(self, value:sy.Symbol) :
        self._tf = value

    @property
    def BoundaryConditionExpressions(self) -> List[sy.Expr] :
        return self._boundaryConditions # must equal 0
    
    @property
    def SymbolsToSolveForWithBoundaryConditions(self) -> List[sy.Symbol] :
        return self._symbolsToSolveForWithBcs # these need to be in terms of t0 or tf

    def CreateCallbackForBoundaryConditionsWithFullState(self) ->tuple[List[sy.Expr], Callable[..., float]]: 
        stateForBoundaryConditions = [] #type: List[sy.Expr]
        stateForBoundaryConditions.append(self.t0)
        stateForBoundaryConditions.extend(SafeSubs(self.NonTimeLambdifyArguments, {self.Time: self.t0}))
        stateForBoundaryConditions.append(self.tf)
        stateForBoundaryConditions.extend(SafeSubs(self.NonTimeLambdifyArguments, {self.Time: self.tf}))
        stateForBoundaryConditions.extend(self.OtherArguments) #even if things are repeated, that is ok
                
        #stateForBoundaryConditions.extend(fSolveOnlyParameters)

        boundaryConditionEvaluationCallbacks = LambdifyHelper.CreateLambdifiedExpressions(stateForBoundaryConditions, self.BoundaryConditionExpressions, self.SubstitutionDictionary)    
        return (stateForBoundaryConditions,boundaryConditionEvaluationCallbacks)
    
    
    def createCallbackToSolveForBoundaryConditions(self, solveIvpCallback, tArray, preSolveInitialGuessForIntegrator : List[float]) :
        (stateForBoundaryConditions,boundaryConditionEvaluationCallbacks) = self.CreateCallbackForBoundaryConditionsWithFullState()
        mapForIntegrator = [] #type: List[int]
        mapForBcs = [] #type: List[int]
        for i in range(0, len(self.SymbolsToSolveForWithBoundaryConditions)) :
            mapForBcs.append(stateForBoundaryConditions.index(self.SymbolsToSolveForWithBoundaryConditions[i]))
            try :
                indexForIntegrator = self.NonTimeLambdifyArguments.index(SafeSubs(self.SymbolsToSolveForWithBoundaryConditions[i], {self.t0: self.Time})) #TODO: Do I need to do TF?
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