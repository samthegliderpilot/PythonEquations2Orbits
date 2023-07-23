import sympy as sy
from typing import List, Dict, Callable, Optional, Any
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
# NOTE, that if EVER the source for the overall project is not available, the source 
# for this MUST be published.

class LambdifyHelper :
    """Creating a lambdified callback from a sympy expression requires coordination between 
    the state and the thing getting lambdified.  I would prefer not to require problems 
    to maintain that order, so this helper type is a simple container to handle that coupling.
    """
    def __init__(self, time : sy.Expr, stateVariablesList : List[sy.Expr], expressionsToLambdify : List[sy.Expr], otherArgsList : List[sy.Expr], substitutionDictionary : Dict) :
        """Initialize a new instance.

        Args:
            time (sy.Expr): The time symbol.  This may be None.
            stateVariablesList (List[sy.Expr]): A list of the state variables.  If this is empty (and will be filled in later) note that the order may matter.
            expressionsToLambdify (List[sy.expr]): The expressions to lambdify.  Even if there is only 1 item, it should still be in a list.  The order of items 
            in this list may need to align with the order of stateVariablesList.
            otherArgsList (List[sy.Expr]): If there are other arguments that what ever uses the lambdified callback needs that shouldn't be part of the state, they are here.  This can be None.
            substitutionDictionary (Dict): If there are constants that need to be substituted into the expressions to lambdify, include them here.
        """
        if stateVariablesList == None :
            stateVariablesList = []
        if expressionsToLambdify == None :
            expressionsToLambdify = []
        if otherArgsList == None :
            otherArgsList = []
        if substitutionDictionary == None:
            substitutionDictionary = {}

        self._stateVariableList = stateVariablesList
        self._expressionsToGetLambdified = expressionsToLambdify
        self._time = time
        self._otherArgs = otherArgsList
        self._substitutionDictionary = substitutionDictionary

    @property
    def StateVariableListOrdered(self) -> List[sy.Expr]:
        """The list of state variables that are used to lambdify an expression.  This list is never 
        None.  It is likely the lambdified expressions will care about the order of the values in 
        this list.

        Returns:
            List[sy.Expr]: The list of state variables, often sy.Symbol's of sy.Functions of the Time symbol.
        """
        return self._stateVariableList

    @property
    def ExpressionsToLambdify(self) -> List[sy.Expr]:
        """The expressions that will get lambdified.  This list is never None, and the order of these 
        expressions may need to align to the order of the StateVariableListOrdered (for example, doing 
        multi-variable numerical integration, the first equation of motion in this list must be the 
        derivative of the first value in StateVariableListOrdered).

        Returns:
           List[sy.Expr]: The expressions to get lambdified.
        """
        return self._expressionsToGetLambdified

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

    @property 
    def OtherArguments(self) -> List[sy.Expr] :
        """If there are other arguments that need to be passed to the lambdified expression that are not 
        part of the state, those arguments are specified here.  This list is never None but may be empty.

        Returns:
            List[sy.Expr]: The list of other arguments.
        """
        return self._otherArgs

    @property 
    def SubstitutionDictionary(self) -> Dict :
        """Constants that should be substituted into the expressions to lambdify before lambdify'ing them.
        This is never None, but it may be empty.

        Returns:
            Dict: Constants that should be substituted into the expressions to lambdify.
        """
        return self._substitutionDictionary

    def CreateDefaultState(self) ->List[sy.Expr]:
        """ Makes a best guess for what the state ought to be.  You ABSOLUTELY should override this 
        function on your instances when it makes sense to.  Often the shape and order and structure 
        of the state is unique to your problem.  Often similar to but not exactly the same and hard to 
        generalize.  And there are times when you just want to do something manually and not with all 
        of the ceremony and cruft of a system like this overall module getting in the way.

        Returns:
            List[sy.Expr]: _description_
        """
        stateArray = [] # type: List[Any]
        if self.Time != None :
            stateArray.append(self.Time)
        if self.StateVariableListOrdered != None and len(self.StateVariableListOrdered) != 0:
            stateArray.append(self.StateVariableListOrdered)
        if self.OtherArguments != None and len(self.OtherArguments) != 0:
            stateArray.append(self.OtherArguments)
        return stateArray

    def CreateSimpleCallbackForSolveIvp(self, odeState :Optional[List[sy.Expr]]=None) -> Callable : 
        """Creates a lambdified expression of the (assumed) equations of motion in ExpressionsToLambdify.

        Args:
            odeState (List[sy.Expr], optional): The state that should be used to lambdify the 
            ExpressionsToLambdify. Defaults to None which will use the CreateDefaultState function.

        Returns:
            Callable: A callback to use in scipy.ode.solveivp functions (and odeint if you put time first).
        """
        if odeState == None :
            odeState = self.CreateDefaultState()
        equationsOfMotion = self.ExpressionsToLambdify
        eomList = []
        for thisEom in equationsOfMotion :
            # eom's could be constant equations.  Check, add if it doesn't
            if(hasattr(thisEom, "subs")) :
                thisEom = thisEom.subs(self.SubstitutionDictionary) 
            eomList.append(thisEom)   
        eomCallback = sy.lambdify(odeState, eomList)

        # don't need the next wrapper if there are no other args
        if self.OtherArguments == None or len(self.OtherArguments) == 0 :            
            return eomCallback

        # but if there are other arguments, handle that
        def callbackFunc(t, y, *args) :
            return eomCallback(t, y, args)
        return callbackFunc        

    def CreateSimpleCallbackForOdeint(self, odeState : Optional[List[sy.Expr]]=None) -> Callable : 
        """Creates a lambdified expression of the (assumed) equations of motion in ExpressionsToLambdify.

        Args:
            odeState (List[sy.Expr], optional): The state that should be used to lambdify the 
            ExpressionsToLambdify. Defaults to None which will use the CreateDefaultState function.

        Returns:
            Callable: A callback to use in scipy.integrate.odeint functions where time comes after y.
        """        
        originalCallback = self.CreateSimpleCallbackForSolveIvp(odeState)
        # don't need the next wrapper if there are no other args
        if self.OtherArguments == None or len(self.OtherArguments) == 0 :            
            def switchTimeOrderCallback(y, t) :
                return originalCallback(t, y)
            return switchTimeOrderCallback
        #else...        
        def switchTimeOrderCallback2(y, t, *args) :
            return originalCallback(t, y, *args)
        return switchTimeOrderCallback2   
    
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
        bcs = []
        for exp in expressionsToLambdify :
            bc = SymbolicProblem.SafeSubs(exp, constantsSubstitutionDictionary)
            bcs.append(bc)
        return sy.lambdify(stateExpressionList, bcs)    