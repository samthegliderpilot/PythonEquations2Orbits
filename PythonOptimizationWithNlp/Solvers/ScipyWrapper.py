from scipy.optimize import minimize, OptimizeResult
from typing import List,Dict,Callable

from PythonOptimizationWithNlp.NumericalOptimizerProblem import NumericalOptimizerProblemBase

#TODO: Consider making a base solver type that takes some of the load off of this type

class ScipyMinimizeWrapper:
    """Provides wrapper functions to work with scipy.optimize functions.  This base type solves 
    for a optimal value by creating a discretized optimizer state (z) from the state and control 
    of a problem using a trapizod rule.
    """
    def __init__(self, problem : NumericalOptimizerProblemBase) : 
        """ Initialize a new instance.
        
        Args:
            problem (NumericalOptimizerProblemBase): The problem getting solved.
        """ 
        self.Problem = problem
        self.NumberOfVariables = problem.NumberOfOptimizerStateVariables

    def ScipyOptimize(self, n : int, tArray: List[float] = None, zGuess : List[float] = None, optMethod : str ="SLSQP") -> OptimizeResult: 
        """A default implimentation of calling scipy.optimize.minimize

        Args:
            n (int): The number of segments to districtrize the solution into
            tArray (List[float], optional): The time array. Defaults to None which will trigger calling CreateTimeRange on the problem.
            zGuess (List[float], optional): An initial guess at the optimizer state (fully discretized). Defaults to None which will trigger calling CreateInitialGuess on the problem.
            optMethod (str, optional): The "method" parameter for scipy.optimize.minimize. Defaults to "SLSQP", and be sure to pick something that can handle constraints.

        Returns:
            OptimizeResult: The structure returned from a scipy.optimize.minimize call.
        """
        # handle default case of no time range specified
        if(tArray == None) :
            tArray = self.Problem.CreateTimeRange(n)

        # and create the initial guess if none was given
        if zGuess == None :
            zGuess = self.CreateDiscretizedInitialGuess(tArray)

        # create the constraints
        cons = self.CreateIndividualCollocationConstraints(tArray)
        cons.extend(self.CreateIndividualBoundaryValueConstraintCallbacks())

        costFunctionFlippedArgsForScipy = lambda z, t : self.CostFunctionInTermsOfZ(t, z)

        # call scipy
        ans = minimize(costFunctionFlippedArgsForScipy, zGuess, args=(tArray), method=optMethod, constraints=cons)
        return ans

    def CreateDiscretizedInitialGuess(self, timeArray : List[float]) ->List[float] :
        """Creates the discretized initial guess for the scipy minimizer to start with.

        Args:
            timeArray (List[float]): The optimal time array to use for the districtrization points.

        Returns:
            List[float]: A vector of the districtrized points, starting with the state[0] from t0 to tf, followed by state[1] from t0 to tf...
        """
        n = len(timeArray)-1
        z = [0.0] * (n+1) * self.NumberOfVariables
        for i in range(0, n+1) :
            t = timeArray[i]
            guessArrays = self.Problem.InitialGuessCallback(t)
            guessAtT = []
            guessAtT.extend(guessArrays) # states
            j=0
            for guessVal in guessAtT :                
                z[i + j*n+j] = guessVal
                j=j+1
        return z

    
    def ConvertScipyOptimizerOutputToDictionary(self, optStruct: OptimizeResult) ->Dict[object, List[float]]:
        """After evaluating a set of results, this will take those results and put them into a form that NumericalOptimizerProblemBase.AddResultsToFigure 
        expects."

        Args:
            optStruct (OptimizeResult): The results from a scipy.optimize.minimize call.

        Returns:
            Dict[object, List[float]]: The state variable and control variables in a dictionary key'ed off of the State and Control array in the problem.
        """
        return self.ConvertDiscretizedStateToDict(optStruct.x)

    def ConvertDiscretizedStateToDict(self, z :List[float]) ->Dict[object, List[float]]: 
        """With a districtrized optimizer state, this will that state and put them into a form that NumericalOptimizerProblemBase.AddResultsToFigure 
        expects."

        Args:
            z (List[float]): The optimizer state.

        Returns:
            Dict[object, List[float]]: The state variable and control variables in a dictionary key'ed off of the State and Control array in the problem.
        """
        tempList = [] 
        tempList.extend(self.Problem.State)
        tempList.extend(self.Problem.Control)
        n = int(len(z)/self.Problem.NumberOfOptimizerStateVariables)-1
        np1 = n+1
        start = 0
        endMultiplier = 1
        finalDict = {}
        for state in tempList :
            finalDict[state] = z[start:(np1)*endMultiplier]
            start = (np1)*endMultiplier
            endMultiplier=endMultiplier+1
        return finalDict

    # @staticmethod
    # def ExtractFinalValuesFromDisctictrizedState(problem : NumericalOptimizerProblemBase, z : List[float]) -> List[float]:
    #     finalValues = []
    #     everyN = len(z) / problem.NumberOfOptimizerStateVariables
    #     for i in range(0, problem.NumberOfOptimizerStateVariables) :
    #         finalValues.append(z[int(everyN*(i+1))-1])
    #     return finalValues


    def CreateIndividualBoundaryValueConstraintCallbacks(self) -> List[Callable[[List[float]], float]]:
        """Creates callbacks for the difference between the boundary conditions of each state variable in the optimizer state and 
        the desired value of each of boundary conditions as defined by the problem.

        Returns:
            List[Callable[[List[float]], float]]: The evaluate'able boundary condition callbacks in a list.
        """
        cons = []
        nMultiplier = 0
        for thisOtherBc in self.Problem.BoundaryConditionCallbacks :
            def moreVerboseCallback(z, thisOtherBc=thisOtherBc,) : 
                return thisOtherBc(0.0, self.GetOptimizerStateAtIndex(z, 0), 1.0, self.GetOptimizerStateAtIndex(z, -1))
            #callbackWithProperlyClosedOverValues = lambda z, thisOtherBc=thisOtherBc, : thisOtherBc(0.0, self.GetOptimizerStateAtIndex(z, 0), 1.0, self.GetOptimizerStateAtIndex(z, -1))# , (z[(n+1)*(nMultiplier+1)-1]))
            cons.append({'type': 'eq', 'fun': moreVerboseCallback})
            nMultiplier=nMultiplier+1
        return cons
    
    def CollocationConstraintIntegrationRule(self, t : List[float], z : List[float], timeIndex :int, stateIndex : int) -> float:
        """This is the trapizod rule implimented as a colocation constraint.  Derived types should 
        override this function to implement other rules. scipy's minimize function is a little
        frustrating in that it wants to call this for each colocation constraint instead of taking 
        a vector.

        Args:
            t (List[float]): The time array for this run
            z (List[float]): The optimizer state (discretized)
            timeIndex (int): The current index for time
            stateIndex (int): The state/optimization variable index

        Returns:
            float: The constraint value for this step that the optimizer will try to drive to 0.
        """

        # This is where the trapezoidal rule is captured.  Trying to figure out the best way to 
        # make this something configurable instead of baked into this solver type
        # BUT, how the state is layed out and managed is something that is solver specific.
        # I'm not sure there is a good way to extract that that isn't just a lot of trouble.
        eom =  self.EquationOfMotionInTermsOfOptimizerState
        optimizerStateAtI = self.GetOptimizerStateAtIndex(z, timeIndex)
        optimizerStateAtIPlus1 = self.GetOptimizerStateAtIndex(z, timeIndex+1)
        step = (t[timeIndex+1]-t[timeIndex])
        
        return 0.5*step*(eom(t[timeIndex+1], optimizerStateAtIPlus1)[stateIndex]+eom(t[timeIndex], optimizerStateAtI)[stateIndex]) - (optimizerStateAtIPlus1[stateIndex]-optimizerStateAtI[stateIndex])

    def CreateIndividualCollocationConstraints(self, t : List[float]) -> List[Dict]:
        """Creates the collocation constraints for the function.  This will call self.CollocationConstraintIntegrationRule, and 
        to try different districtrization schemes, override that function.

        Args:
            t (List[float]): The time array to districtrize the equations of motion over.

        Returns:
            List[Dict]: A list dictionaries that can immediately be used by the scipy optimizer. 
        """
        cons = []
        n = len(t)-1      
        for stateStep in range(0, self.Problem.NumberOfStateVariables) :
            for i in range(0, n) :
                def makeCallback(i, stateStep) :
                    # jumpting through this extra callback function to handle the closure over 1 correctly
                    cb = lambda z : self.CollocationConstraintIntegrationRule(t, z, i, stateStep)   
                    return cb
                cons.append({'type': 'eq', 'fun': makeCallback(i, stateStep)})
        return cons

    def GetOptimizerStateAtIndex(self, z : List[float], i : int)->List[float]  :
        """Returns the state of index i from the z.  The base function assumes 
        that the z is List[float], and state variable 1 from T_0 to T_f is first, then 
        state variable 2 follows in the same time range ([x_0, x_1, ... x_tf, y_0, y_1... y_tf, u_1_0, u_1_1... u_1_tf]) and 
        will return an array of just x, y, and u for the given index (if i=1, then the array returned is [x_1, y_1, u_1]).

        Args:
            z (List[float]): The discretized optimizer state from the 
            i (int): The index to get a state from.

        Returns:
            List[float]: The state at index i.
        """
        n = int(len(z)/self.NumberOfVariables)
        if i == -1 :
            i = n+i
        stateAtI = []
        for j in range(0,self.NumberOfVariables) :
            stateAtI.append(z[i+(j*n)])
        return stateAtI        

    def EquationOfMotionInTermsOfOptimizerState(self, t : float, z : List[float]) -> List[float]:
        """Evaluates the equations of motion where the state passed in is the state the optimizer sees. 
        That state is a single flat array of the state with the control values appended to it.

        Args:
            t (float): The current time.
            z (List[float]): The entire state of the optimizer at t.

        Returns:
            List[float]: The flat list of the values of the equations of motion at t.
        """
        #state = z[0:self.Problem.NumberOfStateVariables]
        #controlState = z[self.Problem.NumberOfStateVariables:self.Problem.NumberOfOptimizerStateVariables]
        return self.Problem.EquationOfMotion(t, z)        


    def CostFunctionInTermsOfZ(self, time : List[float], z : List[float]) -> float:
        """Evalutes the cost of the function from the optimizer state z.

        Args:
            time (List[float]): The time array that the function is evaluated over.
            z (List[float]): The optimizer state (the discretized state and control).


        Returns:
            float: The cost of this function to be minimized.
        """
        return self.Problem.CostFunction(time, self.ConvertDiscretizedStateToDict(z))