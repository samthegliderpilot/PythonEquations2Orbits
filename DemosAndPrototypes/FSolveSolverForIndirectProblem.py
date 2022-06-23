from scipy.integrate import odeint

class FSolveSolverForIndirectProblem :
    def __init__(self, stateVarabiles, lambdas, eoms, stateForBcs, bcs, timeArray, nus) :
        self._stateVariables = stateVarabiles
        self._lambdas = lambdas
        self._eoms = eoms
        self._stateForBcs = stateForBcs
        self._bcs = bcs
        self._timeArray = timeArray
        if nus == None :
            nus = []
        self._nus = nus

    def integrateFromInitialValues(self, z0) :
        integratableCb = lambda z,t : self._eoms(t, z[0:len(self._stateVariables)], z[len(self._stateVariables):len(self._stateVariables)+len(self._lambdas)])
        return odeint(integratableCb, z0[0:len(self._stateVariables)+len(self._lambdas)], self._timeArray)

    def createBoundaryConditionStateFromIntegrationResult(self, ans) :
        finalState = []
        finalState.extend(ans[0])
        finalState.extend(ans[-1])
        
        return finalState
                
    def createCallbackForFSolve(self, initialEomState) :
        def callbackForFsolve(lambdaGuesses) :
            z0 = []
            z0.extend(initialEomState)
            #for sv in initialEomState :
            #    z0.append(constantsSubsDict[sv])
            z0.extend(lambdaGuesses)
            
            ans = self.integrateFromInitialValues(z0)
            finalState = self.createBoundaryConditionStateFromIntegrationResult(ans)  
            
            for i in range(len(self._nus), 0, -1) :
                finalState.append(lambdaGuesses[-1*i])   #TODO, need to generalize!!!
            #finalState.append(lambdaGuesses[-1])
            finalAnswers = []
            for bcCallback in self._bcs: 
                finalAnswers.append(bcCallback(*finalState))
        
            return finalAnswers    
        return callbackForFsolve