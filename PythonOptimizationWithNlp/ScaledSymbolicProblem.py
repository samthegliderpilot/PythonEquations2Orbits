import sympy as sy
from typing import List, Dict
from PythonOptimizationWithNlp.SymbolicOptimizerProblem import SymbolicProblem
import numpy as np

"""A Symbolic problem that has scaling factors over another problem.  Those 
factors can be constants, or symbols themselves that are in the substitution 
dictionary that get used by various solvers.
"""
class ScaledSymbolicProblem(SymbolicProblem) :
    def __init__(self, wrappedProblem : SymbolicProblem, newStateVariableSymbols : Dict, valuesToDivideStateVariablesWith : Dict, scaleTime : bool) :        
        """Initializes a new instance.

        Args:
            wrappedProblem (SymbolicProblem): The problem to scale.
            newStateVariableSymbols (Dict): The new state variables.  Often these are the same ones with some accent on them.  These should still be in terms 
            the wrappedProblems time symbol.
            valuesToDivideStateVariablesWith (Dict): The scaling factors on the state variables. These can be floats, or symbols of some sort that 
            end up being constants in the SubstitutionDictionary
            scaleTime (bool): Should the time be scaled to be between 0 and 1.
        """
        super().__init__()
        self._wrappedProblem = wrappedProblem

        self._timeFinalSymbol = wrappedProblem.TimeFinalSymbol
        self._timeInitialSymbol = wrappedProblem.TimeInitialSymbol
        self._timeSymbol = wrappedProblem.TimeSymbol
        
        self._scaleTime=scaleTime
        self._substitutionDictionary = wrappedProblem._substitutionDictionary        
        self._stateVariables = newStateVariableSymbols
        
        self._scalingDict = valuesToDivideStateVariablesWith

        newSvsInTermsOfOldSvs = []
        counter=0
        for sv in wrappedProblem.StateVariables :
            newSvsInTermsOfOldSvs.append(sv/valuesToDivideStateVariablesWith[sv])
            counter=counter+1

        fullSubsDict = {} # don't add in sv(t_0), that will confuse common scaling paradigms
        counter = 0
        for sv in wrappedProblem.StateVariables :
            fullSubsDict[sv] = newStateVariableSymbols[counter]*valuesToDivideStateVariablesWith[sv]            
            fullSubsDict[sv.subs(wrappedProblem.TimeSymbol, wrappedProblem.TimeFinalSymbol)] = newStateVariableSymbols[counter].subs(wrappedProblem.TimeSymbol, wrappedProblem.TimeFinalSymbol)*valuesToDivideStateVariablesWith[sv]            
            counter = counter+1


        self._controlVariables = wrappedProblem.ControlVariables
        scaledEoms = {}
        counter = 0
        for sv in wrappedProblem.StateVariables :
            if hasattr(wrappedProblem.EquationsOfMotion[sv], "subs") :
                scaledEoms[self._stateVariables[counter]] = wrappedProblem.EquationsOfMotion[sv].subs(fullSubsDict)*newSvsInTermsOfOldSvs[counter].diff(sv)
            else :
                scaledEoms[self._stateVariables[counter]] = wrappedProblem.EquationsOfMotion[sv]*newSvsInTermsOfOldSvs[counter].diff(sv)
            counter=counter+1
        self._equationsOfMotion = scaledEoms
        
        counter = 0
        bcSubsDict = {}
        for sv in wrappedProblem.StateVariables :
            svAtTf=sv.subs(wrappedProblem.TimeSymbol, wrappedProblem.TimeFinalSymbol)
            newSvAtTf = newStateVariableSymbols[counter].subs(self.TimeSymbol, self.TimeFinalSymbol)
            bcSubsDict[svAtTf] = newSvAtTf*valuesToDivideStateVariablesWith[sv]
            counter = counter+1
        bcs = []
        
        for bc in wrappedProblem.BoundaryConditions :

            bcs.append(bc.subs(bcSubsDict))
        self._boundaryConditions = bcs
        
        # should the path cost get scaled?  I think so, but if you know better, or if it doesn't work...
        self._unIntegratedPathCost = SymbolicProblem.SafeSubs(wrappedProblem.UnIntegratedPathCost, fullSubsDict)
        # do not scale the cost function! conditions made from the cost get scaled later
        self._terminalCost = wrappedProblem.TerminalCost
        if scaleTime :
            tf = wrappedProblem.TimeFinalSymbol
            tau = sy.Symbol(r'\tau')
            tauF = sy.Symbol(r'\tau_f')
            tau0 = sy.Symbol(r'\tau_0')
            timeSubs = {wrappedProblem.TimeInitialSymbol: tau0, wrappedProblem.TimeSymbol: tau, wrappedProblem.TimeFinalSymbol:tauF}
            self._controlVariables = ScaledSymbolicProblem.SafeSubs(self._controlVariables, timeSubs)
            orgSv = self._stateVariables
            self._stateVariables = ScaledSymbolicProblem.SafeSubs(self._stateVariables, timeSubs)
            self._unIntegratedPathCost = ScaledSymbolicProblem.SafeSubs(self._unIntegratedPathCost, timeSubs)
            self._terminalCost = ScaledSymbolicProblem.SafeSubs(self._terminalCost, timeSubs)
            self._boundaryConditions = ScaledSymbolicProblem.SafeSubs(self._boundaryConditions, timeSubs)

            # the correct thing to do is to make your state and control variables functions of time in Sympy
            # myCv = sy.Function('Control')(t)
            # BUT, we need to substitute t with tau*tf
            # BUT, we don't want myCv to be a function of tau*tf, just of tau
            # I don't know how to make the sympy subs function not go deep like that, so, we substitute these back...
            toSimple = {}
            fromSimple = {}
            for sv in self._stateVariables :
                adjustedSv = sv.subs(tau, self._wrappedProblem.TimeSymbol)
                toSimple[adjustedSv] = sy.Symbol(sv.name)
                fromSimple[toSimple[adjustedSv]] = sv
                
            for cv in self._controlVariables :
                adjustedCv = cv.subs(tau, self._wrappedProblem.TimeSymbol)
                toSimple[adjustedCv] = sy.Symbol(cv.name)
                fromSimple[toSimple[adjustedCv]] = cv
                
            realEom = {}
            i = 0
            timeSubs = { wrappedProblem.TimeSymbol: tau*tf}
            for sv in orgSv :
                # substitute in the dummy symbols in terms of something other than time or tau
                realEom[self._stateVariables[i]] = ScaledSymbolicProblem.SafeSubs(self._equationsOfMotion[sv], toSimple)
                # substitute in the scaled values
                realEom[self._stateVariables[i]] = ScaledSymbolicProblem.SafeSubs(realEom[self._stateVariables[i]], timeSubs)*wrappedProblem.TimeFinalSymbol
                # substitute back in the state variables in terms of time
                realEom[self._stateVariables[i]] = ScaledSymbolicProblem.SafeSubs(realEom[self._stateVariables[i]], fromSimple)
                i=i+1
            self._equationsOfMotion = realEom
            
            self._timeSymbol = tau 
            self._timeInitialSymbol = tau0
            self._timeFinalSymbol = tauF

    @property
    def ScaleTime(self) -> bool :
        """Gets if the time should be scaled.  This cannot be changed.

        Returns:
            bool: Will the time be scaled.
        """
        return self._scaleTime

    @property
    def ScalingValues(self) -> Dict :
        """Gets the scaling parameters.

        Returns:
            Dict: The scaling parameters.
        """
        return self._scalingDict

    @property
    def WrappedProblem(self) -> SymbolicProblem :
        """Gets the wrapped problem.

        Returns:
            SymbolicProblem: The problem getting scaled.
        """
        return self._wrappedProblem

    def DescaleResults(self, resultsDictionary : Dict[sy.Symbol, List[float]]) -> Dict[sy.Symbol, List[float]] :
        """After evaluating the problem numerically, descale the results to be back in terms of the original units.

        Args:
            resultsDictionary (Dict[sy.Symbol, List[float]]): The results dictionary.

        Returns:
            Dict[sy.Symbol, List[float]]: A new dictionary where the values are descaled AND the keys are the wrappedProblems's 
            state variables.
        """
        returnDict = {}
        counter = 0
        for key, value in resultsDictionary.items() :
            sv = key
            if sv in self.StateVariables and counter < len(self.WrappedProblem.StateVariables):
                originalSv = self.WrappedProblem.StateVariables[self.StateVariables.index(sv)]
                convertedArray = np.array(value, copy=True)* SymbolicProblem.SafeSubs(self.ScalingValues[originalSv], self.SubstitutionDictionary)
                returnDict[originalSv] = convertedArray
                counter = counter+1
            else :
                returnDict[key]=value
        return returnDict
    
    @property
    def TimeFinalSymbolOriginal(self)-> sy.Symbol:
        """Gets the original time final symbol from the wrapped problem.

        Returns:
            sy.Symbol: The wrapped time final symbol.
        """
        return self.WrappedProblem.TimeFinalSymbol

    @staticmethod
    def CreateBarVariables(orgVariables : List[sy.Expr], timeSymbol :sy.Expr) :
        """A helper function to make a 

        Args:
            orgVariables (List[sy.Expr]): _description_
            timeSymbol (sy.Expr): _description_

        Returns:
            _type_: _description_
        """
        baredVariables = []
        for var in orgVariables :
            baredVariables.append(sy.Function(r'\bar{' + var.name+ '}')(timeSymbol))
        return baredVariables

    def ScaleExpressions(self, expressions : List[sy.Expr]) -> List[sy.Expr]:
        """For some expression (or list of expressions), 

        Args:
            expressions (List[sy.Expr]): The expressions to scale.

        Returns:
            List[sy.Expr]: The scaled expressions
        """
        simpleSubsDict={} # for terminal cost
        counter=0
        for sv in self.WrappedProblem.StateVariables :
            oldSv = sv
            sv = self.StateVariables[counter]
            simpleSubsDict[oldSv] = sv*self.ScalingValues[oldSv]
            simpleSubsDict[oldSv.subs(self.WrappedProblem.TimeSymbol, self.WrappedProblem.TimeFinalSymbol)] = sv.subs(self.TimeSymbol, self.TimeFinalSymbol)*self.ScalingValues[oldSv]
            counter=counter+1

        return SymbolicProblem.SafeSubs(expressions, simpleSubsDict)

    def TransversalityConditionsByAugmentation(self, nus : List[sy.Symbol], lambdasFinal : List[sy.Symbol] = None) -> List[sy.Expr]:
        """Creates the transversality conditions by augmenting the terminal constraints to the terminal cost. Note that 
        this calls the wrapped problems TransversalityConditionsByAugmentation and then scales that expression.

        Args:
            nus (List[sy.Symbol]): The constant parameters to augment the constraints to the terminal cost with.
            lambdasFinal (List[sy.Symbol]): The costate symbols at the final time.  If None it will use the problems
            CostateSymbols at the final time, and if those are not set, then an exception will be raised.

        Returns:
            List[sy.Expr]: The list of transversality conditions, that ought to be treated like normal boundary conditions.
        """
        if lambdasFinal == None :
            if self.CostateSymbols != None and len(self.CostateSymbols) > 0:
                lambdasFinal = SymbolicProblem.SafeSubs(self.CostateSymbols, {self.TimeSymbol: self.TimeFinalSymbol})
            else :
                raise Exception("No source of costate symbols.") 

        finalConditions = self.WrappedProblem.TransversalityConditionsByAugmentation(nus, lambdasFinal)
        return self.ScaleExpressions(finalConditions)
    
    def TransversalityConditionInTheDifferentialForm(self, hamiltonian : sy.Expr, dtf, lambdasFinal : List[sy.Symbol] = None) ->List[sy.Expr]:
        """Creates the transversality conditions by with the differential form of the transversality conditions. Note that 
        this calls the wrapped problems TransversalityConditionsByAugmentation and then scales that expression.

        Args:
            hamiltonian (sy.Expr): The hamiltonian in terms of the costate values (as opposed to the control variable)
            dtf (_type_): Either 0 if the final time is fixed, or a symbol indicating that the final time is not fixed.
            lambdasFinal (List[sy.Symbol]): The costate symbols at the final time.  If None it will use the problems
            CostateSymbols at the final time, and if those are not set, then an exception will be raised.

        Returns:
            List[sy.Expr]: The list of transversality conditions, that ought to be treated like normal boundary conditions.
        """
        if lambdasFinal == None :
            if self.CostateSymbols != None and len(self.CostateSymbols) > 0:
                lambdasFinal = SymbolicProblem.SafeSubs(self.CostateSymbols, {self.TimeSymbol: self.TimeFinalSymbol})
            else :
                raise Exception("No source of costate symbols.") 

        finalConditions = self.WrappedProblem.TransversalityConditionInTheDifferentialForm(hamiltonian, dtf, lambdasFinal)
        return self.ScaleExpressions(finalConditions)
       