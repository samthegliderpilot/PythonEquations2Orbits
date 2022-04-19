import sympy as sy
from typing import TypedDict, List, Dict
from collections import OrderedDict
from PythonOptimizationWithNlp.SymbolicOptimizerProblem import SymbolicProblem
import math
import matplotlib.pyplot as plt
import numpy as np
from PythonOptimizationWithNlp.Utilities.inherit import inherit_docstrings

@inherit_docstrings
class PlanerLeoToGeoProblem(SymbolicProblem) :
    def __init__(self) :
        """Initializes a new instance.
        """
        super().__init__()

        self._constantSymbols = []
        mu = sy.Symbol(r'\mu', real=True, positive=True)
        self._constantSymbols.append(mu)

        thrust = sy.Symbol('T', real=True, positive=True)
        self._constantSymbols.append(thrust)

        m0 = sy.Symbol('m_0', real=True, positive=True)
        self._constantSymbols.append(m0)

        g = sy.Symbol('g', real=True, positive=True)
        self._constantSymbols.append(g)

        isp = sy.Symbol('I_{sp}', real=True, positive=True)
        self._constantSymbols.append(isp)

        self.Ts = sy.Symbol('t', real=True)
        self._t0 = sy.Symbol('t_0', real=True)
        self._tf = sy.Symbol('t_f', real=True, positive=True)

        self._stateVariables=[
            sy.Function('r', real=True, positive=True)(self.Ts),
            sy.Function('u', real=True, nonnegative=True)(self.Ts),
            sy.Function('v', real=True, nonnegative=True)(self.Ts),
            sy.Function('\\theta', real=True, nonnegative=True)(self.Ts)]

        self._controlVariables = [sy.Function('\\alpha', real=True)(self.Ts)]

        self._pathConstraints = {} # none for this instance

        self._finalBoundaryConditions = [
                self._stateVariables[1].subs(self.Ts, self._tf),
                sy.sqrt(mu/self._stateVariables[0].subs(self.Ts, self._tf))-self._stateVariables[2].subs(self.Ts, self._tf)
        ]

        self._terminalCost = -1.0*self._stateVariables[0] # negative because we are minimizing

        rs = self._stateVariables[0]
        us = self._stateVariables[1]
        vs = self._stateVariables[2]
        longS = self._stateVariables[3]
        control = self._controlVariables[0]

        self.MassFlowRate = -1*thrust/(isp*g)
        self.MassEquation = m=m0+self.Ts*self.MassFlowRate

        self._equationsOfMotion = OrderedDict()
        self._equationsOfMotion[rs] = us
        self._equationsOfMotion[us] = vs*vs/rs - mu/(rs*rs) + thrust*sy.sin(control)/self.MassEquation
        self._equationsOfMotion[vs] = -vs*us/rs + thrust*sy.cos(control)/self.MassEquation
        self._equationsOfMotion[longS] = vs/rs

    @property
    def StateVariables(self) -> List[sy.Symbol]:
        return self._stateVariables

    @property
    def ControlVariables(self) -> List[sy.Symbol]:
        return self._controlVariables

    @property
    def EquationsOfMotion(self) -> Dict[sy.Symbol, sy.Expr]:
        return self._equationsOfMotion

    @property
    def FinalBoundaryConditions(self) ->List[sy.Expr] : #TODO: Rename to just boundary conditions
        return self._finalBoundaryConditions

    @property
    def PathConstraints(self) -> Dict[sy.Symbol, sy.Expr] :
        return self._pathConstraints
    
    @property
    def ConstantSymbols(self) -> List[sy.Symbol] :
        return self._constantSymbols

    @property
    def TimeSymbol(self) -> sy.Expr :
        return self.Ts

    @property 
    def Time0Symbol(self) -> sy.Expr :
        return self._t0

    @property 
    def TimeFinalSymbol(self) -> sy.Expr :
        return self._tf

    @property
    def TerminalCost(self) -> sy.Expr :
        return self._terminalCost

    @property
    def UnIntegratedPathCost(self) -> sy.Expr :
        return 0

    def AppendConstantsToSubsDict(self, existingDict : dict, muVal : float, gVal : float, thrustVal : float, m0Val : float, ispVal : float) :
        """Helper function to make the substitution dictionary that is often needed when lambdifying 
        the symbolic equations.

        Args:
            existingDict (dict): The dictionary to add the values to.
            muVal (float): The gravitational parameter for the CB
            gVal (float): The value of gravity for fuel calculations (so use 1 Earth G)
            thrustVal (float): The value of the thrust
            m0Val (float): The initial mass of the spacecraft
            ispVal (float): The isp of the engines
        """
        existingDict[self.ConstantSymbols[0]] = muVal
        existingDict[self.ConstantSymbols[1]] = thrustVal
        existingDict[self.ConstantSymbols[2]] = m0Val
        existingDict[self.ConstantSymbols[3]] = gVal
        existingDict[self.ConstantSymbols[4]] = ispVal

    def PlotSolution(self, tUnscaled : List[float], orderedDictOfUnScaledValues : Dict[object, List[float]], label : str) :
        
        solsLists = []
        for key, val in orderedDictOfUnScaledValues.items() :
            solsLists.append(val)
        plt.title("Longitude (rad)")
        plt.plot(tUnscaled/86400, solsLists[3]%(2*math.pi))

        plt.tight_layout()
        plt.grid(alpha=0.5)
        plt.legend(framealpha=1, shadow=True)
        plt.show()        
        plt.title("Radius")
        plt.plot(tUnscaled/86400, solsLists[0])

        plt.tight_layout()
        plt.grid(alpha=0.5)
        plt.legend(framealpha=1, shadow=True)
        plt.show()

        plt.title('Lambdas')
        plt.plot(tUnscaled/86400, solsLists[4], label=r'$\lambda_r$ = ' + str(solsLists[4][0]))
        plt.plot(tUnscaled/86400, solsLists[5], label=r'$\lambda_u$ = ' + str(solsLists[5][0]))
        plt.plot(tUnscaled/86400, solsLists[6], label=r'$\lambda_v$ = ' + str(solsLists[6][0]))
        if len(solsLists) > 7 :
            plt.plot(tUnscaled/86400, solsLists[7], label=r'$\lambda_{\theta}}$ = ' + str(solsLists[7][0]))

        plt.tight_layout()
        plt.grid(alpha=0.5)
        plt.legend(framealpha=1, shadow=True)
        plt.show()

        ax = plt.subplot(111, projection='polar')
        thetas = []
        rads = []
        for ang, rad in zip(solsLists[3], solsLists[0]):
            thetas.append(ang)
            rads.append(rad)
        plt.title("Position Polar")    
        ax.plot(thetas, rads)

        plt.tight_layout()
        plt.grid(alpha=0.5)
        plt.legend(framealpha=1, shadow=True)
        plt.show()

        plt.tight_layout()
        plt.grid(alpha=0.5)
        plt.legend(framealpha=1, shadow=True)
        plt.title("Thrust angle (degrees vs days)")

        plt.plot(tUnscaled/86400, np.arctan2(solsLists[5], solsLists[6])*180.0/math.pi)
        plt.show()
          