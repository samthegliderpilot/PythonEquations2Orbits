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
        self.Mu = mu

        thrust = sy.Symbol('T', real=True, positive=True)
        self.Thrust = thrust

        m0 = sy.Symbol('m_0', real=True, positive=True)
        self.MassInitial = m0

        g = sy.Symbol('g', real=True, positive=True)
        self.Gravity = g

        isp = sy.Symbol('I_{sp}', real=True, positive=True)
        self.Isp = isp

        self._timeSymbol = sy.Symbol('t', real=True)
        self._timeInitialSymbol = sy.Symbol('t_0', real=True)
        self._timeFinalSymbol = sy.Symbol('t_f', real=True, positive=True)

        self._stateVariables.extend([
            sy.Function('r', real=True, positive=True)(self._timeSymbol),
            sy.Function('u', real=True, nonnegative=True)(self._timeSymbol),
            sy.Function('v', real=True, nonnegative=True)(self._timeSymbol),
            sy.Function('\\theta', real=True, nonnegative=True)(self._timeSymbol)])

        self._controlVariables.extend([sy.Function('\\alpha', real=True)(self._timeSymbol)])

        self._boundaryConditions.extend([
                self._stateVariables[1].subs(self._timeSymbol, self._timeFinalSymbol),
                self._stateVariables[2].subs(self._timeSymbol, self._timeFinalSymbol)-sy.sqrt(mu/self._stateVariables[0].subs(self._timeSymbol, self._timeFinalSymbol))
        ])

        self._terminalCost = self._stateVariables[0].subs(self._timeSymbol, self._timeFinalSymbol) # maximization problem

        #self._terminalCost = 0.0
        #self._unintegratedCost = -1*self.StateVariables[0].diff(self.Ts) - self.StateVariables[0].subs(self.Ts, self._t0)
        rs = self._stateVariables[0]
        us = self._stateVariables[1]
        vs = self._stateVariables[2]
        longS = self._stateVariables[3]
        control = self._controlVariables[0]

        self.MassFlowRate = -1*thrust/(isp*g)
        self.MassEquation = m0+self._timeSymbol*self.MassFlowRate

        self._equationsOfMotion[rs] = us
        self._equationsOfMotion[us] = vs*vs/rs - mu/(rs*rs) + thrust*sy.sin(control)/self.MassEquation
        self._equationsOfMotion[vs] = -vs*us/rs + thrust*sy.cos(control)/self.MassEquation
        self._equationsOfMotion[longS] = vs/rs
   
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
          


