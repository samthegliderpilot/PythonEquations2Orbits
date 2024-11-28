from typing import List
import sympy as sy
from pyeq2orb.Utilities.Typing import SymbolOrNumber
import math

class gravationalBody:
    def __init__(self, x: sy.Symbol, y: sy.Symbol, z: sy.Symbol, mu: SymbolOrNumber):
        self.x = x
        self.y = y
        self.z = z
        self.mu = mu

def nBodyDifferentialEquation(x : sy.Symbol, y: sy.Symbol, z: sy.Symbol, listOfNBodies : List[gravationalBody]) -> sy.Matrix:
    sum = sy.Matrix([[0.0], [0.0], [0.0]])
    for i in range(0, len(listOfNBodies)):
        cb = listOfNBodies[i]
        mu = cb.mu
        xcb = cb.x
        ycb = cb.y
        zcb = cb.z
        rVector = sy.Matrix([[x-xcb], [y-ycb], [z-zcb]])
        rMag = sy.sqrt(rVector[0]**2 + rVector[1]**2+rVector[2]**2)
        term = -1*mu*rVector/(rMag**3)
        sum = sum + term
    return sum


class cr3bpPlanetCallbacks:

    @staticmethod
    def secondaryXExpression(t, mu):
        return sy.cos(t)*(1-mu) # bigger mu value here...

    @staticmethod
    def secondaryYExpression(t, mu):
        return sy.sin(t)*(1-mu)

    @staticmethod
    def secondaryZExpression(t, mu):
        return 0.0
    
    @staticmethod
    def primaryXExpression(t, mu):
        return -1*sy.cos(t)*(mu)
    
    @staticmethod
    def primaryYExpression(t, mu):
        return -1*sy.sin(t)*(mu)
    
    @staticmethod
    def primaryZExpression(t, mu):
        return 0.0

    # I don't like the copy/paste here, but it's not the end of the world
    
    # REMEMBER these are for numerical operations AFTER things have been lambdified
    @staticmethod
    def secondaryXCallbackOfTime(t : float, mu: float):
        localMu = mu
        return lambda time : math.cos(time)*(1-localMu)

    @staticmethod
    def secondaryYCallbackOfTime(t, mu):
        localMu = mu
        return lambda time : math.sin(time)*(1-localMu)    
    
    @staticmethod
    def secondaryZCallbackOfTime(t, mu):
        return lambda time : 0.0
        
    @staticmethod
    def primaryXCallbackOfTime(t, mu):
        localMu = mu
        return lambda time : -1*math.cos(time)*(localMu)
        
    @staticmethod
    def primaryYCallbackOfTime(t, mu):
        localMu = mu
        return lambda time: -1*math.sin(time)*(localMu)
        
    @staticmethod
    def primaryZCallbackOfTime(t, mu):
        return lambda time : 0.0