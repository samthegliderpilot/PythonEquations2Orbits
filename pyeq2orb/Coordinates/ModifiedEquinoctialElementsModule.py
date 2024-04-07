from __future__ import annotations
import sympy as sy
import math
import pyeq2orb.Coordinates.KeplerianModule as Keplerian
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from typing import List, Dict, Optional, cast, Union, cast
from numbers import Real

class ModifiedEquinoctialElements:
    def __init__(self, p:SymbolOrNumber, f:SymbolOrNumber, g:SymbolOrNumber, h:SymbolOrNumber, k:SymbolOrNumber, l:SymbolOrNumber, mu:SymbolOrNumber) :
        self.SemiParameter = p
        self.EccentricityCosTermF = f
        self.EccentricitySinTermG = g
        self.InclinationCosTermH = h
        self.InclinationSinTermK = k
        self.TrueLongitude = l
        self.GravitationalParameter = mu

        self._wSymbol = sy.Symbol('w')
        self._sSquaredSymbol = sy.Symbol('s^2')
        self._alphaSymbol = sy.Symbol(r'\alpha')
        self._rSymbol = sy.Symbol('r')

    @staticmethod
    def FromCartesian(x, y, z, vx, vy, vz) -> ModifiedEquinoctialElements :
        raise Exception("Not implemented")
    
    @property
    def W(self) -> SymbolOrNumber:
        return 1+self.EccentricityCosTermF*sy.cos(self.TrueLongitude)+self.EccentricitySinTermG*sy.sin(self.TrueLongitude)

    @property
    def SSquared(self)-> SymbolOrNumber :
        return 1+self.InclinationCosTermH**2+self.InclinationSinTermK**2
    
    @property
    def AlphaSquared(self)-> SymbolOrNumber :
        return self.InclinationCosTermH**2-self.InclinationSinTermK**2

    @property
    def Radius(self)-> SymbolOrNumber :
        return self.SemiParameter / self.W

    @property
    def WSymbol(self)-> sy.Symbol :
        return self._wSymbol

    @property
    def SSquaredSymbol(self)-> sy.Symbol :
        return self._sSquaredSymbol
    
    @property
    def AlphaSquaredSymbol(self)-> sy.Symbol :
        return self._alphaSymbol

    def AuxiliarySymbolsDict(self) -> dict[sy.Symbol, SymbolOrNumber] :
        return {self.WSymbol: self.W, 
                self.SSquaredSymbol: self.SSquared,
                self.AlphaSquaredSymbol: self.AlphaSquared,
                self.RadiusSymbol: self.Radius}

    @property
    def RadiusSymbol(self)->sy.Symbol :
        return self._rSymbol

    def ToKeplerian(self) -> Keplerian.KeplerianElements :
        l = self.TrueLongitude
        f = self.EccentricityCosTermF
        g = self.EccentricitySinTermG
        h = self.InclinationCosTermH
        k = self.InclinationSinTermK
        p = self.SemiParameter

        a = p/(1-f**2-g**2)
        e = sy.sqrt(f*f+g*g)
        # it is not clear to me if the atan2 in the MME PDF passes in x,y or y,x (y,x is right for sy)
        i = sy.atan2(2*sy.sqrt(h*h+k*k), 1-h*h-k*k)
        w = sy.atan2(g*h-f*k, f*h+g*k)
        if math.isnan(w):
            w=0.0
        raan = sy.atan2(k, h)
        ta = l-raan+w
        return Keplerian.KeplerianElements(a, e, i, w, raan, ta, self.GravitationalParameter)
    
    def CreateFgwToInertialAxes(self)->sy.Matrix:
        p = self.InclinationSinTermK
        q = self.InclinationCosTermH

        firstTerm = 1/(1+p**2+q**2)
        fm = firstTerm*sy.Matrix([[1-p**2+q**2], [2*p*q], [-2*p]])
        gm = firstTerm*sy.Matrix([[2*p*q], [1+p**2-q**2], [2*q]])
        wm = firstTerm*sy.Matrix([[2*p], [-2*q], [1-p**2-q**2]])

        return sy.Matrix([[fm[0,0], fm[1,0], fm[2,0]],[gm[0,0],gm[1,0],gm[2,0] ],[wm[0,0], wm[1,0], wm[2,0]]]).transpose()

    def ToMotionCartesian(self, useSymbolsForAuxiliaryElements : bool= False) -> MotionCartesian :
        l = self.TrueLongitude
        f = self.EccentricityCosTermF
        g = self.EccentricitySinTermG        
        h = self.InclinationCosTermH
        k = self.InclinationSinTermK
        p = self.SemiParameter
        mu = self.GravitationalParameter
        
        if useSymbolsForAuxiliaryElements :
            w:SymbolOrNumber = self.WSymbol
            r:SymbolOrNumber = self.RadiusSymbol
            sSquared:SymbolOrNumber = self.SSquaredSymbol
            alpSq:SymbolOrNumber = self.AlphaSquaredSymbol
        else :
            w = self.W
            r = self.Radius
            sSquared = self.SSquared        
            alpSq = self.AlphaSquared
        
        cosL = sy.cos(l)
        sinL = sy.sin(l)
        rx = (r/sSquared)*(cosL + alpSq*cosL + 2*h*k*sinL)
        ry = (r/sSquared)*(sinL - alpSq*sinL + 2*h*k*cosL)
        rz = 2*(r/sSquared)*(h*sinL - k*cosL)

        sqrtMuOverP = sy.sqrt(mu/p)
        vx = (-1/sSquared) *sqrtMuOverP*(   sinL + alpSq*sinL - 2*h*k*cosL + g - 2*f*h*k + alpSq*g)
        vy = (-1/sSquared) *sqrtMuOverP*(-cosL + alpSq*cosL + 2*h*k*sinL - f + 2*g*h*k + alpSq*f)
        vz = (2/(sSquared))*sqrtMuOverP*(h*cosL + k*sinL + f*h + g*k)

        return MotionCartesian(Cartesian(rx, ry, rz), Cartesian(vx, vy, vz))

    def ToCartesianArray(self) ->List[List[SymbolOrNumber]]:
        motion = self.ToMotionCartesian()
        return [[motion.Position.X,motion.Position.Y,motion.Position.Z], [motion.Velocity.X,motion.Velocity.Y,motion.Velocity.Z]]

    # def ToPseudoNormalizedCartesian(self, useSymbolsForAuxiliaryElements = False) :
    #     # sorry for the copy/paste of the above
    #     p = self.SemiParameter
    #     f = self.EccentricityCosTermF
    #     g = self.EccentricitySinTermG        
    #     k = self.InclinationSinTermK
    #     h = self.InclinationCosTermH
    #     tl = self.TrueLongitude
    #     mu = self.GravitationalParameter
    #     if useSymbolsForAuxiliaryElements :
    #         alp2 = self.AlphaSquaredSymbol
    #         s2 = self.SSquaredSymbol
    #         w = self.WSymbol
    #         rM = self.Radius
    #     else :
    #         alp2 = h*h-k*k
    #         s2 = 1+h*h+k*k
    #         w = 1+f*sy.cos(tl)+g*sy.sin(tl)
    #         rM = p/w
        
    #     x = (sy.cos(tl) + alp2*sy.cos(tl) + 2*h*k *sy.sin(tl))
    #     y = (sy.sin(tl) + alp2*sy.sin(tl) + 2*h*k *sy.cos(tl))
    #     z = (1/s2)*(h*sy.sin(tl) - k*sy.cos(tl))

    #     # not being rigorous about the normalizing here, just removing various parameters
    #     vx = (-1) * (sy.sin(tl) + alp2*sy.sin(tl) - 2*h*k*sy.cos(tl) + g - 2*f*h*k + alp2*g)
    #     vy = (-1) * (-1*sy.cos(tl) + alp2*sy.cos(tl) + 2*h*k*sy.sin(tl) - f - 2*g*h*k + alp2*f)
    #     vz = (1/s2) * (h*sy.cos(tl) + k*sy.sin(tl) + f*h + g*k)

    #     return MotionCartesian(Cartesian(x, y, z), Cartesian(vx, vy, vz))   

    def ToArray(self) -> List[SymbolOrNumber] :
        return [self.SemiParameter, self.EccentricityCosTermF, self.EccentricitySinTermG, self.InclinationCosTermH, self.InclinationSinTermK, self.TrueLongitude]

    @staticmethod
    def FromMotionCartesian(motion, gravitationalParameter :SymbolOrNumber) ->ModifiedEquinoctialElements:
        # TODO: something that avoids keplerian elements
        return ConvertKeplerianToEquinoctial(Keplerian.KeplerianElements.FromMotionCartesian(motion, gravitationalParameter))

    @staticmethod
    def CreateEphemeris(equinoctialElementsList : List[ModifiedEquinoctialElements]) -> List[MotionCartesian] :
        motions = []
        for equi in equinoctialElementsList :
            motions.append(equi.ToMotionCartesian())
        return motions

    def CreatePerturbationMatrix(self, useSymbolsForAuxElements :Optional[bool]= False) ->sy.Matrix :
        eqElements=self
        mu = eqElements.GravitationalParameter
        pEq = eqElements.SemiParameter        
        fEq = eqElements.EccentricityCosTermF
        gEq = eqElements.EccentricitySinTermG        
        hEq = eqElements.InclinationCosTermH
        kEq = eqElements.InclinationSinTermK
        lEq = eqElements.TrueLongitude
        if useSymbolsForAuxElements : 
            w = self.WSymbol
            s2 = self.SSquaredSymbol
        else :
            w = 1+fEq*sy.cos(lEq)+gEq*sy.sin(lEq)
            s2 = cast(sy.Symbol, 1+kEq**2+hEq**2)
        sqrtPOverMu=sy.sqrt(pEq/mu)
        # note that in teh 3rd row, middle term, I added a 1/w because I think it is correct even though the MME pdf doesn't have it
        # note that SAO/NASA (Walked, Ireland and Owens) says that the final term of the third element is feq/w instead of heq/w
        B = sy.Matrix([[0, (2*pEq/w)*sqrtPOverMu, 0],
                    [   sqrtPOverMu*sy.sin(lEq), sqrtPOverMu*(1.0/w)*((w+1)*sy.cos(lEq)+fEq), -sqrtPOverMu*(gEq/w)*(kEq*sy.sin(lEq)-hEq*sy.cos(lEq))],
                    [-sqrtPOverMu*sy.cos(lEq), sqrtPOverMu*(1.0/w)*((w+1)*sy.sin(lEq)+gEq),    sqrtPOverMu*(fEq/w)*(kEq*sy.sin(lEq)-hEq*sy.cos(lEq))],
                    [0.0,0.0,sqrtPOverMu*(s2*sy.cos(lEq)/(2*w))],
                    [0.0,0.0,sqrtPOverMu*(s2*sy.sin(lEq)/(2*w))],
                    [0.0,0.0,sqrtPOverMu*(hEq*sy.sin(lEq)-kEq*sy.cos(lEq))/w]])
        return B        
    def __getitem__(self, i) :
        if i == 0 :
            return self.SemiParameter
        elif i == 1 :
            return self.EccentricityCosTermF
        elif i == 2:
            return self.EccentricitySinTermG
        elif i == 3 :
            return self.InclinationCosTermH
        elif i == 4:
            return self.InclinationSinTermK
        elif i == 5:
            return self.TrueLongitude
        raise IndexError("Index {i} is too high")

    def __len__(self):
        return 6


def ConvertKeplerianToEquinoctial(keplerianElements : Keplerian.KeplerianElements, nonModdedLongitude : Optional[bool]= True) ->ModifiedEquinoctialElements :
    a = keplerianElements.SemiMajorAxis
    e = keplerianElements.Eccentricity
    i = keplerianElements.Inclination
    w = keplerianElements.ArgumentOfPeriapsis
    raan = keplerianElements.RightAscensionOfAscendingNode
    ta = keplerianElements.TrueAnomaly

    per = a*(1-e**2)
    f = e*sy.cos(w+raan)
    g = e*sy.sin(w+raan)
    
    h = sy.tan(i/2)*sy.cos(raan)
    k = sy.tan(i/2)*sy.sin(raan)
    if nonModdedLongitude :
        l = w+raan+ta
    else :
        l = ((w+raan+ta) % (2*math.pi))
        if isinstance(l, sy.Expr):
            l = l.simplify()

    return ModifiedEquinoctialElements(per, f, g, h, k, l, keplerianElements.GravitationalParameter)

def CreateSymbolicElements(elementOf :Optional[SymbolOrNumber]= None, mu : Optional[SymbolOrNumber] = None) -> ModifiedEquinoctialElements : #TODO named arguments of mu and element of
    if mu is None:
        mu = sy.Symbol(r'\mu', positive=True, real=True)

    if(elementOf == None) : 
        p = sy.Symbol('p', positive=True, real=True)
        f = sy.Symbol('f', real=True)
        g = sy.Symbol('g', real=True)
        h = sy.Symbol('h', real=True)
        k= sy.Symbol('k', real=True)
        l = sy.Symbol('L', real=True)
    else :
        p = sy.Function('p', positive=True, real=True)(elementOf)
        f = sy.Function('f', real=True)(elementOf)
        g = sy.Function('g', real=True)(elementOf)
        h = sy.Function('h', real=True)(elementOf)
        k= sy.Function('k', real=True)(elementOf)
        l = sy.Function('L', real=True)(elementOf)
    
    return ModifiedEquinoctialElements(p, f, g, h, k, l, mu)

from abc import ABC, abstractmethod
class EquinoctialElementsHalfI(ABC):

    @staticmethod
    @abstractmethod
    def CreateSymbolicElements(elementOf :Optional[SymbolOrNumber]= None, mu : Optional[SymbolOrNumber] = None) ->EquinoctialElementsHalfI : 
        pass

    @staticmethod
    def CreateEphemeris(equinoctialElementsList : List[EquinoctialElementsHalfI]) -> List[MotionCartesian] :
        motions = []
        for equi in equinoctialElementsList :
            motions.append(equi.ToCartesian())
        return motions

    @abstractmethod
    def ToCartesian(self)->MotionCartesian:
        pass

    @abstractmethod
    def CreatePerturbationMatrix(self, subsDict : Optional[Dict[sy.Expr, SymbolOrNumber]]= None) ->sy.Matrix:
        pass

    @abstractmethod
    def UnperturbedLongitudeTimeDerivative(self) -> SymbolOrNumber:
        pass

    def LongitudeEquationsOfMotionInMatrixForm(self, otherForceVector)->sy.Expr:
        return self.CreatePerturbationMatrix() * otherForceVector + sy.Matrix([[0],[0],[1]])*self.UnperturbedLongitudeTimeDerivative()

    @staticmethod
    def CreateFgwToInertialAxesStatic(p:SymbolOrNumber, q:SymbolOrNumber) -> sy.Matrix:
        multiplier = 1/(1+p**2+q**2)
        fm = multiplier*sy.Matrix([[1-p**2+q**2], [2*p*q], [-2*p]]) # first point of aries
        gm = multiplier*sy.Matrix([[2*p*q], [1+p**2-q**2], [2*q]]) # completes triad
        wm = multiplier*sy.Matrix([[2*p], [-2*q], [1-p**2-q**2]]) # orbit normal

        return sy.Matrix([[fm[0,0], fm[1,0], fm[2,0]],[gm[0,0],gm[1,0],gm[2,0] ],[wm[0,0], wm[1,0], wm[2,0]]]).transpose()
    
    def CreateFgwToInertialAxes(self) -> sy.Matrix:
        p = self.InclinationSinTermP
        q = self.InclinationCosTermQ
        return EquinoctialElementsHalfIMeanLongitude.CreateFgwToInertialAxesStatic(p, q)

    @property
    def SemiMajorAxis(self) :
        return self._semiMajorAxis

    @property
    def EccentricitySinTermH(self) :
        return self._eccentricitySinTermH

    @property
    def EccentricityCosTermK(self) :
        return self._eccentricityCosTermK

    @property
    def InclinationSinTermP(self) :
        return self._inclinationSinTermP

    @property
    def InclinationCosTermQ(self) :
        return self._inclinationCosTermQ

    @property
    def GravitationalParameter(self) :
        return self._gravitationalParameter

    def MeanMotion(self):
        a = self.SemiMajorAxis
        return sy.sqrt(mu/(a**3))

    def ToKeplerianIgnoringAnomaly(self) -> Keplerian.KeplerianElements :
        h = self.EccentricitySinTermH
        k = self.EccentricityCosTermK
        p = self.InclinationSinTermP
        q = self.InclinationCosTermQ
        a = self.SemiMajorAxis
        
        e = sy.sqrt(h*h+k*k)
        # it is not clear to me if the atan2 in the MME PDF passes in x,y or y,x (y,x is right for sy)
        i = sy.atan(sy.sqrt(p*p+q*q))
        w = sy.atan2(k, h) - sy.atan2(q,p)
        if math.isnan(w):
            w=0.0
        raan = sy.atan2(q, p)
        
        return Keplerian.KeplerianElements(a, e, i, w, raan, 0, self.GravitationalParameter)        

class EquinoctialElementsHalfITrueLongitude(EquinoctialElementsHalfI) :
    def __init__(self, a:SymbolOrNumber, h:SymbolOrNumber, k:SymbolOrNumber, p:SymbolOrNumber, q:SymbolOrNumber, trueLongitude:SymbolOrNumber, mu:SymbolOrNumber) :
        self._semiMajorAxis = a
        self._eccentricitySinTermH = h
        self._eccentricityCosTermK = k
        self._inclinationSinTermP = p
        self._inclinationCosTermQ = q
        self._trueLongitude = trueLongitude
        self._gravitationalParameter = mu

        self.BetaSy = sy.Function(r'\beta')(h, k)
        self.NSy = sy.Function('n')(a)
        self.ROverASy = sy.Function(r'\frac{r}{a}')(h, k, self.TrueLongitude)# assuming true longitude

        self.Beta = 1/(1+sy.sqrt(1-h**2-k**2))
        self.N = sy.sqrt(mu/(a**3))
        self.ROverA= (1-h**2-k**2)/(1+h*sy.sin(trueLongitude)+k*sy.cos(trueLongitude))

    @property
    def TrueLongitude(self):
        return self._trueLongitude

    def ToCartesian(self) -> MotionCartesian :
        tempKep = self.ToKeplerianIgnoringAnomaly()        
        ta = self.TrueLongitude - tempKep.ArgumentOfPeriapsis-tempKep.RightAscensionOfAscendingNode
        return tempKep.ToInertialMotionCartesianOverridingTrueAnomaly(ta)

    def ConvertToModifiedEquinoctial(self) ->ModifiedEquinoctialElements:
        eSq = self.EccentricityCosTermK**2+self.EccentricitySinTermH**2
        param = self.SemiMajorAxis * (1.0 - eSq)
        return ModifiedEquinoctialElements(param, self.EccentricityCosTermK, self.EccentricitySinTermH, self.InclinationCosTermQ, self.InclinationSinTermP, self.TrueLongitude, self.GravitationalParameter)

    @staticmethod
    def FromModifiedEquinoctialElements(mee : ModifiedEquinoctialElements)->EquinoctialElementsHalfITrueLongitude :
        eSq = mee.EccentricityCosTermF**2+mee.EccentricitySinTermG**2
        sma = mee.SemiParameter/(1-eSq)
        return EquinoctialElementsHalfITrueLongitude(sma, mee.EccentricitySinTermG, mee.EccentricityCosTermF, mee.InclinationSinTermK, mee.InclinationCosTermH, mee.TrueLongitude, mee.GravitationalParameter)

    @staticmethod
    def CreateSymbolicElements(elementOf :Optional[SymbolOrNumber]= None, mu : Optional[SymbolOrNumber] = None) ->EquinoctialElementsHalfITrueLongitude : 
        if(elementOf == None) : 
            a = sy.Symbol('a', real=True)
            h = sy.Symbol('h', real=True)
            k = sy.Symbol('k', real=True)
            p = sy.Symbol('p', real=True)
            q = sy.Symbol('q', real=True)
            l = sy.Symbol('L', real=True)
        else :
            a = sy.Function('a', positive=True)(elementOf)
            h = sy.Function('h', real=True)(elementOf)
            k = sy.Function('k', real=True)(elementOf)
            p = sy.Function('p', real=True)(elementOf)
            q= sy.Function('q', real=True)(elementOf)
            l = sy.Function('L', real=True)(elementOf)
        if mu is None :
            mu = cast(SymbolOrNumber, sy.Symbol(r'\mu', positive=True, real=True))

        return EquinoctialElementsHalfITrueLongitude(a, h, k, p, q, l, mu)
    
    def CreatePerturbationMatrix(self, subsDict : Optional[Dict[sy.Expr, SymbolOrNumber]]= None) ->sy.Matrix:
        if subsDict == None:
            subsDict = {} #type: ignore
        p = self.InclinationSinTermP
        q = self.InclinationCosTermQ
        h = self.EccentricitySinTermH
        k = self.EccentricityCosTermK
        mu = self.GravitationalParameter
        a = self.SemiMajorAxis
        l = self.TrueLongitude

        G = sy.sqrt(1-h**2-k**2)
        K = (1+p**2+q**2)
        n = self.NSy
        subsDict[n] = self.N #type: ignore
        GExp = sy.sqrt(1-h*h-k*k)
        G = sy.Function("G")(h, k)
        subsDict[G] = GExp#type: ignore

        KExp = K
        K = sy.Function("K")(p, q)
        subsDict[K] = KExp#type: ignore

        r = sy.Function('r')(a)
        subsDict[r] = self.ROverA*a#type: ignore

        sl = sy.sin(self.TrueLongitude)
        cl = sy.cos(self.TrueLongitude)
        #u is radial, intrack, out of plane, AKA r, theta, h

        onephspkc = 1+h*sl+k*cl
        aDotMult = (2/(n*G))
        b11 = aDotMult*(k*sl-h*cl) #aDot in r direction
        b12 = aDotMult*(onephspkc) #aDot in theta direction
        b13 = 0  # a dot in h direction, you get the pattern...

        hDotMult = G/(n*a*onephspkc)
        b21 = hDotMult*(-(onephspkc)*cl)
        b22 = hDotMult*((h+(2+h*sl+k*cl)*sl))
        b23 = -hDotMult*(k*(p*cl-q*sl))

        kDotMult = G/(n*a*onephspkc)
        b31 = kDotMult*((onephspkc)*sl)
        b32 = kDotMult*((k+(2+h*sl+k*cl)*cl))
        b33 = kDotMult*(h*(p*cl-q*sl)) 

        pDotMult = G/(2*n*a*onephspkc)
        b41 = 0
        b42 = 0
        b43 = pDotMult*K*sl
        
        qDotMult = G/(2*n*a*onephspkc)
        b51 = 0
        b52 = 0
        b53 = qDotMult*K*cl
        
        b61 = 0
        b62 = 0
        b63 = (G*(q*sl-p*cl))/(n*a*onephspkc)
        #b63 = r*(q*sl-p*cl)/(n*G*a**2)

        #M = sy.Matrix([[m11, m12, m13], [m21, m22, m23],[m31, m32, m33],[m41, m42, m43],[m51, m52, m53]])
        B = sy.Matrix([[b11, b12, b13], [b21, b22, b23],[b31, b32, b33],[b41, b42, b43],[b51, b52, b53],[b61, b62, b63]])   
        return B     

    def UnperturbedLongitudeTimeDerivative(self, subsDict : Optional[Dict[sy.Expr, SymbolOrNumber]]=None) ->sy.Expr :
        p = self.InclinationSinTermP
        q = self.InclinationCosTermQ
        h = self.EccentricitySinTermH
        k = self.EccentricityCosTermK
        mu = self.GravitationalParameter
        a = self.SemiMajorAxis
        l = self.TrueLongitude 
        n = self.N
        sl = sy.sin(l)
        cl = sy.cos(l)
        return (n*(1+h*sl+k*cl)**2)/(1-h**2-k**2)**(3/2)



class EquinoctialElementsHalfIMeanLongitude(EquinoctialElementsHalfI) :
    def __init__(self, a:SymbolOrNumber, h:SymbolOrNumber, k:SymbolOrNumber, p:SymbolOrNumber, q:SymbolOrNumber, meanLongitude:SymbolOrNumber, mu:SymbolOrNumber) :
        self._semiMajorAxis = a
        self._eccentricitySinTermH = h
        self._eccentricityCosTermK = k
        self._inclinationSinTermP = p
        self._inclinationCosTermQ = q
        self._meanLongitude = meanLongitude
        self._gravitationalParameter = mu

        self.BetaSy = sy.Function(r'\beta')(h, k)
        self.NSy = sy.Function('n')(a)

        self.Beta = 1/(1+sy.sqrt(1-h**2-k**2))
        self.N = sy.sqrt(mu/(a**3))

    @property
    def MeanLongitude(self):
        return self._meanLongitude

    def ToCartesian(self) -> MotionCartesian :
        tempKep = self.ToKeplerianIgnoringAnomaly()        
        ma = self.MeanLongitude - tempKep.ArgumentOfPeriapsis-tempKep.RightAscensionOfAscendingNode
        ta = Keplerian.TrueAnomalyFromMeanAnomaly(ma, tempKep.Eccentricity)
        return tempKep.ToInertialMotionCartesianOverridingTrueAnomaly(ta)

    @staticmethod
    def CreateSymbolicElements(elementOf :Optional[SymbolOrNumber]= None, mu : Optional[SymbolOrNumber] = None) ->EquinoctialElementsHalfIMeanLongitude : 
        if(elementOf == None) : 
            a = sy.Symbol('a', real=True)
            h = sy.Symbol('h', real=True)
            k = sy.Symbol('k', real=True)
            p = sy.Symbol('p', real=True)
            q = sy.Symbol('q', real=True)
            m = sy.Symbol('m', real=True)
        else :
            a = sy.Function('a', positive=True)(elementOf)
            h = sy.Function('h', real=True)(elementOf)
            k = sy.Function('k', real=True)(elementOf)
            p = sy.Function('p', real=True)(elementOf)
            q= sy.Function('q', real=True)(elementOf)
            m = sy.Function('m', real=True)(elementOf)
        if mu is None :
            mu = cast(SymbolOrNumber, sy.Symbol(r'\mu', positive=True, real=True))

        return EquinoctialElementsHalfIMeanLongitude(a, h, k, p, q, m, mu)
    


    def CreatePerturbationMatrix(self, subsDict : Optional[Dict[sy.Expr, SymbolOrNumber]]= None) ->sy.Matrix:
        a = self.SemiMajorAxis
        h = self.EccentricitySinTermH
        k = self.EccentricityCosTermK
        p = self.InclinationSinTermP
        q = self.InclinationCosTermQ
        l = self.MeanLongitude
        sl = sy.sin(l)
        cl = sy.cos(l)
        #page 80
        # f = l-k*
        # mu = self.GravitationalParameter
        # n = self.N
        # G = sy.sqrt(1-h*h-k*k)
        # K = 3 #TODO!!!
        
        # beta = 1/(1+G)

        # b11 = 2*(k*sl-h*cl)/(G*n) #this ended up being the true longitude implimentation
        # b12 = 2*G*a/(n*r)
        # b13 = 0

        # b21 = -1*G*cl/(n*a)
        # b22 = r*(h+sl)/(n*a*a*G) + G*sl/(n*a)
        # b23 = r*k*(p*cl-q*sl)/(n*a*a*G)

        # b31 = G*sl/(n*a)
        # b32 = r*(k+cl)/(n*a*a*G)+G*sl/(n*a)
        # b33 = r*h*(p*cl-q*sl)/(n*a*a*G)

        # b41 = 0
        # b42 = 0
        # b43 = r*K*sl/(2*n*a*a*G)

        # b51 = 0
        # b52 = 0
        # b53 = r*K*cl/(2*n*a*a*G)

        # b61 = -1*(1-beta)*(h*sl+k*cl)/(n*a) -2*r/(n*a*a)
        # b62 = -1*(1-beta)*(h*cl-k*sl)*(1+r/(a*G*G))/(n*a)
        # b63 = -1*r*(p*cl-q*sl)/(n*a*a*G)

        # return sy.Matrix([[b11, b12, b13], [b21, b22, b23], [b31, b32, b33], [b41, b42, b43], [b51, b52, b53], [b61, b62, b63]])#TODO
        return sy.Matrix([[0]]) #TODO


    def UnperturbedLongitudeTimeDerivative(self) -> SymbolOrNumber:
        return self.N



class EquinoctialElementsHalfIEccentricLongitude(EquinoctialElementsHalfI) :
    def __init__(self, a:SymbolOrNumber, h:SymbolOrNumber, k:SymbolOrNumber, p:SymbolOrNumber, q:SymbolOrNumber, eccentricLongitude:SymbolOrNumber, mu:SymbolOrNumber) :
        self._semiMajorAxis = a
        self._eccentricitySinTermH = h
        self._eccentricityCosTermK = k
        self._inclinationSinTermP = p
        self._inclinationCosTermQ = q
        self._eccentricLongitude = eccentricLongitude
        self._gravitationalParameter = mu

        self.BetaSy = sy.Function(r'\beta')(h, k)
        self.NSy = sy.Function('n')(a)

        self.Beta = 1/(1+sy.sqrt(1-h**2-k**2))
        self.N = sy.sqrt(mu/(a**3))

    @property
    def EccentricLongitude(self):
        return self._eccentricLongitude

    def ToCartesian(self) -> MotionCartesian :
        tempKep = self.ToKeplerianIgnoringAnomaly()        
        ea = self.EccentricLongitude - tempKep.ArgumentOfPeriapsis-tempKep.RightAscensionOfAscendingNode
        ta = Keplerian.TrueAnomalyFromMeanAnomaly(ea, tempKep.Eccentricity)
        return tempKep.ToInertialMotionCartesianOverridingTrueAnomaly(ta)

    @staticmethod
    def CreateSymbolicElements(elementOf :Optional[SymbolOrNumber]= None, mu : Optional[SymbolOrNumber] = None) ->EquinoctialElementsHalfIEccentricLongitude : 
        if(elementOf == None) : 
            a = sy.Symbol('a', real=True)
            h = sy.Symbol('h', real=True)
            k = sy.Symbol('k', real=True)
            p = sy.Symbol('p', real=True)
            q = sy.Symbol('q', real=True)
            f = sy.Symbol('F', real=True)
        else :
            a = sy.Function('a', positive=True)(elementOf)
            h = sy.Function('h', real=True)(elementOf)
            k = sy.Function('k', real=True)(elementOf)
            p = sy.Function('p', real=True)(elementOf)
            q= sy.Function('q', real=True)(elementOf)
            f = sy.Function('F', real=True)(elementOf)
        if mu is None :
            mu = cast(SymbolOrNumber, sy.Symbol(r'\mu', positive=True, real=True))

        return EquinoctialElementsHalfIEccentricLongitude(a, h, k, p, q, f, mu)
    


    def CreatePerturbationMatrix(self, subsDict : Optional[Dict[sy.Expr, SymbolOrNumber]]= None) ->sy.Matrix:
        a = self.SemiMajorAxis
        h = self.EccentricitySinTermH
        k = self.EccentricityCosTermK
        p = self.InclinationSinTermP
        q = self.InclinationCosTermQ
        l = self.EccentricLongitude
        mu = self.GravitationalParameter

        #Page 105
        return sy.Matrix([[0]])#TODO


    def UnperturbedLongitudeTimeDerivative(self) -> SymbolOrNumber:
        return -1