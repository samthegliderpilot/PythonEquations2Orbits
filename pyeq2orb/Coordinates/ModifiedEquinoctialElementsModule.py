from __future__ import annotations
import sympy as sy
import math
from pyeq2orb.Coordinates.KeplerianModule import KeplerianElements
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

    def ToKeplerian(self) -> KeplerianElements :
        l = self.TrueLongitude
        f = self.EccentricityCosTermF
        g = self.EccentricitySinTermG
        h = self.InclinationCosTermH
        k = self.InclinationSinTermK
        p = self.SemiParameter

        a = p/(1-f**2-g**2)
        e = sy.sqrt(f*f+g*g)
        # it is not clear to me if the atan2 in the MME PDF passes in x,y or y,x (y,x is right for sy)
        i = sy.atan2(1-h*h-k*k, 2*sy.sqrt(h*h+k*k))
        w = sy.atan2(f*h+g*k, g*h-f*k)
        raan = sy.atan2(h, k)
        ta = l-raan+w
        return KeplerianElements(a, e, i, w, raan, ta, self.GravitationalParameter)
    
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
        return ConvertKeplerianToEquinoctial(KeplerianElements.FromMotionCartesian(motion, gravitationalParameter))

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
        raise Exception('Index of {i} is too high')

    def __len__(self):
        return 6

def ConvertKeplerianToEquinoctial(keplerianElements : KeplerianElements, nonModdedLongitude : Optional[bool]= True) ->ModifiedEquinoctialElements :
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



# def TwoBodyGravityForceOnElements(elements : ModifiedEquinoctialElements, useSymbolsForAuxiliaryElements = False) ->sy.Matrix:
#     if useSymbolsForAuxiliaryElements :
#         w = elements.WSymbol
#     else :
#         w = elements.W
#     return sy.Matrix([[0],[0],[0],[0],[0],[sy.sqrt(elements.GravitationalParameter*elements.SemiParameter)*(w/elements.SemiParameter)**2]])




class EquinoctialElementsHalfI :
    def __init__(self, a:SymbolOrNumber, h:SymbolOrNumber, k:SymbolOrNumber, p:SymbolOrNumber, q:SymbolOrNumber, longitude:SymbolOrNumber, mu:SymbolOrNumber) :
        self.SemiMajorAxis = a
        self.EccentricitySinTermH = h
        self.EccentricityCosTermK = k
        self.InclinationSinTermP = p
        self.InclinationCosTermQ = q
        self.Longitude = longitude # consider how things would work if longitude was left ambiguous
        self.GravitationalParameter = mu

    def ConvertToModifiedEquinoctial(self) ->ModifiedEquinoctialElements:
        eSq = self.EccentricityCosTermK**2+self.EccentricitySinTermH**2
        param = self.SemiMajorAxis * (1.0 - eSq)
        return ModifiedEquinoctialElements(param, self.EccentricityCosTermK, self.EccentricitySinTermH, self.InclinationCosTermQ, self.InclinationSinTermP, self.Longitude, self.GravitationalParameter)

    @staticmethod
    def FromModifiedEquinoctialElements(mee : ModifiedEquinoctialElements)->EquinoctialElementsHalfI :
        eSq = mee.EccentricityCosTermF**2+mee.EccentricitySinTermG**2
        sma = mee.SemiParameter/(1-eSq)
        return EquinoctialElementsHalfI(sma, mee.EccentricitySinTermG, mee.EccentricityCosTermF, mee.InclinationSinTermK, mee.InclinationCosTermH, mee.TrueLongitude, mee.GravitationalParameter)

    @staticmethod
    def CreateSymbolicElements(elementOf :Optional[SymbolOrNumber]= None, mu : Optional[SymbolOrNumber] = None) ->EquinoctialElementsHalfI : 
        if(elementOf == None) : 
            a = sy.Symbol('a', real=True)
            h = sy.Symbol('h', real=True)
            k = sy.Symbol('k', real=True)
            p = sy.Symbol('p', real=True)
            q = sy.Symbol('q', real=True)
            l = sy.Symbol('l', real=True)
        else :
            a = sy.Function('a', positive=True)(elementOf)
            h = sy.Function('h', real=True)(elementOf)
            k = sy.Function('k', real=True)(elementOf)
            p = sy.Function('p', real=True)(elementOf)
            q= sy.Function('q', real=True)(elementOf)
            l = sy.Function('l', real=True)(elementOf)
        if mu is None :
            mu = cast(SymbolOrNumber, sy.Symbol(r'\mu', positive=True, real=True))

        return EquinoctialElementsHalfI(a, h, k, p, q, l, mu)
    
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
        return EquinoctialElementsHalfI.CreateFgwToInertialAxesStatic(p, q)

    def RadiusInFgw(self, eccentricLongitude : SymbolOrNumber, subsDict : Optional[dict[sy.Expr, SymbolOrNumber]]=None) -> List[SymbolOrNumber]:
        p = self.InclinationSinTermP
        q = self.InclinationCosTermQ
        h = self.EccentricitySinTermH
        k = self.EccentricityCosTermK
        mu = self.GravitationalParameter
        a = self.SemiMajorAxis
        f = eccentricLongitude

        b = 1/(1+sy.sqrt(1-h**2-k**2))
        n = sy.sqrt(mu/a)
        rOverA = 1-k*sy.cos(f)-h*sy.sin(f)
        if(subsDict != None) :          
            subsDict = cast(dict, subsDict)  
            bSy = sy.Function(r'\beta')(h, k)
            nSy = sy.Function('n')(a)
            rOverASy = sy.Function(r'\frac{r}{a}')(k, f, h)
            subsDict[bSy] = b
            subsDict[nSy] = n
            subsDict[rOverASy] = rOverA
            b = bSy
            n = nSy
            rOverA = rOverASy

        x1 = a*((1-h**2*b)*sy.cos(f)+h*k*b*sy.sin(f)-k)
        x2 = a*((1-k**2*b)*sy.sin(f)+h*k*b*sy.cos(f)-h)

        return [x1, x2]

    def VelocityInFgw(self, eccentricLongitude : SymbolOrNumber, subsDict : Optional[Dict[sy.Expr, SymbolOrNumber]]=None) -> List[SymbolOrNumber]:
        p = self.InclinationSinTermP
        q = self.InclinationCosTermQ
        h = self.EccentricitySinTermH
        k = self.EccentricityCosTermK
        mu = self.GravitationalParameter
        a = self.SemiMajorAxis
        f = eccentricLongitude

        b = 1/(1+sy.sqrt(1-h**2-k**2))
        n = sy.sqrt(mu/(a**3))
        rOverA = 1-k*sy.cos(f)-h*sy.sin(f)
        if(subsDict != None) :      
            subsDict = cast(dict, subsDict)  
            bSy = sy.Function(r'\beta')(h, k)
            nSy = sy.Function('n')(a)
            rOverASy = sy.Function(r'\frac{r}{a}')(k, f, h)
            subsDict[bSy] = b
            subsDict[nSy] = n
            subsDict[rOverASy] = rOverA
            b = bSy
            n = nSy
            rOverA = rOverASy
        x1Dot = (n*a/rOverA)*(h*k*b*sy.cos(f)-(1-(h**2)*b*sy.sin(f)))
        x2Dot = (n*a/rOverA)*((1-(k**2)*b*sy.cos(f)-h*k*b*sy.sin(f)))

        return [x1Dot, x2Dot]

    @staticmethod
    def InTermsOfX1And2AndTheirDots(x1:SymbolOrNumber, x2:SymbolOrNumber, x3:SymbolOrNumber, x1Dot:SymbolOrNumber, x2Dot:SymbolOrNumber, x3Dot:SymbolOrNumber, p:SymbolOrNumber, q:SymbolOrNumber, mu:SymbolOrNumber, subsDict :Optional[Dict[sy.Symbol, SymbolOrNumber]] = None, longitude :Optional[SymbolOrNumber]= 0.0) ->EquinoctialElementsHalfI:
        rotMat = sy.Matrix.eye(3,3)# EquinoctialElementsHalfI.CreateFgwToInertialAxesStatic(p, q)
        # if(subsDict != None) :
        #     rotSymbol = sy.Matrix.zeros(3,3)
        #     fgw = ["f", "g", "w"]
        #     xyz = ["x","y","z"]
        #     for r in range(0, 3) :                
        #         for c in range(0, 3) :
        #             rotSymbol[r,c] = sy.Function(str(fgw[r]) + "_{" + str(xyz[c]) +"}", real=True)(p, q)
        #             subsDict[rotSymbol[r,c]] = rotMat[r,c]
        #     rotMat = rotSymbol
        xVec = rotMat * Cartesian(x1, x2, x3)
        xDotVec = rotMat * Cartesian(x1Dot, x2Dot, x3Dot)
        sma = (1/sy.sqrt(x1*x1+x2**2) - (x1Dot**2+x2Dot**2)/mu)**(-1)
        rXv = xVec.cross(xDotVec)
        eVec = (rXv).cross(xDotVec)/mu
        magSymbol =rXv.Magnitude()#  sy.Function("|rXv|")(rXv)
        wVec = rXv/magSymbol

        pOut = wVec[0]/(1+wVec[2])
        qOut = wVec[1]/(1+wVec[2])

        hOut = eVec.dot(Cartesian(rotMat[1,0], rotMat[1,1], rotMat[1,2]))
        kOut = eVec.dot(Cartesian(rotMat[0,0], rotMat[0,1], rotMat[0,2]))
        if longitude == None:
            longitude = 0.0
        lon : SymbolOrNumber = longitude #type:ignore
        return EquinoctialElementsHalfI(sma, hOut, kOut, pOut, qOut, lon, mu)
    # def ToFgwRotationMatrix(self) :
    #     p = self.InclinationSinTermP
    #     q = self.InclinationCosTermQ

    #     denom = 1+p*p+q*q
    #     f1 = (1/denom)*(1-p*p+q*q)
    #     f2 = 2*p*q
    #     f3 = -2*p

    #     g1 = f2
    #     g2 = 1+p*p-q*q
    #     g3 = 2*q

    #     w1 = 2*p
    #     w2 = -2*q
    #     w3 = 1-p*p-q*q

    #     return Matrix([[]])