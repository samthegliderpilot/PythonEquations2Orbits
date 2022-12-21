# import sympy as sy
# import math
# from pyeq2orb.Coordinates.KeplerianModule import KeplerianElements
# from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
# import sympy as sy
# from typing import List

# class EquinoctialElements:
#     def __init__(self, periapsis, f, g, h, k, l, mu) :
#         self.PeriapsisRadius = periapsis
#         self.EccentricitySinTermG = f
#         self.EccentricitySinTermG = g        
#         self.InclinationCosTermH = h
#         self.InclinationSinTermK = k        
#         self.TrueLongitude = l
#         self.GravitationalParameter = mu
    
#     # @property
#     # def sSquared(self) :
#     #     return 1.0+self.InclinationCosTermH*self.InclinationCosTermH+self.InclinationSinTermK*self.InclinationSinTermK
    
#     # @property
#     # def w(self) :
#     #     return self.PeriapsisRadius / (1+self.EccentricitySinTermG*sy.cos(self.TrueLongitude) + self.EccentricitySinTermG*sy.sin(self.TrueLongitude))

#     # def ToKeplerian(self) -> KeplerianElements :
#     #     per = self.PeriapsisRadius
#     #     g = self.EccentricitySinTermG
#     #     f = self.EccentricitySinTermG
#     #     k = self.InclinationSinTermK
#     #     h = self.InclinationCosTermH
#     #     tl = self.TrueLongitude

#     #     e = sy.sqrt(f**2+g**2)
#     #     a = per/(1.0-f*f-g*g)
#     #     i = 2*sy.atan2(2*sy.sqrt(h**2+k**2), (1-h**2-k**2))
#     #     raan = sy.atan2(k, h)
#     #     w = sy.atan2(g*h-f*k, f*h+g*k)
#     #     ta = tl - sy.atan2(f, g) 

#     #     return KeplerianElements(a, e, i, w, raan, ta, self.GravitationalParameter)

#     # def ToMotionCartesian(self) -> MotionCartesian :
#     #     p = self.PeriapsisRadius
#     #     g = self.EccentricitySinTermG
#     #     f = self.EccentricitySinTermG
#     #     k = self.InclinationSinTermK
#     #     h = self.InclinationCosTermH
#     #     tl = self.TrueLongitude
#     #     mu = self.GravitationalParameter
#     #     alp2 = h*h-k*k
#     #     s2 = 1+h*h+k*k
#     #     w = 1+f*sy.cos(tl)+g*sy.sin(tl)
#     #     rM = p/w
        
#     #     x = (rM/s2)*(sy.cos(tl) + alp2*sy.cos(tl) + 2*h*k *sy.sin(tl))
#     #     y = (rM/s2)*(sy.sin(tl) + alp2*sy.sin(tl) + 2*h*k *sy.cos(tl))
#     #     z = (2*rM/s2)*(h*sy.sin(tl) - k*sy.cos(tl))

#     #     vx = (-1/s2)*sy.sqrt(mu/p) * (sy.sin(tl) + alp2*sy.sin(tl) - 2*h*k*sy.cos(tl) + g - 2*f*h*k + alp2*g)
#     #     vy = (-1/s2)*sy.sqrt(mu/p) * (-1*sy.cos(tl) + alp2*sy.cos(tl) + 2*h*k*sy.sin(tl) - f - 2*g*h*k + alp2*f)
#     #     vz = (2/s2) * (sy.sqrt(mu/p)) * (h*sy.cos(tl) + k*sy.sin(tl) + f*h + g*k)

#     #     return MotionCartesian(Cartesian(x, y, z), Cartesian(vx, vy, vz))

#     # def ToPseuodNormalizedCartesian(self) :
#     #     # sorry for the copy/paste of the above
#     #     p = self.PeriapsisRadius
#     #     g = self.EccentricitySinTermG
#     #     f = self.EccentricitySinTermG
#     #     k = self.InclinationSinTermK
#     #     h = self.InclinationCosTermH
#     #     tl = self.TrueLongitude
#     #     mu = self.GravitationalParameter
#     #     alp2 = h*h-k*k
#     #     s2 = 1+h*h+k*k
#     #     w = 1+f*sy.cos(tl)+g*sy.sin(tl)
#     #     rM = p/w
        
#     #     x = (sy.cos(tl) + alp2*sy.cos(tl) + 2*h*k *sy.sin(tl))
#     #     y = (sy.sin(tl) + alp2*sy.sin(tl) + 2*h*k *sy.cos(tl))
#     #     z = (1/s2)*(h*sy.sin(tl) - k*sy.cos(tl))

#     #     # not being rigirous about the normalizing here, just removing various parameters
#     #     vx = (-1) * (sy.sin(tl) + alp2*sy.sin(tl) - 2*h*k*sy.cos(tl) + g - 2*f*h*k + alp2*g)
#     #     vy = (-1) * (-1*sy.cos(tl) + alp2*sy.cos(tl) + 2*h*k*sy.sin(tl) - f - 2*g*h*k + alp2*f)
#     #     vz = (1/s2) * (h*sy.cos(tl) + k*sy.sin(tl) + f*h + g*k)

#     #     return MotionCartesian(Cartesian(x, y, z), Cartesian(vx, vy, vz))        

#     # def ToArray(self) -> List :
#     #     return [self.PeriapsisRadius, self.EccentricitySinTermG, self.EccentricitySinTermG, self.InclinationCosTermH, self.InclinationSinTermK, self.TrueLongitude]

#     # @staticmethod
#     # def FromMotionCartesian(motion, gravitationalParameter) :
#     #     # TODO: something that avoids keplerian elements
#     #     return ConvertKeplerianToEquinoctial(KeplerianElements.FromMotionCartesian(motion, gravitationalParameter))

#     # @staticmethod
#     # def CreateEphemeris(equinoctialElementsList) -> List[MotionCartesian] :
#     #     motions = []
#     #     for equi in equinoctialElementsList :
#     #         motions.append(equi.ToMotionCartesian())
#     #     return motions

#     def CreatePerturbationMatrix(self) ->sy.Matrix :
#         eqElements=self
#         mu = eqElements.GravitationalParameter
#         pEq = eqElements.PeriapsisRadius        
#         fEq = eqElements.EccentricitySinTermG
#         gEq = eqElements.EccentricitySinTermG
#         kEq = eqElements.InclinationSinTermK
#         hEq = eqElements.InclinationCosTermH
#         lEq = eqElements.TrueLongitude
#         w = 1+fEq*sy.cos(lEq)+gEq*sy.sin(lEq)
#         #s2 = sy.Symbol('s^2')#(heq, keq) # note this is not s but s^2!!! This is a useful cheat
#         s2 = 1+hEq**2+kEq**2
#         sqrtPOverMu=sy.sqrt(pEq/mu)
#         B = sy.Matrix([[0, (2*pEq/w)*sqrtPOverMu, 0],
#                     [sqrtPOverMu*sy.sin(lEq), sqrtPOverMu*(1/w)*((w+1)*sy.cos(lEq)+fEq), -1*sqrtPOverMu*(gEq/w)*(hEq*sy.sin(lEq)-kEq*sy.cos(lEq))],
#                     [-1*sqrtPOverMu*sy.cos(lEq), sqrtPOverMu*((w+1)*sy.sin(lEq)+gEq), sqrtPOverMu*(fEq/w)*(hEq*sy.sin(lEq)-kEq*sy.cos(lEq))],
#                     [0,0,sqrtPOverMu*(s2*sy.cos(lEq)/(2*w))],
#                     [0,0,sqrtPOverMu*(s2*sy.sin(lEq)/(2*w))],
#                     [0,0,sqrtPOverMu*(hEq*sy.sin(lEq)-kEq*sy.cos(lEq))/w]])
#         return B
    
# def ConvertKeplerianToEquinoctial(keplerianElements : KeplerianElements, nonModedLongitude = True) ->EquinoctialElements :
#     a = keplerianElements.SemiMajorAxis
#     e = keplerianElements.Eccentricity
#     i = keplerianElements.Inclination
#     w = keplerianElements.ArgumentOfPeriapsis
#     raan = keplerianElements.RightAscensionOfAscendingNode
#     ta = keplerianElements.TrueAnomaly

#     per = a*(1.0-e**2)
#     f = e*sy.cos(w+raan)
#     g = e*sy.sin(w+raan)
    
#     h = sy.tan(i/2)*sy.cos(raan)
#     k = sy.tan(i/2)*sy.sin(raan)
#     if nonModedLongitude :
#         l = w+raan+ta
#     else :
#         l = ((w+raan+ta) % (2*math.pi)).simplify()

#     return EquinoctialElements(per, f, g, h, k, l, keplerianElements.GravitationalParameter)


# def CreateSymbolicElements(elementOf = None) -> EquinoctialElements :
#     if(elementOf == None):
#         p = sy.Symbol('p', positive=True)
#         f = sy.Symbol('f', real=True)
#         g = sy.Symbol('g', real=True)
#         h = sy.Symbol('h', real=True)
#         k= sy.Symbol('k', real=True)
#         l = sy.Symbol('L', real=True)
#     else :
#         p = sy.Function('p', positive=True)(elementOf)
#         f = sy.Function('f', real=True)(elementOf)
#         g = sy.Function('g', real=True)(elementOf)
#         h = sy.Function('h', real=True)(elementOf)
#         k= sy.Function('k', real=True)(elementOf)
#         l = sy.Function('L', real=True)(elementOf)

#     mu = sy.Symbol(r'\mu', positive=True)
#     return EquinoctialElements(p, f, g, h, k, l, mu)


# def CreateFromNonModifiedElements(a, h, k, p, q, lon, mu) :
#     p = a*(1.0-sy.sqrt(h*h+k*k))
#     f = h
#     g = k
