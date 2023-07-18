#%%
import sympy.physics.vector as vec
import __init__ #type: ignore
import sympy as sy
import os
import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0]))) # need to import 2 directories up (so pyeq2orb is a subfolder)
sy.init_printing()

from pyeq2orb.ForceModels.TwoBodyForce import CreateTwoBodyMotionMatrix, CreateTwoBodyListForModifiedEquinoctialElements
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Coordinates.KeplerianModule import KeplerianElements
import pyeq2orb.Coordinates.KeplerianModule as KepModule
import pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule as mee
from IPython.display import display
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
import scipyPaperPrinter as jh #type: ignore
from typing import Dict

t=sy.Symbol('t')
mu = sy.Symbol(r'\mu')
kepElements = KepModule.CreateSymbolicElements()
simpleEquiElements = mee.CreateSymbolicElements()
simpleBoringEquiElements = mee.EquinoctialElementsHalfI.CreateSymbolicElements()
equiInTermsOfKep = mee.ConvertKeplerianToEquinoctial(kepElements)
eccentricAnomaly = sy.Symbol('E')
eccentricLongitude = sy.Function('F')(t)
f1 = eccentricLongitude
f2 = eccentricAnomaly + kepElements.ArgumentOfPeriapsis + kepElements.RightAscensionOfAscendingNode
x1Sy = sy.Symbol('X_1')
x2Sy = sy.Symbol('X_2')
x1DotSy = sy.Symbol(r'\dot{X_1}')
x2DotSy = sy.Symbol(r'\dot{X_2}')

xSuperSimple = Cartesian(x1Sy, x2Sy, 0)
xDotSuperSimple = Cartesian(x1DotSy, x2DotSy, 0)
fullSubsDictionary = {} #type: Dict[sy.Expr, float]
[x1SimpleEqui, x2SimpleEqui] = simpleBoringEquiElements.RadiusInFgw(eccentricLongitude, fullSubsDictionary)
[x1DotSimpleEqui, x2DotSimpleEqui] = simpleBoringEquiElements.VelocityInFgw(eccentricLongitude, fullSubsDictionary)
normalEquiElementsInTermsOfKep = mee.EquinoctialElementsHalfI.FromModifiedEquinoctialElements(equiInTermsOfKep)
[x1Complete, x2Complete] = normalEquiElementsInTermsOfKep.RadiusInFgw(f1)
[x1DotComplete, x2DotComplete] = normalEquiElementsInTermsOfKep.VelocityInFgw(f1)
x1PerAdFunc = sy.Function('x_1', real=True)(simpleBoringEquiElements.SemiMajorAxis, simpleBoringEquiElements.EccentricitySinTermH, simpleBoringEquiElements.EccentricityCosTermJ)
x2PerAdFunc = sy.Function('x_2', real=True)(simpleBoringEquiElements.SemiMajorAxis, simpleBoringEquiElements.EccentricitySinTermH, simpleBoringEquiElements.EccentricityCosTermJ)
x1DotPerAdFunc = sy.Function('\dot{x_1}', real=True)(simpleBoringEquiElements.SemiMajorAxis, simpleBoringEquiElements.EccentricitySinTermH, simpleBoringEquiElements.EccentricityCosTermJ)
x2DotPerAdFunc = sy.Function('\dot{x_2}', real=True)(simpleBoringEquiElements.SemiMajorAxis, simpleBoringEquiElements.EccentricitySinTermH, simpleBoringEquiElements.EccentricityCosTermJ)

x1AsFunc = sy.Function('X_1', real=True)(x1PerAdFunc, x2PerAdFunc, simpleBoringEquiElements.InclinationCosTermQ, simpleBoringEquiElements.InclinationSinTermP)
x2AsFunc = sy.Function('X_2', real=True)(x1PerAdFunc, x2PerAdFunc, simpleBoringEquiElements.InclinationCosTermQ, simpleBoringEquiElements.InclinationSinTermP)
x3AsFunc = sy.Function('X_3', real=True)(x1PerAdFunc, x2PerAdFunc, simpleBoringEquiElements.InclinationCosTermQ, simpleBoringEquiElements.InclinationSinTermP)
x1DotAsFunc = sy.Function('\dot{X_1}', real=True)(x1PerAdFunc, x2PerAdFunc, simpleBoringEquiElements.InclinationCosTermQ, simpleBoringEquiElements.InclinationSinTermP)
x2DotAsFunc = sy.Function('\dot{X_2}', real=True)(x1PerAdFunc, x2PerAdFunc, simpleBoringEquiElements.InclinationCosTermQ, simpleBoringEquiElements.InclinationSinTermP)
x3DotAsFunc = sy.Function('\dot{X_3}', real=True)(x1PerAdFunc, x2PerAdFunc, simpleBoringEquiElements.InclinationCosTermQ, simpleBoringEquiElements.InclinationSinTermP)

w1Func = sy.Function("w_1", real=True)(x1AsFunc, x2AsFunc, x3AsFunc, x1DotAsFunc, x2DotAsFunc, x3DotAsFunc)
w2Func = sy.Function("w_2", real=True)(x1AsFunc, x2AsFunc, x3AsFunc, x1DotAsFunc, x2DotAsFunc, x3DotAsFunc)
w3Func = sy.Function("w_3", real=True)(x1AsFunc, x2AsFunc, x3AsFunc, x1DotAsFunc, x2DotAsFunc, x3DotAsFunc)

xSuperSimple = Cartesian(x1AsFunc, x2AsFunc, x3AsFunc) 
xDotSuperSimple =Cartesian(x1DotAsFunc, x2DotAsFunc, x3DotAsFunc)
r = simpleEquiElements.CreateFgwToInertialAxes()


x1, x2, x3 = vec.dynamicsymbols('x_1 x_2 x_3')
x1Dot = x1.diff('t')
x2Dot = x2.diff('t')
x3Dot = x3.diff('t')

I = vec.ReferenceFrame("I")
equiFrame = vec.ReferenceFrame("E")

inert = vec.ReferenceFrame("i")
inert.orient(equiFrame, 'DCM', r)

posInertial = x1*inert.x+x2*inert.y+x3*inert.z
velInertial = x1Dot*inert.x+x2Dot*inert.y+x3Dot*inert.z

w = posInertial.cross(velInertial)
display(posInertial)
display(velInertial)
display(w)
wMat = w.to_matrix(equiFrame)
p = wMat[0]/(1+wMat[2])
display(p)
display(p.diff(x3Dot).expand().simplify())
