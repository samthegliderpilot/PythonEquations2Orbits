#%%
import __init__
import sympy as sy
import os
import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0]))) # need to import 2 directories up (so pyeq2orb is a subfolder)
sy.init_printing()


from pyeq2orb.ForceModels.TwoBodyForce import CreateTwoBodyMotionMatrix, CreateTwoBodyListForModifiedEquinoctialElements
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Coordinates.KeplerianModule import KeplerianElements
import pyeq2orb.Coordinates.KeplerianModule as KepModule
from pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule import CreateSymbolicElements, ConvertKeplerianToEquinoctial, EquinoctialElementsHalfI
import scipyPaperPrinter as jh
jh.printMarkdown("# Sepspot Recreation")
jh.printMarkdown("In working my way up through low-thrust modeling for satellite maneuvers, it is inevetable to run into Dr. Edelbaum's work.  Newer work such as Jean Albert Kechichian's practicaly requires understanding SEPSPOT as a prerequesit.  This writeup will go through the basics of SEPSPOT's algorithsm as described in the references below.")

jh.printMarkdown("In other work in this python library, I have already created many helper types such as Equinocital elements, their equations of motion, rotation matrices, and more. To start, we will define out set of equinoctial elements.  Unlike the orignial paper, I will be using the modified elements.  This replaces the semi-major axis with the parameter and reorders/renames some of the other elements.")
kepElements = KepModule.CreateSymbolicElements()
simpleEquiElements = CreateSymbolicElements()
simpleBoringEquiElements = EquinoctialElementsHalfI.CreateSymbolicElements()
equiInTermsOfKep = ConvertKeplerianToEquinoctial(kepElements)
jh.showEquation("p", equiInTermsOfKep.SemiParameter)
jh.showEquation("f", equiInTermsOfKep.EccentricitySinTermG)
jh.showEquation("g", equiInTermsOfKep.EccentricitySinTermG)
jh.showEquation("h", equiInTermsOfKep.InclinationCosTermH)
jh.showEquation("k", equiInTermsOfKep.InclinationSinTermK)
jh.showEquation("L", equiInTermsOfKep.TrueLongitude)
eccentricAnomaly = sy.Symbol('E')
eccentricLongitude = sy.Symbol('F')
equiInTermsOfKep.F = eccentricAnomaly + kepElements.ArgumentOfPeriapsis + kepElements.RightAscensionOfAscendingNode
jh.printMarkdown("We want our orbital elements to use the eccentric longitude which is:")
jh.showEquation(eccentricLongitude, equiInTermsOfKep.F) #TODO: Look into how to better include this in the normal equi elements

jh.printMarkdown("The rotation matrix of the axes being used for this analysis to inertial is:")
jh.showEquation("R", simpleEquiElements.CreateFgwToInertialAxes())
jh.printMarkdown("And with keplerian elements:")
jh.showEquation("R", equiInTermsOfKep.CreateFgwToInertialAxes())

jh.printMarkdown("And we need the position and velocity in the FGW axes.  Using the normal equinoctial elements (in order to better compare to the original paper):")
[x1SimpleEqui, x2SimpleEqui] = simpleBoringEquiElements.RadiusInFgw(eccentricLongitude)
[x1DotSimpleEqui, x2DotSimpleEqui] = simpleBoringEquiElements.VelocityInFgw(eccentricLongitude)
jh.showEquation("X_1", x1SimpleEqui)
jh.showEquation("X_2", x2SimpleEqui)
jh.showEquation("\dot{X_1}", x1SimpleEqui)
jh.showEquation("\dot{X_2}", x2SimpleEqui)

simpleBoringEquiElements = EquinoctialElementsHalfI.FromModifiedEquinoctialElements(equiInTermsOfKep)
[x1SimpleEqui, x2SimpleEqui] = simpleBoringEquiElements.RadiusInFgw(equiInTermsOfKep.F)
[x1DotSimpleEqui, x2DotSimpleEqui] = simpleBoringEquiElements.VelocityInFgw(equiInTermsOfKep.F)
jh.showEquation("X_1", x1SimpleEqui)
jh.showEquation("X_2", x2SimpleEqui)
jh.showEquation("\dot{X_1}", x1SimpleEqui)
jh.showEquation("\dot{X_2}", x2SimpleEqui)

meanAnomaly = sy.Symbol("M")
kepElements.M = meanAnomaly
keplerianEquationLhs = kepElements.M + kepElements.ArgumentOfPeriapsis + kepElements.RightAscensionOfAscendingNode
keplerianEquationHhs = equiInTermsOfKep.F - equiInTermsOfKep.EccentricityCosTermF*sy.sin(equiInTermsOfKep.F) + equiInTermsOfKep.EccentricitySinTermG*sy.cos(equiInTermsOfKep.F)
kepEquation = sy.Eq(keplerianEquationLhs, keplerianEquationHhs)
jh.printMarkdown("And finally, we have Keplers equition")
jh.showEquation(kepEquation)

#%%
if '__file__' in globals() or '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    thisFilePath = os.path.join(dir_path, "ModifiedEquinoctialElementsExplanation.py")
    jh.ReportGeneratorFromPythonFileWithCells.WriteIpynbToDesiredFormatWithPandoc(thisFilePath, keepDirectoryClean=True)
    jh.printMarkdown("done")

