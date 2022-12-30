#%% [markdown]
# (my-label)=## My header

#%%
import sympy as sy
import os
import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0]))) # need to import 2 directories up (so pyeq2orb is a subfolder)
sy.init_printing()

from pyeq2orb.ForceModels.TwoBodyForce import CreateTwoBodyMotionMatrix, CreateTwoBodyList
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Coordinates.KeplerianModule import KeplerianElements
import pyeq2orb.Coordinates.KeplerianModule as KepModule
from pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule import CreateSymbolicElements, ConvertKeplerianToEquinoctial, EquinoctialElementsHalfI
import JupyterHelper as jh
jh.printMarkdown("# Modified Equinoctial Elements Summary")
jh.printMarkdown("While working with optimal control problems for satellite maneuvers, I needed a reference for working with Equinoctial and Modifeid Equinoctial Elements. This will pull together several sources and show the equations and some of the derivations of these equations that are encoded in the library I'm writing.[here's my label](my-label)")

#%%
jh.printMarkdown("Note that this document is made with python code using sympy.  Some of the equations may be simplified or have their terms ordered in an odd way.")

jh.printMarkdown("First off, why even bother with Modified Equinoctial Elements?  There are a few reasons:")
jh.printMarkdown("- Similar to Keplerian Elements, there is only 1 fast variable (as opposed to Cartesian radius and velocity vectors which are all fast variables) which assists with the various optimization techniques that follows.")
jh.printMarkdown("- Keplerian elements have singularties at orbit common in the industry (GEO orbits especally) where the eccentriciy is very low and the inclination is near 0. Equinoctial elements are much better behaved in those orbit regiems")
jh.printMarkdown("- Normal Equinoctial Elements have a singularity at inclinations of 90 degrees and cannot completely describe parabolic orbits.  Modified Equinoctial Elements only have singularities for orbits with an inclination of 180 degrees")
jh.printMarkdown("-- Note that there is ambuguity for orbits with an inclination near 0 or 180 degrees")

jh.printMarkdown("These characteristics make Equinoctial Elements a good choice for many types of problems.")

jh.printMarkdown("## Modified Equinoctial Orbial Elements Relationship with Keplerian Elements")
jh.printMarkdown("We will show how to convert Keplerian elements to Modified Equinoctial Elements:")
kepElements = KepModule.CreateSymbolicElements()
equiInTermsOfKep = ConvertKeplerianToEquinoctial(kepElements)
jh.showEquation("p", equiInTermsOfKep.SemiParameter)
jh.showEquation("f", equiInTermsOfKep.EccentricitySinTermG)
jh.showEquation("g", equiInTermsOfKep.EccentricitySinTermG)
jh.showEquation("h", equiInTermsOfKep.InclinationCosTermH)
jh.showEquation("k", equiInTermsOfKep.InclinationSinTermK)
jh.showEquation("L", equiInTermsOfKep.TrueLongitude)
jh.printMarkdown("Similar to Keplerian Elements, there is a True Longitude, Mean Longitude and Eccentricic Longitude, and converstions between them have the same challanges as the Anomalies do for Keplerian Elements.")
jh.printMarkdown("Also, the True Longitude here can often be greater than 360 degrees.  There may be instances where you will want to modulus-divide it by 1 rev of your angle units.")

jh.printMarkdown("And the conversion of Modified Equinoctial Elements to Keplerian")
equiElements = CreateSymbolicElements()
kepElements = equiElements.ToKeplerian()
jh.showEquation("a", kepElements.SemiMajorAxis)
jh.showEquation("e", kepElements.Eccentricity)
jh.showEquation("i", kepElements.Inclination) #TODO: Look into why the atan functions have ugly pretty printing
jh.showEquation(r'\omega', kepElements.ArgumentOfPeriapsis)
jh.showEquation(r'\Omega', kepElements.RightAscensionOfAscendingNode)
jh.showEquation("t", kepElements.TrueAnomaly)

jh.printMarkdown("Where:")
jh.printMarkdown("- a = semimajor axis")
jh.printMarkdown("- p = semiparameter") 
jh.printMarkdown("- e = orbit eccentricity")
jh.printMarkdown("- i = orbit inclination")
jh.printMarkdown("- $" + r'\omega' + "$ = argument of perigee")
jh.printMarkdown("- $" + r'\Omega' + "$ = right ascension of the ascending node")
jh.printMarkdown("- t = true anomaly") #TODO: Change default symbol in Keplerian Elemements
jh.printMarkdown("- L = true longitude")
jh.printMarkdown("- $" + r'\mu' + "$ = gravitational paraemter")

jh.printMarkdown("## Equinoctial Orbial Elements")
jh.printMarkdown("Indeed, the Modified elements are not the originally defined Equinoctial elements. Many sources describe vanilla Equinoctial elements, and they are fine.")
jh.printMarkdown("The only difference is that the size of the orbit is defined with the semi-major axis, and the order of the other elements (the sin terms come before the cos terms).  The converstion from Modified Equinoctial Elements is:")
jh.printMarkdown("Going from Modified to original Equinoctial Elements:")
meeToEe = EquinoctialElementsHalfI.FromModifiedEquinoctialElements(equiElements)
jh.showEquation("a", meeToEe.SemiMajorAxis)
jh.showEquation("h", meeToEe.EccentricitySinTermH)
jh.showEquation("k", meeToEe.EccentricityCosTermJ)
jh.showEquation("p", meeToEe.InclinationSinTermP)
jh.showEquation("q", meeToEe.InclinationCosTermQ)
jh.showEquation("L", meeToEe.Longitude)

jh.printMarkdown("Going from Original Equinoctial Elements to Modified:")

ee = EquinoctialElementsHalfI.CreateSymbolicElements()
eeToMee = ee.ConvertToModifiedEquinoctial()
jh.showEquation("p", eeToMee.SemiParameter)
jh.showEquation("f", eeToMee.EccentricitySinTermG)
jh.showEquation("g", eeToMee.EccentricitySinTermG)
jh.showEquation("h", eeToMee.InclinationCosTermH)
jh.showEquation("k", eeToMee.InclinationSinTermK)
jh.showEquation("L", eeToMee.TrueLongitude)

jh.printMarkdown("However, I have been told of a set of equinoctial elements that the inclination terms are in terms of the inclination instead of half of the inclination.  This causes the converstion to be much more complicated.  However I can not find a source for the elements in this form, so they will not be covered here.")

#%%
if '__file__' in globals() or '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    reportCreator = jh.ReportGeneratorFromPythonFileWithCells(dir_path, "ModifiedEquinoctialElementsExplination.py", "ModifiedEquinoctialElementsExplination.pdf")
    reportCreator.WritePdfDirectlyFromJupyter()
    jh.printMarkdown("done")