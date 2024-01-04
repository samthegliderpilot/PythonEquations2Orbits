#%%
import sympy as sy
import os
import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0]))) # need to import 2 directories up (so pyeq2orb is a subfolder)
sy.init_printing()

from pyeq2orb.ForceModels.TwoBodyForce import CreateTwoBodyMotionMatrix, CreateTwoBodyListForModifiedEquinoctialElements
import pyeq2orb.Coordinates.KeplerianModule as KepModule
import pyeq2orb.Coordinates.OrbitFunctions as orb
from pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule import CreateSymbolicElements, ConvertKeplerianToEquinoctial, EquinoctialElementsHalfI
import scipyPaperPrinter as jh #type: ignore
jh.printMarkdown("# Modified Equinoctial Elements Summary")
jh.printMarkdown("While working with optimal control problems for satellite maneuvers, I needed a reference for working with Equinoctial and Modified Equinoctial Elements. This will pull together several sources and show the equations and some of the derivations of these equations that are encoded in the library I'm writing.")

jh.printMarkdown("Note that this document is made with the symbolic (sympy) routines I have written to do optimal control problems.  As such, equations will be simplified in minor ways and values in the equations will be in a different order than what sources might show.")

jh.printMarkdown("And many thanks for the Citation Style Language website for making citations so easy and simple @CslDefinition.  Also thanks to the AIAA for publishing their csl file @AiaaCslDef.")

jh.printMarkdown("## Purpose of Modified Equinoctial Elements")
jh.printMarkdown("First off, why even bother with Modified Equinoctial Elements?  There are a few reasons:")
jh.printMarkdown("- Similar to Keplerian Elements, there is only 1 fast variable (as opposed to Cartesian radius and velocity vectors which are all fast variables) which assists with the various optimization techniques that follows.")
jh.printMarkdown("- Keplerian elements have singularities at orbit common in the industry (GEO orbits especially) where the eccentricity is very low and the inclination is near 0. Equinoctial elements are much better behaved in those orbit regimes")
jh.printMarkdown("- Normal Equinoctial Elements have a singularity at inclinations of 90 degrees and cannot completely describe parabolic orbits.  Modified Equinoctial Elements only have singularities for orbits with an inclination of 180 degrees")
jh.printMarkdown("Note that there is ambiguity for orbits with an inclination near 0 or 180 degrees.  A rigorous set of elements should be a 'retrograde factor' that will determine if the orbit is posigrade or retrograde (or within a few degrees of 180 degrees), however for simplicity we will not be adding this factor @Vallado5thEdition.")

jh.printMarkdown("These characteristics make Equinoctial Elements a good choice for many types of problems.")

jh.printMarkdown("## Modified Equinoctial Orbital Elements Relationship with Keplerian Elements")
jh.printMarkdown("We will show how to convert Keplerian elements to Modified Equinoctial Elements:")
kepElements = KepModule.CreateSymbolicElements()
equiInTermsOfKep = ConvertKeplerianToEquinoctial(kepElements)
jh.showEquation("p", equiInTermsOfKep.SemiParameter)
jh.showEquation("f", equiInTermsOfKep.EccentricityCosTermF)
jh.showEquation("g", equiInTermsOfKep.EccentricitySinTermG)
jh.showEquation("h", equiInTermsOfKep.InclinationCosTermH)
jh.showEquation("k", equiInTermsOfKep.InclinationSinTermK)
jh.showEquation("L", equiInTermsOfKep.TrueLongitude)
jh.printMarkdown("Similar to Keplerian Elements, there is a True Longitude, Mean Longitude and Eccentric Longitude, and conversions between them have the same challenges as the Anomalies do for Keplerian Elements.")
jh.printMarkdown("Also, the True Longitude here can often be greater than 360 degrees.  There may be instances where you will want to modulus-divide it by 1 rev of your angle units.")

jh.printMarkdown("And the conversion of Modified Equinoctial Elements to Keplerian")
equiElements = CreateSymbolicElements()
kepElements = equiElements.ToKeplerian()
jh.showEquation("a", kepElements.SemiMajorAxis)
jh.showEquation("e", kepElements.Eccentricity)
jh.showEquation("i", kepElements.Inclination) #TODO: Look into why the atan functions have ugly pretty printing
jh.showEquation(r'\omega', kepElements.ArgumentOfPeriapsis)
jh.showEquation(r'\Omega', kepElements.RightAscensionOfAscendingNode)
jh.showEquation(r'\nu', kepElements.TrueAnomaly)

jh.printMarkdown("Where:")
jh.printMarkdown("- a = semi-major axis")
jh.printMarkdown("- p = semi-parameter") 
jh.printMarkdown("- e = orbit eccentricity")
jh.printMarkdown("- i = orbit inclination")
jh.printMarkdown("- $" + r'\omega' + "$ = argument of perigee")
jh.printMarkdown("- $" + r'\Omega' + "$ = right ascension of the ascending node")
jh.printMarkdown("- $" + r'\nu' +"$ = true anomaly")
jh.printMarkdown("- L = true longitude")
jh.printMarkdown("- $" + r'\mu' + "$ = gravitational parameter")

jh.printMarkdown("## Equinoctial Orbital Elements")
jh.printMarkdown("Indeed, the Modified elements are not the originally defined Equinoctial elements. Many sources describe non-modified Equinoctial elements, and converting them is straight forward @AppliedNonsingularAstrodynamics.")
jh.printMarkdown("The only difference is that the size of the orbit is defined with the semi-major axis, and the order of the other elements (the sin terms come before the cos terms).  The conversion from Modified Equinoctial Elements is:")
jh.printMarkdown("Going from Modified to original Equinoctial Elements:")
meeToEe = EquinoctialElementsHalfI.FromModifiedEquinoctialElements(equiElements)
jh.showEquation("a", meeToEe.SemiMajorAxis)
jh.showEquation("h", meeToEe.EccentricitySinTermH)
jh.showEquation("k", meeToEe.EccentricityCosTermK)
jh.showEquation("p", meeToEe.InclinationSinTermP)
jh.showEquation("q", meeToEe.InclinationCosTermQ)
jh.showEquation("L", meeToEe.TrueLongitude)

jh.printMarkdown("Going from Original Equinoctial Elements to Modified:")

ee = EquinoctialElementsHalfI.CreateSymbolicElements()
eeToMee = ee.ConvertToModifiedEquinoctial()
jh.showEquation("p", eeToMee.SemiParameter)
jh.showEquation("f", eeToMee.EccentricitySinTermG)
jh.showEquation("g", eeToMee.EccentricitySinTermG)
jh.showEquation("h", eeToMee.InclinationCosTermH)
jh.showEquation("k", eeToMee.InclinationSinTermK)
jh.showEquation("L", eeToMee.TrueLongitude)

jh.printMarkdown("I have been told of a set of equinoctial elements that the inclination terms are in terms of the inclination instead of half of the inclination, however I can not find a source for the elements in this form.  As such, conversions will not be shown here.")
#%%
jh.printMarkdown("## To and From Cartesian ")
jh.printMarkdown("Here are the conversions for going back and forth to cartesian position and velocity:")
asMotion = equiElements.ToMotionCartesian()
jh.showEquation(r'\bar{r}', asMotion.Position)
jh.showEquation(r'\bar{v}', asMotion.Velocity)
jh.printMarkdown("Note that this can be simplified with the use for a few other intermediate variables, but for the purposes of this document I am leaving it in terms of the primary Modified Equinoctial Elements.")


#%%
jh.printMarkdown("## Equations of Motion")
jh.printMarkdown("If you want to integrate modified equinoctial elements in time, the expressions are:")
jh.printMarkdown("In matrix form:")
y = sy.Symbol("y")
jh.showEquation(r'\dot{y}=\frac{dy}{dt}',sy.Function("A")(y)*sy.Symbol("P", commutative=False, algebraic =False)+sy.Symbol("b", commutative=False, algebraic =False))
jh.printMarkdown("Starting with $b$, the 2-Body Force, we get:")
twoBodyForceList = CreateTwoBodyMotionMatrix(equiElements)
jh.showEquation("b", twoBodyForceList)
jh.printMarkdown("As expected, for two body motion, the only change is in the True Longitude")
jh.printMarkdown("For the perturbation matrix, the cartesian acceleration is defined as:")

jh.printMarkdown("## To and from the RIC axes")
jh.printMarkdown("Although there is room for significant simplification, taking the basic definitions of the radial/in-tract/cross-track axes and the cartesian conversion, we get the following matrix converting an RIC axes to Inertial")

jh.showEquation("R_{RIC-ECI}", orb.CreateComplicatedRicToInertialMatrix(equiElements.ToMotionCartesian()))
jh.printMarkdown('### Sources')


#%%
if '__file__' in globals() or '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    thisFilePath = os.path.join(dir_path, "004_ModifiedEquinoctialElementsExplanation.py")
    jh.ReportGeneratorFromPythonFileWithCells.WriteIpynbToDesiredFormatWithPandoc(thisFilePath, keepDirectoryClean=True)
    jh.printMarkdown("done")

