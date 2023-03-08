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
from IPython.display import display
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
import scipyPaperPrinter as jh
jh.printMarkdown("# Sepspot Recreation")
jh.printMarkdown("In working my way up through low-thrust modeling for satellite maneuvers, it is inevetable to run into Dr. Edelbaum's work.  Newer work such as Jean Albert Kechichian's practicaly requires understanding SEPSPOT as a prerequesit.  This writeup will go through the basics of SEPSPOT's algorithsm as described in the references below.")

jh.printMarkdown("In other work in this python library, I have already created many helper types such as Equinocital elements, their equations of motion, rotation matrices, and more. To start, we will define out set of equinoctial elements.  Unlike the orignial paper, I will be using the modified elements.  This replaces the semi-major axis with the parameter and reorders/renames some of the other elements.")
t=sy.Symbol('t')
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
eccentricLongitude = sy.Function('F')(t)
simpleEquiElements.F = eccentricLongitude
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

meanAnomaly = sy.Function("M")(t)
kepElements.M = meanAnomaly
keplerianEquationLhs = kepElements.M + kepElements.ArgumentOfPeriapsis + kepElements.RightAscensionOfAscendingNode
keplerianEquationHhs = equiInTermsOfKep.F - equiInTermsOfKep.EccentricityCosTermF*sy.sin(eccentricLongitude) + equiInTermsOfKep.EccentricitySinTermG*sy.cos(eccentricLongitude)
kepEquation = sy.Eq(keplerianEquationLhs, keplerianEquationHhs)
jh.printMarkdown("And finally, we have Keplers equation")
jh.showEquation(kepEquation)


#%%
jh.printMarkdown("### The Optimal Control Problem")
jh.printMarkdown("We will limit ourselves (for now) to the 5-state orbit averaged problem.  We will also for now stick to the 2-body problem with no oblatness of the Earth.")
jh.printMarkdown("The paper defines the Hamiltonian as")
jh.printMarkdown(r'$$H=\underline{\lambda}^{T}\underline{\dot{x}}$$')
jh.printMarkdown("Which is the standard Hamiltonian I've seen in other sources assuming no path cost.")

def makeMatrixOfSymbols(baseString : str, rows, cols, t=None) :
    endString = ''
    if baseString.endswith('}') :
        baseString = baseString[:-1]
        endString = '}'
    mat = sy.Matrix.zeros(rows, cols)
    for r in range(0, rows) :
        for c in range(0, cols):
            

            if t== None :
                mat[r,c] = sy.Symbol(baseString + "_{" + str(r) + "," + str(c)+"}" + endString)
            else:
                mat[r,c] = sy.Function(baseString + "_{" + str(r) + "," + str(c)+"}"+ endString)(t)
    return mat


n = 5
jh.printMarkdown("Staring with our x:")
x = sy.Matrix([[simpleEquiElements.SemiParameter, simpleEquiElements.EccentricityCosTermF, simpleEquiElements.EccentricitySinTermG, simpleEquiElements.InclinationCosTermH, simpleEquiElements.InclinationSinTermK]]).transpose()
xSy = sy.MatrixSymbol('x', n, 1)
jh.showEquation(xSy, x)
jh.printMarkdown(r'We write our $\underline{\dot{x}}$ with the assumed optimal control vector $\underline{\hat{u}}$ as:')
g1Sy = makeMatrixOfSymbols(r'g_{1}', n, 1, t)
aSy = sy.Function('a', commutative=True)(x, t)
uSy = sy.Matrix([["u1", "u2", "u3"]]).transpose()
g2Sy = makeMatrixOfSymbols('G_{2}', 5, 3)
display(g2Sy)
xDotSy = SymbolicProblem.CreateCoVector(x, r'\dot{x}', t)
xDot = g1Sy+ aSy*g2Sy*uSy
jh.printMarkdown("Filling in our Hamiltonian, we get the following expression for our optimal thrust direction:")
#lambdas = sy.Matrix([[r'\lambda_{1}',r'\lambda_{2}',r'\lambda_{3}',r'\lambda_{4}',r'\lambda_{5}']]).transpose()
lambdas = SymbolicProblem.CreateCoVector(x, r'\lambda', t)
#lambdasSymbol = sy.Symbol(r'\lambda^T', commutative=False)
hamiltonin = lambdas.transpose()*xDot
# print(hamiltonin)
# display(hamiltonin)
# jh.showEquation("H", hamiltonin)
# stationaryCondition = sy.diff(hamiltonin, uSy)
# print(stationaryCondition)
# optU = sy.solve(stationaryCondition, uSy)
# jh.showEquation(uSy, optU)
#jh.printMarkdown(r'Sympy is having some trouble doing the derivative with MatrixSymbol\'s, so I\'ll explain instead.  The stationary condition will give us an expression for the optimal control, $\underline}{\hat{u}}$ by taking the partial derivative of H with respect to the control and setting it equal to zero. Then, solve for the control.  If we do that, noting that the control only appears with the $G_2$ term, and remembering that we want the normalized direction of the control vector, we get:')
jh.printMarkdown(r'Although normally we would take the partial derivative of the Hamiltonian with respect to the control, since the Hamiltonian is linear in the control, we need to take a more intuitive approch.  We want to maximize the $G_2$ term.  It ends up being $\lambda^T a G_2 u$.  Remembering that a is a magnitude scalar and u will be a normalized direction, we can drop it.  U is a 3 by 1 matrix, and $\lambda^T G_2$ will be a 1 by 3 matrix.  Clearly to maximize this term, the optimal u needs to be in the same direction as $\lambda^T G_2$, giving us our optimal u of')
uOpt = lambdas.transpose()*g2Sy / ((lambdas.transpose()*g2Sy).norm())
display(uOpt)

jh.printMarkdown("Putting this back into our Hamiltonian, we get")
hStar = (lambdas.transpose() * g1Sy)[0,0] + aSy*(uOpt.norm())
jh.showEquation("H^{*}", hStar)
jh.printMarkdown("Although not as cleanly defined as in the paper, we will soon be substituting expressions into this to create our equations of motion and boundary conditions.")
#%%
jh.printMarkdown("## Averaging of the Hamiltonian")

jh.printMarkdown("In order to get the averaged Hamiltonian, we need to make the following transformation:")
def createAveragedHamiltonian(h, lowerBound, upperBound,averageringVariabe, dtdAveragingVariable) :
    T = upperBound - lowerBound
    oneOverT = 1/T
    hamltAveraged = oneOverT * sy.integrate(h*dtdAveragingVariable, (averageringVariabe, lowerBound, upperBound))
    return hamltAveraged

kepEquationEquiElementsRhs = eccentricLongitude - simpleEquiElements.EccentricitySinTermG*sy.sin(eccentricLongitude) + simpleEquiElements.EccentricityCosTermF*sy.cos(eccentricLongitude)
jh.printMarkdown("The derivative of the left hand side of Keplers equation is the mean motion, where T is the period")
period = sy.Symbol('T')
dmdt = 2*sy.pi/period
display(dmdt)
jh.printMarkdown("And the right hand side will give us an expression for $\frac{dt}{dF}$")
dKepDtRhs = sy.diff(kepEquationEquiElementsRhs, t)
equToGetDFDt = sy.Eq(dmdt, dKepDtRhs)
dtdF=1/sy.solve(equToGetDFDt, sy.diff(eccentricLongitude, t))[0]
jh.showEquation(r'\frac{dt}{dF}', dtdF)
hAveraged = createAveragedHamiltonian(hStar, -1*sy.pi, sy.pi, eccentricLongitude, dtdF)
display(hAveraged)

jh.printMarkdown("With this, we need to start filling in our G1 and G2 expressions.  After that, it is applying the Optimal Control Euler-Lagrange expressions.")

display(simpleEquiElements.CreatePerturbationMatrix())

#%%
# if '__file__' in globals() or '__file__' in locals():
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     thisFilePath = os.path.join(dir_path, "ModifiedEquinoctialElementsExplanation.py")
#     jh.ReportGeneratorFromPythonFileWithCells.WriteIpynbToDesiredFormatWithPandoc(thisFilePath, keepDirectoryClean=True)
#     jh.printMarkdown("done")

