from EquinicotialDemo import *
from pyeq2orb.Graphics import CzmlUtilities
from datetime import datetime
tfVal = 3*86400.0
m0Val = 2000.0
isp = 3000.0
nRev = 2.0
thrustVal =  0.1997*1.2
g = 9.8065 
n = 201
tSpace = np.linspace(0.0, tfVal, n)

Au = 149597870700.0
AuSy = sy.Symbol('A_u')
muVal = 3.986004418e14  
r0 = Cartesian(8000.0e3, 8000.0e3, 0.0)
v0 = Cartesian(0, 5.000e3, 4.500e3)
initialElements = ModifiedEquinoctialElements.FromMotionCartesian(MotionCartesian(r0, v0), muVal)

gSy = sy.Symbol('g', real=True, positive=True)


twoBodyMatrix = CreateTwoBodyListForModifiedEquinoctialElements(symbolicElements)
simpleTwoBodyLambidfyCreator = LambdifyHelper(t, symbolicElements.ToArray(), twoBodyMatrix, [], {symbolicElements.GravitationalParameter: muVal})
odeCallback =simpleTwoBodyLambidfyCreator.CreateSimpleCallbackForSolveIvp()
earthSolution = solve_ivp(odeCallback, [0.0, tfVal], initialElements.ToArray(), args=tuple(), t_eval=np.linspace(0.0, tfVal,n), dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)

(tArray, equiElements) = GetEquiElementsOutOfIvpResults(earthSolution)
motions = ModifiedEquinoctialElements.CreateEphemeris(equiElements)
earthEphemeris = prim.EphemerisArrays()
earthEphemeris.InitFromMotions(tArray, motions)
earthPath = prim.PathPrimitive(earthEphemeris)
earthPath.color = "#0000ff"



czmlDoc = CzmlUtilities.createCzmlFromPoints(datetime(year=2020, month=1, day=1), "Planets", [earthPath])
f = open(r'C:\src\CesiumElectronStarter\someCzml.czml', "w")
f.write(str(czmlDoc))
f.close()