#%%
import __init__ #type: ignore
import numpy as np
from pandas import DataFrame#type: ignore
from scipy.integrate import solve_ivp #type: ignore
from pyeq2orb.Numerical.LambdifyHelpers import LambdifyHelper, OdeLambdifyHelper, OdeLambdifyHelperWithBoundaryConditions #type: ignore
import sympy as sy
from pyeq2orb import SymbolicOptimizerProblem
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Graphics.Primitives import EphemerisArrays
import math
import plotly.express as px#type: ignore
from pyeq2orb.Graphics.PlotlyUtilities import PlotlyDataAndFramesAccumulator
from pyeq2orb.Numerical import ScipyCallbackCreators
import pyeq2orb.Graphics.Primitives as prim
from pyeq2orb.Graphics.PlotlyUtilities import PlotAndAnimatePlanetsWithPlotly
from pyeq2orb.Coordinates.RotationMatrix import RotAboutZ
deg2rad = math.pi/180
t = sy.Symbol('t')
mu = sy.Symbol(r'\mu')
muVal = 0.01215

aState = MotionCartesian.CreateSymbolicMotion(t)
x = aState.Position.X
y = aState.Position.Y
z = aState.Position.Z
vx = aState.Velocity.X
vy = aState.Velocity.Y
vz = aState.Velocity.Z

d = sy.sqrt((x+mu)**2 + y**2 + z**2)
r = sy.sqrt((x-1+mu)**2+y**2+z**2)
u = 0.5*(x**2+y**2)+(1-mu)/d+mu/r
xEom = sy.Eq(sy.diff(x, t), vx)
yEom = sy.Eq(sy.diff(y, t), vy)
zEom = sy.Eq(sy.diff(z, t), vz)
vxEom = sy.Eq(sy.diff(vx, t), u.diff(x)+2*vy)
vyEom = sy.Eq(sy.diff(vy, t), u.diff(y)-2*vx)
vzEom = sy.Eq(sy.diff(vz, t), u.diff(z))

subsDict = {mu: muVal}
helper = OdeLambdifyHelper(t, [xEom, yEom, zEom, vxEom, vyEom, vzEom], [], subsDict)

integratorCallback = helper.CreateSimpleCallbackForSolveIvp()
tArray = np.linspace(0.0, 10.0, 1000)
# values were found on degenerate conic blog, but are originally from are from https://figshare.com/articles/thesis/Trajectory_Design_and_Targeting_For_Applications_to_the_Exploration_Program_in_Cislunar_Space/14445717/1
nhrlState = [ 	1.0277926091, 0.0, -0.1858044184, 0.0, -0.1154896637, 0.0]
ipvResults = solve_ivp(integratorCallback, [tArray[0], tArray[-1]], nhrlState, t_eval=tArray)
solutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(helper.NonTimeLambdifyArguments, ipvResults)
satEphemeris = EphemerisArrays()
satEphemeris.ExtendValues(ipvResults.t, solutionDictionary[x], solutionDictionary[y], solutionDictionary[z]) #type: ignore
satPath = prim.PathPrimitive(satEphemeris, "#ff00ff")

moonResults = solve_ivp(integratorCallback, [tArray[0], tArray[-1]], [ 1.0, 0.0, 0.0, 0.0, 0.1, 0.0], t_eval=tArray)
moonSolutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(helper.NonTimeLambdifyArguments, moonResults)
moonEphemeris = EphemerisArrays()



# cheat for the moon position in the rotating frame
x_2 = np.linspace(1.0, 1.0, 1000)
y_2 = np.linspace(0.0, 0.0, 1000)
moonEphemeris.ExtendValues(tArray, x_2, y_2, moonSolutionDictionary[z]) #type: ignore
moonPath = prim.PathPrimitive(moonEphemeris, "#ffffff")

fig = PlotAndAnimatePlanetsWithPlotly("NHRL in Rotation Frame", [satPath, moonPath], tArray, None)
fig.update_layout()
fig.show()  
from pyeq2orb.Graphics.Plotly2DModule import plot2DLines
from pyeq2orb.Graphics.Primitives import XAndYPlottableLineData

inertialEphemeris = EphemerisArrays()
moonInertialEphemeris = EphemerisArrays()
moonPos = sy.Matrix([[1.0],[0.0],[0.0]])
for i in range(0, len(tArray)):
    tNow = tArray[i]
    rotMat = RotAboutZ(tNow).evalf()
    newXyz = rotMat*sy.Matrix([[satEphemeris.X[i]],[satEphemeris.Y[i]],[satEphemeris.Z[i]]])
    inertialEphemeris.AppendValues(tNow, float(newXyz[0]), float(newXyz[1]), float(newXyz[2]))
    newMoonXyz = rotMat*moonPos
    moonInertialEphemeris.AppendValues(tNow, float(newMoonXyz[0]), float(newMoonXyz[1]), float(newMoonXyz[2]))

fig = PlotAndAnimatePlanetsWithPlotly("NHRL In Inertial Frame", [prim.PathPrimitive(inertialEphemeris, "#ff00ff", 3), prim.PathPrimitive(moonInertialEphemeris, "#ffffff", 3)], tArray, None)
fig.update_layout()
fig.show()  

