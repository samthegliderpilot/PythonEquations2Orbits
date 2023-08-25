import numpy as np
from pandas import DataFrame#type: ignore
from scipy.integrate import solve_ivp #type: ignore
from pyeq2orb.DemosAndPrototypes.LambidfyHelpers import LambdifyHelper, OdeLambdifyHelper, OdeLambdifyHelperWithBoundaryConditions #type: ignore
import sympy as sy
from pyeq2orb import SymbolicOptimizerProblem
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Graphics.Primitives import EphemerisArrays
import math
import plotly.express as px#type: ignore
from pyeq2orb.Graphics.PlotlyUtilities import PlotlyDataAndFramesAccumulator
from pyeq2orb.Numerical import ScipyCallbackCreators
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
tArray = np.linspace(0.0, 1000.0, 100000)
ipvResults = solve_ivp(integratorCallback, [tArray[0], tArray[-1]], [1.1, 0.0, 0.0, 0.0, 0.3, 0.0], t_eval=tArray)
solutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(helper.NonTimeLambdifyArguments, ipvResults)
satEphemeris = EphemerisArrays()
satEphemeris.AppendValues(ipvResults.t, solutionDictionary[x], solutionDictionary[y], solutionDictionary[z]) #type: ignore
satDf = PlotlyDataAndFramesAccumulator.CreatePlotlyEphemerisDataFrame(satEphemeris)

circleInertialEphemeris = EphemerisArrays()
for i in range(0, 360) :
    #circleInertialEphemeris.AppendValues(i/360.0, math.cos(deg2rad*i), math.sin(deg2rad*i), 0)
    circleInertialEphemeris.AppendValues(i/360.0, 1.0, 0.0, 0.0)

#df = PlotlyDataAndFramesAccumulator.CreatePlotlyEphemerisDataFrame(circleInertialEphemeris)

#acumulator = PlotlyDataAndFramesAccumulator()
#acumulator.data.append(df)
#acumulator.data.append(satDf)

#fig = px.line_3d(df, x="x", y="y", z="z")
fig2 = px.line_3d(satDf, x="x", y="y", z="z")
#fig.show()
fig2.show()

from pyeq2orb.Graphics.Plotly2DModule import plot2DLines
from pyeq2orb.Graphics.Primitives import XAndYPlottableLineData

plot2DLines([XAndYPlottableLineData(solutionDictionary[x], solutionDictionary[y], "orbit", "#ff0000", 2, 1)], "Orbit") #type: ignore