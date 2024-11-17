#%%
import numpy as np
from pandas import DataFrame #type: ignore
from scipy.integrate import solve_ivp #type: ignore
from pyeq2orb.Numerical.LambdifyHelpers import LambdifyHelper, OdeLambdifyHelper #type: ignore
import sympy as sy #type: ignore
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian #type: ignore
from pyeq2orb.Graphics.Primitives import EphemerisArrays #type: ignore
import math
import plotly.express as px #type: ignore
from pyeq2orb.Graphics.PlotlyUtilities import PlotlyDataAndFramesAccumulator #type: ignore
from pyeq2orb.Numerical import ScipyCallbackCreators #type: ignore
import pyeq2orb.Graphics.Primitives as prim #type: ignore
from pyeq2orb.Graphics.PlotlyUtilities import PlotAndAnimatePlanetsWithPlotly
from pyeq2orb.Coordinates.RotationMatrix import RotAboutZ #type: ignore
from pyeq2orb.Graphics.Plotly2DModule import plot2DLines #type: ignore
from pyeq2orb.Graphics.Primitives import XAndYPlottableLineData #type: ignore
import plotly.graph_objects as go #type: ignore 
import plotly.io as pio #type: ignore
pio.renderers.default = "vscode"
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
helper = OdeLambdifyHelper(t, [x,y,z,vx,vy,vz], [xEom.rhs, yEom.rhs, zEom.rhs, vxEom.rhs, vyEom.rhs, vzEom.rhs], [], subsDict)

integratorCallback = helper.CreateSimpleCallbackForSolveIvp()
tArray = np.linspace(0.0, 10.0, 1000)
# values were found on degenerate conic blog, but are originally from are from https://figshare.com/articles/thesis/Trajectory_Design_and_Targeting_For_Applications_to_the_Exploration_Program_in_Cislunar_Space/14445717/1
nhrlState = [1.02134, 0, -0.18162, 0, -0.10176, 9.76561e-07]# [ 	1.0277926091, 0.0, -0.1858044184, 0.0, -0.1154896637, 0.0]
ipvResults = solve_ivp(integratorCallback, [tArray[0], tArray[-1]], nhrlState, t_eval=tArray)
solutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(helper.NonTimeLambdifyArguments, ipvResults)
satEphemeris = EphemerisArrays()
satEphemeris.ExtendValues(ipvResults.t, solutionDictionary[x], solutionDictionary[y], solutionDictionary[z]) #type: ignore
satPath = prim.PathPrimitive(satEphemeris, "#ff00ff")

moonResults = solve_ivp(integratorCallback, [tArray[0], tArray[-1]], [ 1.0, 0.0, 0.0, 0.0, 0.1, 0.0], t_eval=tArray, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
moonSolutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(helper.NonTimeLambdifyArguments, moonResults)
moonEphemeris = EphemerisArrays()



# cheat for the moon position in the rotating frame
x_2 = np.linspace(1.0, 1.0, 1000)
y_2 = np.linspace(0.0, 0.0, 1000)
moonEphemeris.ExtendValues(tArray, x_2, y_2, moonSolutionDictionary[z]) #type: ignore
moonPath = prim.PathPrimitive(moonEphemeris, "#000000")

fig = PlotAndAnimatePlanetsWithPlotly("NHRL in Rotation Frame", [satPath, moonPath], tArray, None)
fig.update_layout(
     margin=dict(l=20, r=20, t=20, b=20))

fig.show()  


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
print(inertialEphemeris.X[0])
print(inertialEphemeris.Y[0])
print(inertialEphemeris.Z[0])
fig = PlotAndAnimatePlanetsWithPlotly("NHRL In Inertial Frame", [prim.PathPrimitive(inertialEphemeris, "#ff00ff", 3), prim.PathPrimitive(moonInertialEphemeris, "#000000", 3)], tArray, None)
fig.update_layout(
     margin=dict(l=20, r=20, t=20, b=20))

fig.show()  
#%%


def jacobi(x, y, z, xDot, yDot, zDot, mu) :
    r1 = np.sqrt((x+mu)**2+y*y+z*z)
    r2 = np.sqrt((x-1.0+mu)**2+y*y+z*z)
    u = 0.5*(x*x+y*y)+(1-mu)/r1 + mu/r2
    vSquared = xDot*xDot + yDot*yDot + zDot*zDot
    return 2*u-vSquared
  
feature_x = np.arange(-1.7, 1.7, 0.01) 
feature_y = np.arange(-1.7, 1.7, 0.01) 
  
# Creating 2-D grid of features 
[X, Y] = np.meshgrid(feature_x, feature_y) 
  
Z = jacobi(X, Y, 0, 0, 0, 0, muVal)
    
fig = go.Figure(data = go.Contour(
        x = feature_x, 
        y = feature_y, 
        z = Z,
        zmin = 2.5, 
        zmax = 3.5, 
        zauto=False, 
        contours_coloring='heatmap', 
        ncontours=60))

fig.layout.yaxis.scaleanchor="x"     
fig.update_layout(xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False),
                  plot_bgcolor = "rgb(255, 255, 255)",
                  paper_bgcolor = "rgb(255, 255, 255)",
        width=600, height=500)
# layout = grob.Layout(

# )
fig.update_layout(
     margin=dict(l=20, r=20, t=20, b=20))


fig.show(width=400, autosize=False)

# %%

#%%
fig = go.Figure(data=[go.Surface(z=-1*Z, x=X, y=Y)])
fig.layout.yaxis.scaleanchor="x"     
fig.update_layout(title=r'3Body Potential \mu = ' +str(muVal), autosize=False,
                  width=500, height=500,
                  scene=dict(zaxis = dict(nticks=4, range=[-3.5,-2.5])),
                  margin=dict(l=20, r=20, t=20, b=20))
fig.show()

