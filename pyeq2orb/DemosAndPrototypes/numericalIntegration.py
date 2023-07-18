#%%
import __init__ #type: ignore
from typing import List, Optional
import sympy as sy
from pyeq2orb.Numerical.OdeHelperModule import OdeHelper

import numpy as np 
import numpy.typing as npt
import plotly.express as px#type: ignore
import plotly.graph_objects as go#type: ignore
import pandas as pd #type: ignore
from scipy.optimize import fsolve #type: ignore
import scipyPaperPrinter as jh #type: ignore
from pyeq2orb.Graphics.Primitives import XAndYPlottableLineData
from pyeq2orb.Graphics.Plotly2DModule import plot2DLines

def ActuallySolvedCallback(callbackTThenState, y0 : npt.NDArray, t0 : float, tNext :float, parametersArray : Optional[List[float]] = None) -> List[float] :
    return callbackTThenState(tNext, y0, parametersArray)

def TrapizodalForward(callbackTThenState, y0 :npt.NDArray, t0 : float, tNext :float, parametersArray : Optional[List[float]] = None) -> List[float] :
    dy0 = np.array(callbackTThenState(t0, y0, parametersArray))
    dt = tNext - t0
    return y0 + dt*dy0

def TrapizodalBackward(callbackTThenState, y0 : npt.NDArray, t0 : float, tNext :float, parametersArray : Optional[List[float]] = None) -> List[float] :
    dt = tNext - t0
    def fSolveCallback(y1) :
        return y0+ dt*np.array(callbackTThenState(tNext, y1, parametersArray)) - y1

    forwardGuess = TrapizodalForward(callbackTThenState, y0, t0, tNext, parametersArray)
    y1 = fsolve(fSolveCallback, forwardGuess)
    return y1

def integrateSystemOverTimeRange(callback, integrationMethod, initialCondition, parametersArray, t0, tf, count) :
    allData = []
    allData.append(initialCondition)
    curState = np.array(initialCondition)
    step = (tf-t0)/count
    lastT = t0
    for i in range(0, count+1) :
        nextT = lastT+step
        stateNext = integrationMethod(callback, curState, lastT, nextT, parametersArray)
        allData.append(stateNext)
        curState = stateNext
        lastT = nextT

    allData = np.array(allData).transpose()
    return allData

plotData = []

g = 9.806
t0 = 0
tf = 20

t = sy.Symbol('t')
x = sy.Function('x')
y = sy.Function('y')
vx = sy.Function('v_x')
vy = sy.Function('v_y')

gSy = sy.Symbol('g')

initialValues = [0.0, 0.0, 10.0, 100.0]

gravityOdeHelper = OdeHelper(t)
gravityOdeHelper.setStateElement(x(t), vx(t), x(0))
gravityOdeHelper.setStateElement(y(t), vy(t), y(0))
gravityOdeHelper.setStateElement(vx(t), 0.0, vx(0))
gravityOdeHelper.setStateElement(vy(t), -1*gSy, vy(0))

#gravityOdeHelper.lamdifyParameterSymbols.append(gSy)
gravityOdeHelper.constants[gSy] = g

print(gravityOdeHelper.makeStateForLambdififedFunction())

callback = gravityOdeHelper.createLambdifiedCallback()
desolveAns = gravityOdeHelper.tryDeSolve()
deSolveCb = gravityOdeHelper.deSolveResultsToCallback(desolveAns, initialValues)

count = 20
allDataForward = integrateSystemOverTimeRange(callback, TrapizodalForward, initialValues, [g], t0, tf, count)
trap = XAndYPlottableLineData(allDataForward[0], allDataForward[1], "Trap_Forward", '#ff0000', 0, 4)

allDataBackward = integrateSystemOverTimeRange(callback, TrapizodalBackward, initialValues, [g],t0, tf, count)
trapBack = XAndYPlottableLineData(allDataBackward[0], allDataBackward[1], "Trap_Backward", '#0000ff', 0, 4)

allData = integrateSystemOverTimeRange(deSolveCb, ActuallySolvedCallback, initialValues, [g], t0, tf, count)
truthData = XAndYPlottableLineData(allData[0], allData[1], "Truth", '#00ff00', 1, 0)

plotData.append(trap)
plotData.append(trapBack)
plotData.append(truthData)

plot2DLines(plotData, "Integration Experements")
