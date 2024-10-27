#%%
import matplotlib.pyplot as plt #type: ignore
import pyeq2orb.Problems.OneDimensionalMinimalWorkProblem as OneDWorkProblemModule #type: ignore
from pyeq2orb.Solvers.ScipyDiscretizationMinimizeWrapper import ScipyDiscretizationMinimizeWrapper #type: ignore
from typing import Dict, List
import sympy as sy #type: ignore
import numpy as np
n = 9 # coarse enough to see a small difference from the analytical solution
oneDWorkProblem = OneDWorkProblemModule.OneDWorkProblem()
scipySolver = ScipyDiscretizationMinimizeWrapper(oneDWorkProblem)
t = np.linspace(oneDWorkProblem.T0, oneDWorkProblem.Tf, n+1)
ans = scipySolver.ScipyOptimize(t)
plotAbleNumericalAnswer = scipySolver.ConvertScipyOptimizerOutputToDictionary(ans)

print(ans.success)
print(ans.message)

analyticalAnswerEvaluator = OneDWorkProblemModule.AnalyticalAnswerToProblem()
analyticalAnswer = analyticalAnswerEvaluator.EvaluateAnswer(oneDWorkProblem, t)


def AddResultsToFigure(figure : plt.Figure, t : List[float], dictionaryOfValueArraysKeyedOffState : Dict[sy.Expr, List[float]], label : str) -> None :
    plt.title("1D Work Problem")
    subPlotInt = 311
    for key in dictionaryOfValueArraysKeyedOffState:
        values = dictionaryOfValueArraysKeyedOffState[key]
        plt.subplot(subPlotInt)        
        plt.plot(t, values, label=label)
        plt.ylabel(key)
        subPlotInt = subPlotInt+1

    plt.legend()

fig = plt.figure()
oneDWorkProblem.AddResultsToFigure(fig, t, analyticalAnswer, "Analytical Answer")
oneDWorkProblem.AddResultsToFigure(fig, t, plotAbleNumericalAnswer, "Numerical Answer")
plt.show()
