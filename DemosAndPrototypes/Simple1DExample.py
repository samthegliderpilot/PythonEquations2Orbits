#%%
import matplotlib.pyplot as plt #type: ignore
import pyeq2orb.Problems.OneDimensionalMinimalWorkProblem as OneDWorkProblemModule
from pyeq2orb.Solvers.ScipyDiscretizationMinimizeWrapper import ScipyDiscretizationMinimizeWrapper

n = 9 # coarse enough to see a small difference from the analytical solution
oneDWorkProblem = OneDWorkProblemModule.OneDWorkProblem()
scipySolver = ScipyDiscretizationMinimizeWrapper(oneDWorkProblem)
ans = scipySolver.ScipyOptimize(n)
plottableNumericalAnswer = scipySolver.ConvertScipyOptimizerOutputToDictionary(ans)

print(ans.success)
print(ans.message)

t = oneDWorkProblem.CreateTimeRange(n)
analyticalAnswerEvaluator = OneDWorkProblemModule.AnalyticalAnswerToProblem()
analyticalAnswer = analyticalAnswerEvaluator.EvaluateAnswer(oneDWorkProblem, t)

fig = plt.figure()
oneDWorkProblem.AddResultsToFigure(fig, t, analyticalAnswer, "Analytical Answer")
oneDWorkProblem.AddResultsToFigure(fig, t, plottableNumericalAnswer, "Numerical Answer")
plt.show()

# %%
