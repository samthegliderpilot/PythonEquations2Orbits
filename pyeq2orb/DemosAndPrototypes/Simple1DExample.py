#%%
import sys
sys.path.append("..") # treating this as a jupyter-like cell requires adding one directory up
sys.path.append("../pyeq2orb") # and this line is needed for running like a normal python script
# these two appends do not conflict with eachother
import matplotlib.pyplot as plt #type: ignore
import pyeq2orb.Problems.OneDimensionalMinimalWorkProblem as OneDWorkProblemModule
from pyeq2orb.Solvers.ScipyDiscretizationMinimizeWrapper import ScipyDiscretizationMinimizeWrapper

n = 9 # coarse enough to see a small difference from the analytical solution
oneDWorkProblem = OneDWorkProblemModule.OneDWorkProblem()
scipySolver = ScipyDiscretizationMinimizeWrapper(oneDWorkProblem)
ans = scipySolver.ScipyOptimize(n)
plotableNumericalAnswer = scipySolver.ConvertScipyOptimizerOutputToDictionary(ans)

print(ans.success)
print(ans.message)

t = oneDWorkProblem.CreateTimeRange(n)
analyticalAnswerEvaluator = OneDWorkProblemModule.AnalyticalAnswerToProblem()
analyticalAnswer = analyticalAnswerEvaluator.EvaluateAnswer(oneDWorkProblem, t)

fig = plt.figure()
oneDWorkProblem.AddResultsToFigure(fig, t, analyticalAnswer, "Analytical Answer")
oneDWorkProblem.AddResultsToFigure(fig, t, plotableNumericalAnswer, "Numerical Answer")
plt.show()
