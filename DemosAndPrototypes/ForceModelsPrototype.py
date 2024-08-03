#%%
import __init__ #type: ignore
import sympy as sy
import math
import os
import sys
from IPython.display import display
from collections import OrderedDict
sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0]))) # need to import 2 directories up (so pyeq2orb is a subfolder)
sy.init_printing()
import scipyPaperPrinter as jh #type: ignore
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from pyeq2orb.Coordinates.CartesianModule import MotionCartesian
from pyeq2orb.Coordinates.QuaternionModule import MotionQuaternion

class NumericallyEvaluatableAxes(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def Evaluate(self, x :datetime) -> MotionQuaternion:
        pass 

class AxesTimeVarying(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def GetExpression(self) ->sy.Expr :
        pass

    @abstractmethod
    def CreateNumericallyEvaluatableAxes(self) -> NumericallyEvaluatableAxes:
        pass



class NumericallyEvaluatablePoint(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def Evaluate(self, x :datetime) -> MotionCartesian:
        pass

class PointTimeVarying(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def GetExpression(self) ->sy.Expr :
        pass

    @abstractmethod
    def CreateNumericallyEvaluatablePoint(self) -> NumericallyEvaluatablePoint:
        pass



class NumericallyEvaluatableVector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def Evaluate(self, x :datetime) -> MotionCartesian:
        pass

class VectorTimeVarying(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def GetExpression(self) ->sy.Expr :
        pass    

    @abstractmethod
    def CreateNumericallyEvaluatableVector(self) -> NumericallyEvaluatableVector:
        pass    