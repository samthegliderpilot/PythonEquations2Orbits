import spiceypy as spice
import sympy as sy
from enum import Enum
from typing import Optional, List, Dict, Any

class rotationMatrixFunction:

    class matrixNameMode(Enum):
        xyz = 1
        ijk =2
        OneTwoThree = 3

    @staticmethod 
    def makeSymbolicMatrix(base : str, subscriptMode : matrixNameMode, args : Optional[List[sy.Symbol]]= None):
        if args is None:
            args = []

        subScripts = []
        if subscriptMode == rotationMatrixFunction.matrixNameMode.xyz:
            subScripts = ["x", "y", "z"]
        elif subscriptMode == rotationMatrixFunction.matrixNameMode.ijk:
            subScripts = ["i", "j", "k"]
        elif subscriptMode == rotationMatrixFunction.matrixNameMode.OneTwoThree:
            subScripts = ["1", "2", "3"]

        # symbolicMatrix = sy.Matrix([[sy.Function(base + '_{' + subScripts[0] + subScripts[0] + "}", real=True)(*args), sy.Function(base + '_{' + subScripts[0] + subScripts[1] + "}", real=True)(*args), sy.Function(base + '_{' + subScripts[0] + subScripts[2] + "}", real=True)(*args)],
        #                        [sy.Function(base + '_{' + subScripts[1] + subScripts[0] + "}", real=True)(*args), sy.Function(base + '_{' + subScripts[1] + subScripts[1] + "}", real=True)(*args), sy.Function(base + '_{' + subScripts[1] + subScripts[2] + "}", real=True)(*args)],
        #                        [sy.Function(base + '_{' + subScripts[2] + subScripts[0] + "}", real=True)(*args), sy.Function(base + '_{' + subScripts[2] + subScripts[1] + "}", real=True)(*args), sy.Function(base + '_{' + subScripts[2] + subScripts[2] + "}", real=True)(*args)]])        
        
        symbolicMatrix = sy.Matrix([[sy.Function(base + '_' + subScripts[0] + subScripts[0] + "", real=True)(*args), sy.Function(base + '_' + subScripts[0] + subScripts[1] + "", real=True)(*args), sy.Function(base + '_' + subScripts[0] + subScripts[2] + "", real=True)(*args)],
                        [sy.Function(base + '_' + subScripts[1] + subScripts[0] + "", real=True)(*args), sy.Function(base + '_' + subScripts[1] + subScripts[1] + "", real=True)(*args), sy.Function(base + '_' + subScripts[1] + subScripts[2] + "", real=True)(*args)],
                        [sy.Function(base + '_' + subScripts[2] + subScripts[0] + "", real=True)(*args), sy.Function(base + '_' + subScripts[2] + subScripts[1] + "", real=True)(*args), sy.Function(base + '_' + subScripts[2] + subScripts[2] + "", real=True)(*args)]])        

        return symbolicMatrix

    def __init__(self, from_frame : str, to_frame  :str):
        self._lastEt = None
        self._lastMatrix = None
        self._from_frame = from_frame
        self._to_frame = to_frame
        rotationMatrixFunction._instance = self
    
    def evaluateMatrix(self, t):
        if self._lastEt == t:
            return self._lastMatrix

        self._lastMatrix = spice.sxform(self._from_frame, self._to_frame, t)[0:3,0:3]
        self._lastEt = t
        return self._lastMatrix

    def callbackForElement(self, m, n, t):
        #TODO: Caching
        return self.evaluateMatrix(t)[m,n]

    def populateRedirectionDictWithCallbacks(self, symbolicMatrix : sy.Matrix, redictionDict : Dict[sy.Symbol, Any], subsDict : Optional[Dict[sy.Symbol, sy.Expr]]=None): #TODO: this any should be a callback of some sort
        # trying to do this with a loop is difficult, the closure over the loop variable 
        # is tricky, so just do it this way...

        nameCleanup = lambda x : str(x).replace("(t)", "")

        if subsDict is not None:
            nameCleanup = lambda x : str(x).replace("(t)", "").replace("{", "").replace("}", "")
            for i in range(0, 3):
                for j in range(0, 3):
                    newSy = sy.Function(nameCleanup(symbolicMatrix[i, j]), real=True)(*symbolicMatrix[i, j].args)
                    subsDict[symbolicMatrix[i, j]] = newSy

        redictionDict[nameCleanup(symbolicMatrix[0,0])] = lambda t : self.callbackForElement(0, 0, t)
        redictionDict[nameCleanup(symbolicMatrix[0,1])] = lambda t : self.callbackForElement(0, 1, t)
        redictionDict[nameCleanup(symbolicMatrix[0,2])] = lambda t : self.callbackForElement(0, 2, t)
        redictionDict[nameCleanup(symbolicMatrix[1,0])] = lambda t : self.callbackForElement(1, 0, t)
        redictionDict[nameCleanup(symbolicMatrix[1,1])] = lambda t : self.callbackForElement(1, 1, t)
        redictionDict[nameCleanup(symbolicMatrix[1,2])] = lambda t : self.callbackForElement(1, 2, t)
        redictionDict[nameCleanup(symbolicMatrix[2,0])] = lambda t : self.callbackForElement(2, 0, t)
        redictionDict[nameCleanup(symbolicMatrix[2,1])] = lambda t : self.callbackForElement(2, 1, t)
        redictionDict[nameCleanup(symbolicMatrix[2,2])] = lambda t : self.callbackForElement(2, 2, t)
