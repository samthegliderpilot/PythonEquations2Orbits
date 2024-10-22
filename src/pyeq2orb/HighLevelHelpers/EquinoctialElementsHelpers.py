
import sympy as sy
from pyeq2orb.ForceModels.TwoBodyForce import CreateTwoBodyMotionMatrix, CreateTwoBodyListForModifiedEquinoctialElements
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule import ModifiedEquinoctialElements, CreateSymbolicElements
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from pyeq2orb import SafeSubs
from pyeq2orb.Numerical.LambdifyHelpers import LambdifyHelper, OdeLambdifyHelper, OdeLambdifyHelperWithBoundaryConditions
import scipyPaperPrinter as jh#type: ignore
import math as math
from typing import Union, Dict, List, Callable, Any, Optional, Tuple
import pyeq2orb.Graphics.Primitives as prim
from pyeq2orb.Graphics.PlotlyUtilities import PlotAndAnimatePlanetsWithPlotly
from pyeq2orb.Numerical.ScalingHelpers import scaledEquationOfMotionHolder
from IPython.display import display
import numpy as np
import pyeq2orb.Coordinates.OrbitFunctions as orb

class ModifiedEquinoctialElementsHelpers:
    @staticmethod
    def createSatPathFromIvpSolution(ivpSolution, mu, color : str, scalingFactors : Optional[List[float]] = None, timeScalingFactor : Optional[float] = 1)->prim.PlanetPrimitive:
        
        (tArray, satArrays) = OdeLambdifyHelper.SolveIvpResultsReshaped(ivpSolution)
        if scalingFactors is not None or timeScalingFactor != 1:
            if scalingFactors == None:
                scalingFactors = []
            tArray, satArrays = scaledEquationOfMotionHolder.descaleStates(tArray, satArrays, scalingFactors, tfVal)
        (satMees) = [ModifiedEquinoctialElements(*x[:6], mu) for x in satArrays]
        motions = ModifiedEquinoctialElements.CreateEphemeris(satMees)
        satPath = prim.PlanetPrimitive.fromMotionEphemeris(tArray, motions, color)
        return satPath    

     

    @staticmethod
    def getInertialThrustVectorFromDataDict(az : List[float], el: List[float], mag: List[float], equiElements : List[ModifiedEquinoctialElements]) -> List[Cartesian] :
        cartesians = []
        for i in range(0, len(az)) :
            x = mag[i] * math.cos(az[i])*math.cos(el[i])
            y = mag[i] * math.sin(az[i])*math.cos(el[i])
            z = mag[i] * math.sin(el[i])      
            equiElement = equiElements[i]
            ricToInertial = orb.CreateComplicatedRicToInertialMatrix(equiElement.ToMotionCartesian())
            cartesians.append(Cartesian(*(ricToInertial*Cartesian(x,y,z))))
        return cartesians

    @staticmethod
    def createScattersForThrustVectors(ephemeris : prim.EphemerisArrays, inertialThrustVectors : List[Cartesian], scale) -> List[Tuple[List[float], List[float]]] :
        scats = []
        for i in range(0, len(ephemeris.T)) :
            hereX = ephemeris.X[i]
            hereY = ephemeris.Y[i]
            hereZ = ephemeris.Z[i]

            thereX = inertialThrustVectors[i].X*scale + hereX
            thereY = inertialThrustVectors[i].Y*scale + hereY
            thereZ = inertialThrustVectors[i].Z*scale + hereZ
            scats.append(([hereX, hereY, hereZ], [thereX, thereY, thereZ]))
        return scats        