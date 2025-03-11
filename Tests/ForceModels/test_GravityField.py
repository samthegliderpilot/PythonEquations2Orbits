#%%
from mpmath import mp
from pyeq2orb.Numerical.LambdifyHelpers import LambdifyHelper, OdeLambdifyHelper
from pyeq2orb import SafeSubs
import pytest
import math
import sympy as sy
import pyeq2orb.ForceModels.GravityField as nsGravity
from pyeq2orb.ForceModels.TwoBodyForce import TwoBodyAccelerationDifferentialExpression
from pyeq2orb.Spice.spiceScope import spiceScope
import spiceypy as spice
import os
import json
from typing import List
from scipy.integrate import solve_ivp
import oem
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

def testReadingInFile():
    fileToRead = os.path.normpath(os.path.join(os.path.dirname(__file__), "../testData/EGM96.cof"))
    data = nsGravity.gravityField.readFromCoefFile(fileToRead)
    assert data is not None
    assert data.getC(0,0) == 0.0
    assert data.getS(0,0) == 0.0

    assert data.getC(1,0) == 0.0
    assert data.getS(1,0) == 0.0

    assert data.getC(1,1) == 0.0
    assert data.getS(1,1) == 0.0

    # test the first real values
    assert data.getC(2, 0) == -4.84165371736000E-04
    assert data.getC(2, 1) == -1.86987635955000E-10
    assert data.getC(2, 2) == 2.43914352398000E-06

    assert data.getS(2, 0) == 0.0
    assert data.getS(2, 1) == 1.19528012031000E-09
    assert data.getS(2, 2) == -1.40016683654000E-06

    # test a line that has 2 negative values
    assert data.getC(4, 1) == -5.36321616971000E-07
    assert data.getS(4, 1) == -4.73440265853000E-07

    # test the last values
    assert data.getC(360, 360) == -4.47516389678000E-25
    assert data.getS(360, 360) == -8.30224945525000E-11

def testCreateShpericalHarmonicGravityForce():
    mu = sy.Symbol(r'\mu', real=True, positive=True)
    lat = sy.Symbol(r'\gamma')
    lon = sy.Symbol(r'\lambda')
    r = sy.Symbol('r', real=True, positive=True)
    rVec = sy.Matrix([[sy.Symbol('r_x', real=True)], [sy.Symbol('r_y', real=True)],[sy.Symbol('r_z', real=True)]])
    rNorm = rVec/r
    rCb = sy.Symbol(r'r_{E}', real=True, positive=True)
    upToJ3 = nsGravity.makeOverallAccelerationExpression(3,3, mu, r, rCb, lat, lon, r)

    j3 = nsGravity.makeConstantForSphericalHarmonicCoefficient(3,3)

    expectedExprI = (5*j3*mu*rCb**3*rVec[0]/(2*r**7))*(3*rVec[2]-7*rVec[2]**3/(r**2))
    assert 0 == (upToJ3 - expectedExprI).simplify().expand().simplify()

def rk4(f, y0,  times):    
    # Initial conditions
    usol = [y0]
    u = np.copy(y0)
    dt = times[1] - times[0]
    # RK4
    for t in times:
        u0 = np.array(f(t, u))
        u1 = np.array(f(t + 0.5*dt, u + 0.5 * dt* u0))
        u2 = np.array(f(t + 0.5*dt, u + 0.5 * dt* u1))
        u3 = np.array(f(t +     dt, u +       dt* u2))
        u = u + (1.0/6.0)*dt*( u0 + 2*u1 + 2*u2 + u3)
        usol.append(u)
    return usol, times


class RotationMatrixFunction:

    _instance :"RotationMatrixFunction" = None

    def __init__(self):
        self._lastMatrix = None
        self._lastEt = None
        RotationMatrixFunction._instance = self
    
    def evaluateMatrix(self, t):
        if self._lastEt == t:
            return self._lastMatrix

        self._lastMatrix = spice.sxform("J2000", "ITRF93", t)[0:3,0:3]
        self._lastEt = t
        return self._lastMatrix

    def callbackForElement(self, m, n, t):
        return self.evaluateMatrix(t)[m,n]

    def populateRedirectionDictWithCallbacks(self, symbolicMatrix : sy.Matrix, dict):
        dict[str(symbolicMatrix[0,0]).replace("(t)", "")] = lambda t : self.callbackForElement(0, 0, t)
        dict[str(symbolicMatrix[0,1]).replace("(t)", "")] = lambda t : self.callbackForElement(0, 1, t)
        dict[str(symbolicMatrix[0,2]).replace("(t)", "")] = lambda t : self.callbackForElement(0, 2, t)
        dict[str(symbolicMatrix[1,0]).replace("(t)", "")] = lambda t : self.callbackForElement(1, 0, t)
        dict[str(symbolicMatrix[1,1]).replace("(t)", "")] = lambda t : self.callbackForElement(1, 1, t)
        dict[str(symbolicMatrix[1,2]).replace("(t)", "")] = lambda t : self.callbackForElement(1, 2, t)
        dict[str(symbolicMatrix[2,0]).replace("(t)", "")] = lambda t : self.callbackForElement(2, 0, t)
        dict[str(symbolicMatrix[2,1]).replace("(t)", "")] = lambda t : self.callbackForElement(2, 1, t)
        dict[str(symbolicMatrix[2,2]).replace("(t)", "")] = lambda t : self.callbackForElement(2, 2, t)
        # for i in range(0, 3):
        #     for j in range(0, 3):
        #         #dict[f"symbolicMatrix[{i},{j}]"] = parameterName + f".callbackForElement({str(i)}, {str(j)})"
        #         def this_function(me, t):
        #             nonlocal i
        #             nonlocal j
        #             return me.callbackForElement(i, j, t)
        #         dict[str(symbolicMatrix[i,j]).replace("(t)", "")] = lambda t : this_function(self, t)


def testValidation():
    subsDict = {}
    testDataFilePath = os.path.join(os.path.dirname(__file__), "../testSettings.json")
    settings = json.loads(open(testDataFilePath, "r").read())
    kernelPath = settings["kernelsDirectory"]
    spiceBasePath =r'C:\Programs\GMAT\data'# r'/home/sam/Desktop/GMAT/R2022a/data/planetary_coeff/'

    def getCriticalKernelsRelativePaths()-> List[str]:
        criticalKernels = []
        # criticalKernels.append("lsk/naif0012.tls")
        # criticalKernels.append("pck/earth_latest_high_prec.cmt")
        # criticalKernels.append("pck/earth_latest_high_prec.bpc")
        # criticalKernels.append("pck/pck00010.tpc")
        # criticalKernels.append("pck/gm_de440.tpc")
        criticalKernels.append(os.path.join(spiceBasePath, "time/SPICELeapSecondKernel.tls"))
        criticalKernels.append(os.path.join(spiceBasePath, "time/tai-utc.dat"))        
        criticalKernels.append(os.path.join(spiceBasePath, "planetary_coeff/SPICEEarthPredictedKernel.bpc"))
        criticalKernels.append(os.path.join(spiceBasePath, "planetary_coeff/SPICEEarthCurrentKernel.bpc"))
        criticalKernels.append(os.path.join(spiceBasePath, "planetary_coeff/earth_latest_high_prec.bpc"))
        criticalKernels.append(os.path.join(spiceBasePath, "planetary_coeff/SPICEPlanetaryConstantsKernel.tpc"))
        #criticalKernels.append(os.path.join(spiceBasePath, "planetary_coeff/NUTATION.DAT"))
        #criticalKernels.append(os.path.join(spiceBasePath, "planetary_coeff/eopc04_08.62-now"))
        criticalKernels.append(r'Y:\kernels\pck\pck00011.tpc')
        criticalKernels.append(r'Y:\kernels\spk\planets\de440.bsp')

        return criticalKernels



    allMyKernels = getCriticalKernelsRelativePaths()
    #allMyKernels.append(os.path.join(spiceBasePath, "planetary_ephem/spk/DE405AllPlanets.bsp"))
    #allMyKernels.append(os.path.join(spiceBasePath, "planetary_ephem/spk/DE405AllPlanets.bsp"))
    #allMyKernels.append(os.path.join(spiceBasePath, "planetary_ephem/de/leDE1941.405")) # big one here...
    
    with spiceScope(allMyKernels, kernelPath) as scope:
        t =sy.Symbol('t', real=True)
        # an ephemeris file was made from GMAT with the below settings.
        initialPosVel = [7100000.0, 0.0, 1300000.0,   0.0, 7350.0, 1000.0] # m and m/sec, LEO
        etEpoch = 0.0 # TAI at 0.0
        # rk4 at 60 second step
        #4x4 gravity
        
        fileToRead = os.path.normpath(os.path.join(os.path.dirname(__file__), "../testData/JGM2.cof"))
        data = nsGravity.gravityField.readFromCoefFile(fileToRead)    
        nVal = 4
        mVal = 4
        muVal = data._mu # m^3/sec^2
        x,y,z,vx,vy,vz = sy.symbols('x,y,z,vx,vy,vz', real=True)
        xf,yf,zf = sy.symbols('x_f,y_f,z_f', real=True)
        timeVaryingInertialToFixedMatrix = lambda t : spice.sxform("J2000", "ITRF93", t)[0:3,0:3] #I think gmat is using ITRF93 as their ECEF
        i2fSymbol = sy.Matrix([[sy.Function('Rxx', real=True)(t), sy.Function('Rxy', real=True)(t), sy.Function('Rxz', real=True)(t)],
                               [sy.Function('Ryx', real=True)(t), sy.Function('Ryy', real=True)(t), sy.Function('Ryz', real=True)(t)],
                               [sy.Function('Rzx', real=True)(t), sy.Function('Rzy', real=True)(t), sy.Function('Rzz', real=True)(t)]])

        fixedPositionVector = i2fSymbol*sy.Matrix([[x], [y], [z]])
        fixedPositionVectorSy = [xf, yf, zf]
        
        ans = timeVaryingInertialToFixedMatrix(etEpoch)*sy.Matrix([[x],[y],[z]])
        ans1point5 =(timeVaryingInertialToFixedMatrix(etEpoch)@np.array([initialPosVel[0],initialPosVel[1],initialPosVel[2]]))
        norm = math.sqrt(initialPosVel[0]**2+initialPosVel[1]**2+initialPosVel[2]**2)
        ans2 = ans1point5/norm
        lat = math.asin(ans2[2])


        display(ans2)
        display("lat: " + str(180.0*lat/math.pi))
        display("lon: " + str(180.0*math.acos(ans2[0]/math.cos(lat))/math.pi))
        display("should be " + str(10.43478258117238) + "  " + str(79.80635133432148 ))
        rCb = 6378136.3 # from gmat (ultimially from a pck file)
        rCbSy = sy.Symbol('r_e', real=True, positive=True)
        mu = sy.Symbol(r'\mu', real=True, positive=True)
        rSy = sy.Symbol(r'r', real=True, positive=True)
        latSy =sy.Symbol(r'\gamma', real=True)
        lonSy =sy.Symbol(r'\lambda', real=True)

        fixedPositionVectorNorm = fixedPositionVector /rSy
        subsDict[mu] = muVal
        subsDict[rCbSy] = rCb
        subsDict[sy.sin(latSy)] = fixedPositionVectorNorm[2]
        subsDict[sy.cos(latSy)] = sy.sqrt(fixedPositionVectorNorm[0]**2 + fixedPositionVectorNorm[1]**2)

        subsDict[sy.cos(lonSy)] = fixedPositionVectorNorm[0] / (sy.cos(latSy))
        subsDict[sy.sin(lonSy)] = fixedPositionVectorNorm[1] / (sy.cos(latSy))

        subsDict[xf] = fixedPositionVector[0]
        subsDict[yf] = fixedPositionVector[1]
        subsDict[zf] = fixedPositionVector[2]

        subsDict[rSy] = sy.sqrt(x**2 +y**2 +z**2)
        twoBodyAccelerationMatrix = TwoBodyAccelerationDifferentialExpression(x,y,z,muVal)
        twoBodyFullOemMatrix = sy.Matrix([[vx, vy, vz, twoBodyAccelerationMatrix[0], twoBodyAccelerationMatrix[1], twoBodyAccelerationMatrix[2]]]).transpose()
        #sphericalHarmonicGravity = nsGravity.makeOverallAccelerationExpression(4,4, mu, sy.sqrt(x*x+y*y+z*z), rCb, )

        #functionsMap = {}

        for n in range(0, nVal+1):
            for m in range(0, n+1):
                subsDict[nsGravity.makeConstantForSphericalHarmonicCoefficient("C", n, m)] = data.getC(n, m)
                subsDict[nsGravity.makeConstantForSphericalHarmonicCoefficient("S", n, m)] = data.getS(n, m)
                
        fixToInertial = i2fSymbol.transpose()
        nsGravityExpression = fixToInertial*nsGravity.makeOverallAccelerationExpression(nVal, mVal, mu, rSy, rCbSy, latSy, lonSy, fixedPositionVectorSy)
        fullNsGravity = twoBodyFullOemMatrix+sy.Matrix([[0.0, 0.0, 0.0,nsGravityExpression[0], nsGravityExpression[1], nsGravityExpression[2]]]).transpose()
        #display(fullNsGravity[3])
        helper = OdeLambdifyHelper(t, [x,y,z,vx,vy,vz], fullNsGravity, [], subsDict)

        rotHelper = RotationMatrixFunction()
        rotHelper.populateRedirectionDictWithCallbacks(i2fSymbol, helper.FunctionRedirectionDictionary)
        odeCallback = helper.CreateSimpleCallbackForSolveIvp()
                
        twoBodyHelper = OdeLambdifyHelper(t, [x,y,z,vx,vy,vz], twoBodyFullOemMatrix, [], subsDict)
        twoBodyOemCallback = twoBodyHelper.CreateSimpleCallbackForSolveIvp()

        

        
        #solution = solve_ivp(twoBodyOemCallback, [0.0, 12000.0], initialPosVel, method = "RK4", dense_output=True)
        tf =12000
        ts = np.linspace(etEpoch, tf, int(tf/60)+1)
        solutionTwoBody = rk4(twoBodyOemCallback, initialPosVel, ts)
        #solution = rk4(_lambdifygenerated, initialPosVel, ts)
        solution = rk4(odeCallback, initialPosVel, ts)
        oemFilePathFromGmat = os.path.join(os.path.dirname(__file__), "../testData/sampleEphemerisFileForSphericalHarmonicGravity.oem")
        truthData = oem.OrbitEphemerisMessage.open(oemFilePathFromGmat)
        posDiff = []
        velDiff = []
        twoBodyPosDiff = []
        twoBodyVelDiff = []        
        xDiff = []
        yDiff = []
        zDiff = []
        for i in range(0, len(ts)):
            truthPosition = np.array(truthData.states[i].position)*1000.0
            truthVelocity = np.array(truthData.states[i].velocity)*1000.0
            evalState = solution[0][i]

            twoBodyState = solutionTwoBody[0][i]

            truthRad = math.sqrt((truthPosition[0]**2)+(truthPosition[1]**2)+(truthPosition[2]**2))/1000.0
            evalRad =  math.sqrt((evalState[0]**2)+(evalState[1]**2)+(evalState[2]**2)) / 1000.0
            twoBodyRad = math.sqrt(twoBodyState[0]**2+twoBodyState[1]**2+twoBodyState[2]**2) / 1000.0
        
            truthVel = math.sqrt(truthVelocity[0]**2+truthVelocity[1]**2+truthVelocity[2]**2)
            evalVel =  math.sqrt(evalState[3]**2+evalState[4]**2+evalState[5]**2)
            twoBodyVel =  math.sqrt(twoBodyState[3]**2+twoBodyState[4]**2+twoBodyState[5]**2)

            posDiff.append(truthRad - evalRad)
            velDiff.append(truthVel - evalVel)
        
            twoBodyPosDiff.append(truthRad - twoBodyRad)
            twoBodyVelDiff.append(truthVel - twoBodyVel)

            xDiff.append(truthPosition[0]/1000-evalState[0]/1000)
            yDiff.append(truthPosition[1]/1000-evalState[1]/1000)
            zDiff.append(truthPosition[2]/1000-evalState[2]/1000)

        

        # Data for plotting

        fig, ax = plt.subplots()
        ax.plot(ts, posDiff, label="Position (km)")
        ax.plot(ts, xDiff, label="X (km)")        
        ax.plot(ts, yDiff, label="Y (km)")
        ax.plot(ts, zDiff, label="Z (km)")
        #ax.plot(ts, velDiff, label="Velocity (m/sec)")

        ax.plot(ts, twoBodyPosDiff, label="2B Position (km)")
        ax.plot(ts, twoBodyVelDiff, label="2B Velocity (m/sec)")


        ax.set(xlabel='time (s)', ylabel='Difference',
            title='Difference')
        ax.grid()
        fig.legend()
        fig.savefig("test.png")
        plt.show()

if __name__ == "__main__":
    testValidation()
