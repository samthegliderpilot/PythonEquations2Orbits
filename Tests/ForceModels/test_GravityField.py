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
        u0 = np.array(f(u, t))
        u1 = np.array(f(u + 0.5 * u0, t + 0.5*dt))
        u2 = np.array(f(u + 0.5 * u1, t + 0.5*dt))
        u3 = np.array(f(u +       u2, t +     dt))
        u = u + (1.0/6.0)*dt*( u0 + 2*u1 + 2*u2 + u3)
        usol.append(u)
    return usol, times

def testValidation():
    testDataFilePath = os.path.join(os.path.dirname(__file__), "../testSettings.json")
    settings = json.loads(open(testDataFilePath, "r").read())
    kernelPath = settings["kernelsDirectory"]

    def getCriticalKernelsRelativePaths()-> List[str]:
        criticalKernels = []
        criticalKernels.append("lsk/naif0012.tls")
        criticalKernels.append("pck/earth_latest_high_prec.cmt")
        criticalKernels.append("pck/earth_latest_high_prec.bpc")
        criticalKernels.append("pck/pck00010.tpc")
        criticalKernels.append("pck/gm_de440.tpc")

        return criticalKernels



    allMyKernels = getCriticalKernelsRelativePaths()
    allMyKernels.append("spk/planets/de440s.bsp") # big one here...
    
    with spiceScope(allMyKernels, kernelPath) as scope:
        t =sy.Symbol('t', real=True)
        # an ephemeris file was made from GMAT with the below settings.
        initialPosVel = [7100000.0, 0.0, 1300000.0,   0.0, 7350.0, 1000.0] # m and m/sec, LEO
        etEpoch = 0.0 # TAI at 0.0
        # rk4 at 60 second step
        #4x4 gravity
        fileToRead = os.path.normpath(os.path.join(os.path.dirname(__file__), "../testData/JGM2.cof"))
        data = nsGravity.gravityField.readFromCoefFile(fileToRead)    
        n = 4    
        m = 4
        muVal = data._mu # m^3/sec^2
        x,y,z,vx,vy,vz = sy.symbols('x,y,z,vx,vy,vz', real=True)
        timeVaryingInertialToFixedMatrix = lambda t : spice.pxform("J2000", "IAU_Earth", t)#TODO: Proper fixed frame?
        i2fSymbol = sy.MatrixSymbol("I2F", 3, 3)

        ans = timeVaryingInertialToFixedMatrix(0.0)*sy.Matrix([[x],[y],[z]])
        print(ans)
        rCb = 6378136.3 # from gmat (ultimially from a pck file)
        mu = sy.Symbol(r'\mu', real=True, positive=True)
        twoBodyAccelerationMatrix = TwoBodyAccelerationDifferentialExpression(x,y,z,muVal)
        twoBodyFullOemMatrix = [vx, vy, vz, twoBodyAccelerationMatrix[0], twoBodyAccelerationMatrix[1], twoBodyAccelerationMatrix[2]]
        #sphericalHarmonicGravity = nsGravity.makeOverallAccelerationExpression(4,4, mu, sy.sqrt(x*x+y*y+z*z), rCb, )

        #functionsMap = {}

        twoBodyOemCallback = sy.lambdify([[x,y,z,vx,vy,vz],t], twoBodyFullOemMatrix)
        #solution = solve_ivp(twoBodyOemCallback, [0.0, 12000.0], initialPosVel, method = "RK4", dense_output=True)
        tf =12000
        ts = np.linspace(0, tf, int(tf/60)+1)
        #solution = rungekutta4(twoBodyOemCallback, initialPosVel, ts)
        solution = rk4(twoBodyOemCallback, initialPosVel, ts)
        oemFilePathFromGmat = os.path.join(os.path.dirname(__file__), "../testData/sampleEphemerisFileForSphericalHarmonicGravity.oem")
        truthData = oem.OrbitEphemerisMessage.open(oemFilePathFromGmat)
        posDiff = []
        velDiff = []
        
        for i in range(0, len(ts)):
            truthPosition = np.array(truthData.states[i].position)*1000.0
            truthVelocity = np.array(truthData.states[i].velocity)*1000.0
            evalState = solution[0][i]

            truthRad = math.sqrt(truthPosition[0]**2+truthPosition[1]**2+truthPosition[2]**2)/1000.0
            evalRad =  math.sqrt(evalState[0]**2+evalState[1]**2+evalState[2]**2) / 1000.0
        
            truthVel = math.sqrt(truthVelocity[0]**2+truthVelocity[1]**2+truthVelocity[2]**2)
            evalVel =  math.sqrt(evalState[3]**2+evalState[4]**2+evalState[5]**2)

            posDiff.append(truthRad - evalRad)
            velDiff.append(truthVel - evalVel)
        
        

        # Data for plotting

        fig, ax = plt.subplots()
        ax.plot(ts, posDiff, label="Position (m)")
        ax.plot(ts, velDiff, label="Velocity (km/sec)")

        ax.set(xlabel='time (s)', ylabel='Difference',
            title='Difference')
        ax.grid()
        fig.legend()
        fig.savefig("test.png")
        plt.show()

                
                

