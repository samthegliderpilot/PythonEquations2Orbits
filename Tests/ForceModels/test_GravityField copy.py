# # #%%
# # from mpmath import mp
# # from pyeq2orb.Numerical.LambdifyHelpers import LambdifyHelper, OdeLambdifyHelper
# # from pyeq2orb import SafeSubs
# # import pytest
# # import math
# # import sympy as sy
# # import pyeq2orb.ForceModels.GravityField as nsGravity
# # from pyeq2orb.ForceModels.TwoBodyForce import TwoBodyAccelerationDifferentialExpression
# # from pyeq2orb.Spice.spiceScope import spiceScope
# # import spiceypy as spice
# # import os
# # import json
# # from typing import List
# # from scipy.integrate import solve_ivp
# # import oem
# # import matplotlib.pyplot as plt
# # import numpy as np
# # from IPython.display import display
# # from pyeq2orb.Spice.rotationMatrixWrapper import rotationMatrixFunction
# # from collections import OrderedDict
# # def testReadingInFile():
# #     fileToRead = os.path.normpath(os.path.join(os.path.dirname(__file__), "../testData/EGM96.cof"))
# #     data = nsGravity.gravityField.readFromCoefFile(fileToRead)
# #     assert data is not None
# #     assert data.getC(0,0) == 0.0
# #     assert data.getS(0,0) == 0.0

# #     assert data.getC(1,0) == 0.0
# #     assert data.getS(1,0) == 0.0

# #     assert data.getC(1,1) == 0.0
# #     assert data.getS(1,1) == 0.0

# #     # test the first real values
# #     assert data.getC(2, 0) == -4.84165371736000E-04
# #     assert data.getC(2, 1) == -1.86987635955000E-10
# #     assert data.getC(2, 2) == 2.43914352398000E-06

# #     assert data.getS(2, 0) == 0.0
# #     assert data.getS(2, 1) == 1.19528012031000E-09
# #     assert data.getS(2, 2) == -1.40016683654000E-06

# #     # test a line that has 2 negative values
# #     assert data.getC(4, 1) == -5.36321616971000E-07
# #     assert data.getS(4, 1) == -4.73440265853000E-07

# #     # test the last values
# #     assert data.getC(360, 360) == -4.47516389678000E-25
# #     assert data.getS(360, 360) == -8.30224945525000E-11

# # def testCreateShpericalHarmonicGravityForce():
# #     mu = sy.Symbol(r'\mu', real=True, positive=True)
# #     lat = sy.Symbol(r'\gamma')
# #     lon = sy.Symbol(r'\lambda')
# #     r = sy.Symbol('r', real=True, positive=True)
# #     rVec = sy.Matrix([[sy.Symbol('r_x', real=True)], [sy.Symbol('r_y', real=True)],[sy.Symbol('r_z', real=True)]])
# #     rNorm = rVec/r
# #     rCb = sy.Symbol(r'r_{E}', real=True, positive=True)
# #     upToJ3 = nsGravity.makeOverallAccelerationExpression(3,3, mu, r, rCb, lat, lon, r)

# #     j3 = nsGravity.makeConstantForSphericalHarmonicCoefficient(3,3)

# #     expectedExprI = (5*j3*mu*rCb**3*rVec[0]/(2*r**7))*(3*rVec[2]-7*rVec[2]**3/(r**2))
# #     assert 0 == (upToJ3 - expectedExprI).simplify().expand().simplify()

# # def rk4(f, y0,  times):    
# #     # Initial conditions
# #     usol = [y0]
# #     u = np.copy(y0)
# #     dt = times[1] - times[0]
# #     # RK4
# #     for t in times:
# #         u0 = np.array(f(t, u))
# #         u1 = np.array(f(t + 0.5*dt, u + 0.5 * dt* u0))
# #         u2 = np.array(f(t + 0.5*dt, u + 0.5 * dt* u1))
# #         u3 = np.array(f(t +     dt, u +       dt* u2))
# #         u = u + (1.0/6.0)*dt*( u0 + 2*u1 + 2*u2 + u3)
# #         usol.append(u)
# #     return usol, times





# # testDataFilePath = os.path.join(os.path.dirname(__file__), "../testSettings.json")
# # settings = json.loads(open(testDataFilePath, "r").read())
# # kernelPath = settings["kernelsDirectory"]
# # spiceBasePath =r'C:\Programs\GMAT\data'# r'/home/sam/Desktop/GMAT/R2022a/data/planetary_coeff/'

# # def getCriticalKernelsRelativePaths()-> List[str]:
# #     criticalKernels = []
# #     # # criticalKernels.append("lsk/naif0012.tls")
# #     # # criticalKernels.append("pck/earth_latest_high_prec.cmt")
# #     # # criticalKernels.append("pck/earth_latest_high_prec.bpc")
# #     # # criticalKernels.append("pck/pck00010.tpc")
# #     # # criticalKernels.append("pck/gm_de440.tpc")
# #     # criticalKernels.append(os.path.join(spiceBasePath, "time/SPICELeapSecondKernel.tls"))
# #     criticalKernels.append(os.path.join(spiceBasePath, "time/tai-utc.dat"))        
# #     #criticalKernels.append(r'Y:\kernels\spk\planets\de440.bsp')
# #     criticalKernels.append(os.path.join(spiceBasePath, "time/SPICELeapSecondKernel.tls"))
# #     criticalKernels.append(os.path.join(spiceBasePath, "planetary_coeff/SPICEEarthPredictedKernel.bpc"))
# #     criticalKernels.append(os.path.join(spiceBasePath, "planetary_coeff/SPICEEarthCurrentKernel.bpc"))
# #     criticalKernels.append(os.path.join(spiceBasePath, "planetary_coeff/earth_latest_high_prec.bpc"))
# #     criticalKernels.append(os.path.join(spiceBasePath, "planetary_coeff/NUTATION.DAT"))
# #     criticalKernels.append(os.path.join(spiceBasePath, "planetary_coeff/eopc04_08.62-now"))
# #     criticalKernels.append(os.path.join(spiceBasePath, "planetary_ephem/de/leDE1941.405"))
# #     criticalKernels.append(os.path.join(spiceBasePath, "planetary_ephem/spk/DE424AllPlanets.bsp"))
    
    
# #     # criticalKernels.append(r'Y:\kernels\pck\pck00011.tpc')
    
# #     #criticalKernels.append(r'Y:\kernels\pck\earth_latest_high_prec.bpc')
# #     #criticalKernels.append(r'Y:\kernels\pck\earth_latest_high_prec.cmt')

# #     # criticalKernels.append(r'Y:\kernels\lsk\naif0012.tls')
# #     # criticalKernels.append(r'Y:\kernels\spk\planets\de441_part-1.bsp')
# #     # criticalKernels.append(r'Y:\kernels\spk\planets\de441_part-2.bsp')
# #     # criticalKernels.append(r'Y:\kernels\spk\stations\earthstns_itrf93_201023.bsp')
# #     # criticalKernels.append(r'Y:\kernels\fk\stations\earth_topo_201023.tf')
# #     # criticalKernels.append(r'Y:\kernels\pck\pck00010.tpc')       
# #     # criticalKernels.append(r'Y:\kernels\pck\earth_1962_240827_2124_combined.bpc')
# #     # criticalKernels.append(r'C:\Users\enfor\Downloads\spice_lessons_c_win\lessons\binary_pck\kernels\fk\earth_assoc_itrf93.tf')
    
# #     #criticalKernels.append(r'Y:\kernels\fk\planets\earth_assoc_itrf93.tf')

# #     return criticalKernels



# # allMyKernels = getCriticalKernelsRelativePaths()

# # def testValidation():
# #     subsDict = {}
    


# #     #allMyKernels.append(os.path.join(spiceBasePath, "planetary_ephem/spk/DE405AllPlanets.bsp"))
# #     #allMyKernels.append(os.path.join(spiceBasePath, "planetary_ephem/spk/DE405AllPlanets.bsp"))
# #     #allMyKernels.append(os.path.join(spiceBasePath, "planetary_ephem/de/leDE1941.405")) # big one here...
    
# #     with spiceScope(allMyKernels, kernelPath) as scope:
# #         for kernel in allMyKernels:
# #             spice.furnsh(kernel)
# #         t =sy.Symbol('t', real=True)
# #         # an ephemeris file was made from GMAT with the below settings.
# #         initialPosVel = [7100000.0, 0.0, 1300000.0,   0.0, 7350.0, 1000.0] # m and m/sec, LEO
# #         etEpoch = 0.0 # TAI at 0.0
# #         # rk4 at 60 second step
# #         #4x4 gravity
        
# #         fileToRead = os.path.normpath(os.path.join(os.path.dirname(__file__), "../testData/JGM2.cof"))
# #         data = nsGravity.gravityField.readFromCoefFile(fileToRead)    
# #         nVal = 4
# #         mVal = 4
# #         muVal = data._mu # m^3/sec^2
# #         x,y,z,vx,vy,vz = sy.symbols('x,y,z,vx,vy,vz', real=True)
# #         xf,yf,zf = sy.symbols('x_f,y_f,z_f', real=True)
# #         timeVaryingInertialToFixedMatrix = lambda t : spice.sxform("J2000", "ITRF93", t)[0:3,0:3] #I think gmat is using ITRF93 as their ECEF
# #         i2fSymbol = rotationMatrixFunction.makeSymbolicMatrix("R", rotationMatrixFunction.matrixNameMode.xyz, [t])
# #         # sy.Matrix([[sy.Function('Rxx', real=True)(t), sy.Function('Rxy', real=True)(t), sy.Function('Rxz', real=True)(t)],
# #         #                        [sy.Function('Ryx', real=True)(t), sy.Function('Ryy', real=True)(t), sy.Function('Ryz', real=True)(t)],
# #         #                        [sy.Function('Rzx', real=True)(t), sy.Function('Rzy', real=True)(t), sy.Function('Rzz', real=True)(t)]])

# #         fixedPositionVector = i2fSymbol*sy.Matrix([[x], [y], [z]])
# #         fixedPositionVectorSy = [xf, yf, zf]
        
# #         ans = timeVaryingInertialToFixedMatrix(etEpoch)*sy.Matrix([[x],[y],[z]])
# #         ans1point5 =(timeVaryingInertialToFixedMatrix(etEpoch)@np.array([initialPosVel[0],initialPosVel[1],initialPosVel[2]]))
# #         norm = math.sqrt(initialPosVel[0]**2+initialPosVel[1]**2+initialPosVel[2]**2)
# #         ans2 = ans1point5/norm
# #         lat = math.asin(ans2[2])


# #         display(ans2)
# #         display("lat: " + str(180.0*lat/math.pi))
# #         display("lon: " + str(180.0*math.acos(ans2[0]/math.cos(lat))/math.pi))
# #         display("should be " + str(10.43478258117238) + "  " + str(79.80635133432148 ))
# #         rCb = 6378136.3 # from gmat (ultimially from a pck file)
# #         rCbSy = sy.Symbol('r_e', real=True, positive=True)
# #         mu = sy.Symbol(r'mu', real=True, positive=True)
# #         rSy = sy.Symbol(r'r', real=True, positive=True)
# #         latSy =sy.Symbol(r'gamma', real=True)
# #         lonSy =sy.Symbol(r'lambda', real=True)
# #         #latSy =sy.Function(r'gamma', real=True)(t, x, y, z)
# #         #lonSy =sy.Function(r'lon', real=True)(t, x, y, z)

# #         fixedPositionVectorNorm = fixedPositionVector /rSy
# #         subsDict[mu] = muVal
# #         subsDict[rCbSy] = rCb
# #         subsDict[sy.sin(latSy)] = fixedPositionVectorNorm[2]
# #         subsDict[sy.cos(latSy)] = sy.sqrt(fixedPositionVectorNorm[0]**2 + fixedPositionVectorNorm[1]**2)
# #         subsDict[sy.tan(latSy)] = fixedPositionVectorNorm[2]/sy.sqrt(fixedPositionVectorNorm[0]**2 + fixedPositionVectorNorm[1]**2)

# #         subsDict[sy.cos(lonSy)] = fixedPositionVectorNorm[0] / (sy.cos(latSy))
# #         subsDict[sy.sin(lonSy)] = fixedPositionVectorNorm[1] / (sy.cos(latSy))
# #         #subsDict[sy.Abs(sy.sin(lat))]=sy.sin(lat)
# #         subsDict[sy.Abs(sy.cos(lat))]=sy.cos(lat)
# #         #subsDict[sy.Abs(sy.tan(lat))]=sy.tan(lat)
        

# #         subsDict[xf] = fixedPositionVector[0]
# #         subsDict[yf] = fixedPositionVector[1]
# #         subsDict[zf] = fixedPositionVector[2]

# #         subsDict[rSy] = sy.sqrt(xf**2 +yf**2 +zf**2)
# #         twoBodyAccelerationMatrix = TwoBodyAccelerationDifferentialExpression(x,y,z,muVal)
# #         twoBodyFullOemMatrix = sy.Matrix([[vx, vy, vz, twoBodyAccelerationMatrix[0], twoBodyAccelerationMatrix[1], twoBodyAccelerationMatrix[2]]]).transpose()

# #         for n in range(0, nVal+1):
# #             for m in range(0, n+1):
# #                 k = 2
# #                 if m == 0:
# #                     k = 1
# #                 subsDict[nsGravity.makeConstantForSphericalHarmonicCoefficient("C", n, m)] = data.getC(n, m) #/ (math.sqrt(math.factorial(n+m)/(math.factorial(n-m) *k* (2*n+1))))
# #                 subsDict[nsGravity.makeConstantForSphericalHarmonicCoefficient("S", n, m)] = data.getS(n, m) #/ (math.sqrt(math.factorial(n+m)/(math.factorial(n-m) *k* (2*n+1))))
                
# #         fixToInertial = rotationMatrixFunction.makeSymbolicMatrix("RI", rotationMatrixFunction.matrixNameMode.xyz, [t])
# #         #nsGravityInFixed = nsGravity.makeOverallAccelerationExpression(nVal, mVal, mu, rSy, rCbSy, latSy, lonSy, fixedPositionVectorSy)
# #         nSy = sy.Inx
# #         nsGravityInFixed = nsGravity.makeOverallAccelerationExpressionAsSums()

# #         # toDisplay = SafeSubs(nsGravityInFixed, subsDict)
# #         # display(toDisplay)
# #         nsGravityExpression = fixToInertial*nsGravityInFixed 
# #         fullNsGravity = twoBodyFullOemMatrix+sy.Matrix([[0.0, 0.0, 0.0,nsGravityExpression[0], nsGravityExpression[1], nsGravityExpression[2]]]).transpose()
# #         #display(fullNsGravity[3])
# #         helper = OdeLambdifyHelper(t, [x,y,z,vx,vy,vz], fullNsGravity, [], subsDict)

# #         rotHelperInertialToFixed = rotationMatrixFunction("J2000", "ITRF93")
# #         rotHelperInertialToFixed.populateRedirectionDictWithCallbacks(i2fSymbol, helper.FunctionRedirectionDictionary, subsDict)

# #         rotHelperFixedtoInertial = rotationMatrixFunction("ITRF93", "J2000")
# #         rotHelperFixedtoInertial.populateRedirectionDictWithCallbacks(fixToInertial, helper.FunctionRedirectionDictionary, subsDict)

# #         # def evalLat(t, x, y, z)->float:
# #         #     from_frame = "J2000"        # Inertial frame (e.g., J2000)
# #         #     to_frame = "ITRF93"  # Earth-fixed frame
            
# #         #     # Get the rotation matrix from Earth-fixed to inertial frame
# #         #     rotation_matrix = spice.pxform(from_frame, to_frame, t)
# #         #     fixed = rotation_matrix@[x, y, z]
# #         #     recGeo = spice.recgeo(fixed, 6378136.3, 0.0033527)
            
# #         #     return recGeo[1]

# #         # def evalLon(t, x, y, z) ->float:
# #         #     from_frame = "J2000"        # Inertial frame (e.g., J2000)
# #         #     to_frame = "ITRF93"  # Earth-fixed frame
            
# #         #     # Get the rotation matrix from Earth-fixed to inertial frame
# #         #     rotation_matrix = spice.pxform(from_frame, to_frame, t)
# #         #     fixed = rotation_matrix@[x, y, z]
# #         #     recGeo = spice.recgeo(fixed, 6378136.3, 0.0033527)
            
# #         #     return recGeo[0]

# #         # helper.FunctionRedirectionDictionary["gamma"] = evalLat
# #         # helper.FunctionRedirectionDictionary["lon"] = evalLon
# #         odeCallback = helper.CreateSimpleCallbackForSolveIvp()
# #         print(odeCallback.__doc__)
# #         twoBodyHelper = OdeLambdifyHelper(t, [x,y,z,vx,vy,vz], twoBodyFullOemMatrix, [], subsDict)
# #         twoBodyOemCallback = twoBodyHelper.CreateSimpleCallbackForSolveIvp()

              

        
# #         #solution = solve_ivp(twoBodyOemCallback, [0.0, 12000.0], initialPosVel, method = "RK4", dense_output=True)
# #         tf =12000
# #         ts = np.linspace(etEpoch, tf, int(tf/60)+1)
# #         solutionTwoBody = rk4(twoBodyOemCallback, initialPosVel, ts)
# #         #solution = rk4(_lambdifygenerated, initialPosVel, ts)
# #         solution = rk4(odeCallback, initialPosVel, ts)
# #         oemFilePathFromGmat = os.path.join(os.path.dirname(__file__), "../testData/sampleEphemerisFileForSphericalHarmonicGravity.oem")
# #         truthData = oem.OrbitEphemerisMessage.open(oemFilePathFromGmat)
# #         posDiff = []
# #         velDiff = []
# #         twoBodyPosDiff = []
# #         twoBodyVelDiff = []        
# #         xDiff = []
# #         yDiff = []
# #         zDiff = []
# #         for i in range(0, len(ts)):
# #             truthPosition = np.array(truthData.states[i].position)*1000.0
# #             truthVelocity = np.array(truthData.states[i].velocity)*1000.0
# #             evalState = solution[0][i]

# #             twoBodyState = solutionTwoBody[0][i]

# #             truthRad = math.sqrt((truthPosition[0]**2)+(truthPosition[1]**2)+(truthPosition[2]**2))/1000.0
# #             evalRad =  math.sqrt((evalState[0]**2)+(evalState[1]**2)+(evalState[2]**2)) / 1000.0
# #             twoBodyRad = math.sqrt(twoBodyState[0]**2+twoBodyState[1]**2+twoBodyState[2]**2) / 1000.0
        
# #             truthVel = math.sqrt(truthVelocity[0]**2+truthVelocity[1]**2+truthVelocity[2]**2)
# #             evalVel =  math.sqrt(evalState[3]**2+evalState[4]**2+evalState[5]**2)
# #             twoBodyVel =  math.sqrt(twoBodyState[3]**2+twoBodyState[4]**2+twoBodyState[5]**2)

# #             posDiff.append(truthRad - evalRad)
# #             velDiff.append(truthVel - evalVel)
        
# #             twoBodyPosDiff.append(truthRad - twoBodyRad)
# #             twoBodyVelDiff.append(truthVel - twoBodyVel)

# #             xDiff.append(truthPosition[0]/1000-evalState[0]/1000)
# #             yDiff.append(truthPosition[1]/1000-evalState[1]/1000)
# #             zDiff.append(truthPosition[2]/1000-evalState[2]/1000)

        

# #         # Data for plotting

# #         fig, ax = plt.subplots()
# #         ax.plot(ts, posDiff, label="Position (km)")
# #         ax.plot(ts, xDiff, label="X (km)")        
# #         ax.plot(ts, yDiff, label="Y (km)")
# #         ax.plot(ts, zDiff, label="Z (km)")
# #         ax.plot(ts, velDiff, label="Velocity (m/sec)")

# #         # ax.plot(ts, twoBodyPosDiff, label="2B Position (km)")
# #         # ax.plot(ts, twoBodyVelDiff, label="2B Velocity (m/sec)")


# #         ax.set(xlabel='time (s)', ylabel='Difference',
# #             title='Difference')
# #         ax.grid()
# #         fig.legend()
# #         fig.savefig("test.png")
# #         plt.show()


# # def R_xx(t):
# #     return spice.sxform("J2000", "ITRF93", t)[0:3,0:3][0,0]

# # def R_xy(t):
# #     return spice.sxform("J2000", "ITRF93", t)[0:3,0:3][0,1]

# # def R_xz(t):
# #     return spice.sxform("J2000", "ITRF93", t)[0:3,0:3][0,2]

# # def R_yx(t):
# #     return spice.sxform("J2000", "ITRF93", t)[0:3,0:3][1,0]

# # def R_yy(t):
# #     return spice.sxform("J2000", "ITRF93", t)[0:3,0:3][1,1]

# # def R_yz(t):
# #     return spice.sxform("J2000", "ITRF93", t)[0:3,0:3][1,2]

# # def R_zx(t):
# #     return spice.sxform("J2000", "ITRF93", t)[0:3,0:3][2,0]

# # def R_zy(t):
# #     return spice.sxform("J2000", "ITRF93", t)[0:3,0:3][2,1]

# # def R_zz(t):
# #     return spice.sxform("J2000", "ITRF93", t)[0:3,0:3][2,2]




# # def RI_xx(t):
# #     return spice.sxform("ITRF93", "J2000", t)[0:3,0:3][0,0]

# # def RI_xy(t):
# #     return spice.sxform("ITRF93", "J2000", t)[0:3,0:3][0,1]

# # def RI_xz(t):
# #     return spice.sxform("ITRF93", "J2000", t)[0:3,0:3][0,2]

# # def RI_yx(t):
# #     return spice.sxform("ITRF93", "J2000", t)[0:3,0:3][1,0]

# # def RI_yy(t):
# #     return spice.sxform("ITRF93", "J2000", t)[0:3,0:3][1,1]

# # def RI_yz(t):
# #     return spice.sxform("ITRF93", "J2000", t)[0:3,0:3][1,2]

# # def RI_zx(t):
# #     return spice.sxform("ITRF93", "J2000", t)[0:3,0:3][2,0]

# # def RI_zy(t):
# #     return spice.sxform("ITRF93", "J2000", t)[0:3,0:3][2,1]

# # def RI_zz(t):
# #     return spice.sxform("ITRF93", "J2000", t)[0:3,0:3][2,2]



# # from sympy import Function, cos, expand_trig, Integer, pi, sin
# # from sympy.core.logic import fuzzy_and, fuzzy_not

# # class cosLon(Function):
# #     @classmethod
# #     def eval(cls, x):
# #         # If x is an explicit integer multiple of pi, return 1 or -1
# #         n = x/sy.pi
# #         if isinstance(n, Integer):
# #             if n % 2 == 0:
# #                 return 1
# #             return -1

# #     # Define numerical evaluation with evalf().
# #     def _eval_evalf(self, prec):
# #         return sy.cos(self.args[0])

# #     def _eval_rewrite(self, rule, args, **hints):
# #         if rule == cos:
# #             return 1 - cos(*args)
# #         elif rule == sin:
# #             return 2*sin(x/2)**2


# #     def _eval_expand_trig(self, **hints):
# #         x = self.args[0]
# #         return expand_trig(sy.cos(x))


# #     def as_real_imag(self, deep=True, **hints):
# #         # reuse _eval_rewrite(cos) defined above
# #         return self.rewrite(cos).as_real_imag(deep=deep, **hints)

# #     # Define differentiation.
# #     def fdiff(self, argindex=1):
# #         return sin(self.args[0])
    
# #     def _lambdacode(self):
# #         return fixedPositionVectorNorm[0] / (sy.cos(lat))


# # import matplotlib
# # import spiceypy as spice
# # from scipy.special import lpmv




# # if __name__ == "__main__":

# #     for kernel in getCriticalKernelsRelativePaths():
# #         spice.furnsh(kernel)
# #         print(kernel)

    
# #     norg = sy.Symbol("n", integer=True, positive=True, whole=True)
# #     morg = sy.Symbol("m", integer=True, positive=True, whole=True)
# #     i = sy.Symbol("i", integer=True, positive=True, whole=True)
# #     j = sy.Symbol("j", integer=True, positive=True, whole=True)
# #     n = sy.Indexed('n', i)
# #     m = sy.Indexed('m', j)

# #     mu = sy.Symbol("mu", real=True, positive=True)

# #     nMaxSy = sy.Symbol('n_max', real=True, positive=True)
# #     mMaxSy = sy.Symbol('m_max', real=True, positive=True)
# #     #nSy = sy.Indexed("n")
# #     #mSy = sy.Indexed("m")
# #     cs = []
# #     ss = []
# #     csValues = []
# #     ssValues = []
# #     for ii in range(0, 500+1):
# #         cs.append([])
# #         ss.append([])
# #         for jj in range(0, 500+1):
# #             cs[ii].append(sy.Symbol(f'c_{ii}_{jj}', real=True))
# #             ss[ii].append(sy.Symbol(f'c_{ii}_{jj}', real=True))
# #             #csValues[i].append(i+j+10)
# #             #ssValues[i].append(-i-j+10)


# #     fileToRead = os.path.normpath(os.path.join(os.path.dirname(__file__), "../testData/JGM2.cof"))
# #     data = nsGravity.gravityField.readFromCoefFile(fileToRead)    
# #     for nv in range(0, 60):
        
# #         csValues.append([])
# #         ssValues.append([])
# #         for mv in range(0, nv):
# #             k = 2
# #             if m == 0:
# #                 k = 1            
# #             csValues[nv].append(data.getC(nv, mv) / (math.sqrt(math.factorial(nv+mv)/(math.factorial(nv-mv) *k* (2*nv+1)))))
# #             ssValues[nv].append(data.getS(nv, mv) / (math.sqrt(math.factorial(nv+mv)/(math.factorial(nv-mv) *k* (2*nv+1)))))
            
# #     cs = sy.Matrix(cs)
# #     ss = sy.Matrix(ss)

# #     def getC(n, m):
# #         return csValues[n][m]

# #     def getS(n, m):
# #         return ssValues[n][m]
    
# #     def betterAssoc(n, m, expr):
# #         # absurdly faster than sympy's function
# #         # the lpmv function returns 0 if m is higher than n
# #         # might need to do different things with the derivative
# #         return lpmv(n, m, expr) # sy.assoc_legendre(n, m, expr)

# #     t = sy.Symbol('t', real=True)
# #     x,y,z,vx,vy,vz = sy.symbols('x,y,z,vx,vy,vz', real=True)
# #     i2fSymbol = rotationMatrixFunction.makeSymbolicMatrix("R", rotationMatrixFunction.matrixNameMode.xyz, [t])
# #     rotHelperInertialToFixed = rotationMatrixFunction("J2000", "ITRF93")
# #     redictDict = {"cFunc": getC, "sFunc": getS, "assoc_legendre": betterAssoc}
# #     modules = [redictDict, "numpy"]
    
# #     subsDict = OrderedDict()
# #     rotHelperInertialToFixed.populateRedirectionDictWithCallbacks(i2fSymbol, redictDict, subsDict)
# #     fixedVec = i2fSymbol * sy.Matrix([[x],[y],[z]])
# #     xf = fixedVec[0]
# #     yf = fixedVec[1]
# #     zf = fixedVec[2]


# #     xyMag2 = fixedVec[0]**2+fixedVec[1]**2
# #     xyMag = sy.sqrt(xyMag2)


# #     rCbSy = sy.Symbol("R_c", real=True, positive=True)
# #     lat = sy.Symbol('lat', real=True)
# #     lon = sy.Symbol('lon', real=True)
# #     rSy = sy.sqrt(x**2+y**2+z**2)

# #     fixedPositionVectorNorm = fixedVec /rSy

# #     subsDict[sy.sin(lat)] = fixedPositionVectorNorm[2]
# #     subsDict[sy.cos(lat)] = sy.sqrt(fixedPositionVectorNorm[0]**2 + fixedPositionVectorNorm[1]**2)
# #     subsDict[sy.tan(lat)] = fixedPositionVectorNorm[2]/sy.sqrt(fixedPositionVectorNorm[0]**2 + fixedPositionVectorNorm[1]**2)

# #     subsDict[sy.cos(lon)] = fixedPositionVectorNorm[0] / (sy.sqrt(fixedPositionVectorNorm[0]**2 + fixedPositionVectorNorm[1]**2))
# #     subsDict[sy.sin(lon)] = fixedPositionVectorNorm[1] / (sy.sqrt(fixedPositionVectorNorm[0]**2 + fixedPositionVectorNorm[1]**2))
# #     subsDict[sy.Abs(sy.cos(lat))]=sy.cos(lat)
# #     #subsDict[lat] = sy.asin(fixedPositionVectorNorm[2])
# #     #subsDict[lon] = sy.atan2(fixedPositionVectorNorm[1] / (sy.sqrt(fixedPositionVectorNorm[0]**2 + fixedPositionVectorNorm[1]**2)), fixedPositionVectorNorm[0] / (sy.sqrt(fixedPositionVectorNorm[0]**2 + fixedPositionVectorNorm[1]**2))).trigsimp(deep=True)
# #     #subsDict[sy.Abs(sy.sin(lat))]=sy.sin(lat)
    


# #     cFunc = sy.Function("cFunc", real=True)
# #     sFunc = sy.Function("sFunc", real=True)
    
# #     #inTheSum = ((rCbSy/rSy)** n)*(n+1)*sy.assoc_legendre(n, m, sy.sin(lat))*(cFunc(n,m)*sy.cos(m*lon) + sFunc(n,m)*sy.sin(m*lon))
# #     #inTheSum = ((rCbSy/rSy)** nSy)*(nSy+1)*sy.assoc_legendre(nSy, mSy, sy.sin(lat))*(cFunc(nSy,mSy)*sy.cos(mSy*lon) + sFunc(nSy,mSy)*sy.sin(mSy*lon))
# #     #termState = [rCbSy, rSy, n, m, lat, lon]
# #     #innerFunc = sy.lambdify(termState, inTheSum, modules=modules)
# #     #display(innerFunc(6374, 12000, 2, 2, 1.0, 2.))           
# #     #inTheSums = sy.Function("inner1", real=True)
    
# #     delUDelR =  (-mu/(rSy**2))*  sy.Sum( ((rCbSy/rSy)** n)*(n+1)*sy.assoc_legendre(n, m, sy.sin(lat))*(cFunc(n,m)*sy.cos(m*lon) + sFunc(n,m)*sy.sin(m*lon)), (i, 2, nMaxSy), (j, 0, mMaxSy) )
# #     delUDelLat =  (mu/rSy)*  sy.Sum( ((rCbSy/rSy)** n)*(sy.assoc_legendre(n, m+1, sy.sin(lat)) -m*sy.tan(lat))*(cFunc(n,m)*sy.cos(m*lon) + sFunc(n,m)*sy.sin(m*lon)), (i, 2, nMaxSy), (j, 0, mMaxSy) )
# #     delUDelLon =  (mu/rSy)* sy.Sum( ((rCbSy/rSy)** n)*m*(sy.assoc_legendre(n, m, sy.sin(lat)))*(sFunc(n,m)*sy.cos(m*lon) - cFunc(n,m)*sy.cos(m*lon)), (i, 2, nMaxSy), (j, 0, mMaxSy) )

# #     display(delUDelR)
# #     display(delUDelLat)
# #     display(delUDelLon)
# #     dPotdSph = sy.Matrix([[delUDelR], [delUDelLat], [delUDelLon]])
# #     dPotdSph = sy.Matrix([[delUDelR], [delUDelLat], [delUDelLon]])

# #     sumState = [t, x, y, z, vx, vy, vz, mu, rCbSy, nMaxSy, mMaxSy]
# #     # cannot CSE this!!
# #     sumCallback = sy.lambdify(sumState, dPotdSph, modules=modules, dummify=True, docstring_limit=None)
    

# #     timeVaryingInertialToFixedMatrix = lambda t : spice.sxform("J2000", "ITRF93", t)[0:3,0:3] #I think gmat is using ITRF93 as their ECEF
# #     i2fSymbol = rotationMatrixFunction.makeSymbolicMatrix("R", rotationMatrixFunction.matrixNameMode.xyz, [t])

# #     accelxb = ((1.0/rSy) * delUDelR  - (zf/((rSy**2) * sy.sqrt(xyMag))) * delUDelLat) * xf - (1.0/xyMag)* delUDelLon*yf
# #     accelyb = ((1.0/rSy) * delUDelR  - (zf/((rSy**2) * sy.sqrt(xyMag))) * delUDelLat) * yf + (1.0/xyMag)* delUDelLon*xf
# #     accelzb = ((1.0/rSy) * delUDelR)*zf + (sy.sqrt(xyMag)/(rSy**2))*delUDelLat 

# #     accelInertialMat = i2fSymbol.transpose()*sy.Matrix([[accelxb], [accelyb], [accelzb]])
# #     accelInertial = [accelInertialMat[0],accelInertialMat[1],accelInertialMat[2]]
# #     display(accelxb)
# #     display(accelyb)
# #     display(accelzb)
# #     print("subbing 1")
# #     accelInertialSub = SafeSubs(accelInertial, subsDict)
# #     display(accelInertialSub[0])
# #     display(accelInertialSub[1])
# #     display(accelInertialSub[2])
# #     # print("subbing 2")
# #     # accelInertialSub = SafeSubs(accelInertialSub, subsDict)

# #     print("lambdifying")
# #     callback = sy.lambdify(sumState, accelInertialSub, modules=redictDict, docstring_limit=None)
# #     print(callback.__doc__)
# #     posVec = [7100000.0, 0.0, 1300000.0,   0.0, 7350.0, 1000.0]
# #     muVal = data._mu
# #     rCbVal = 6378136.3
# #     print(callback(0, *posVec, muVal, rCbVal, 4, 4))









# #     # display(delUDelR)
# #     # print("start")
# #     # display(sumCallback(3.144e5, 6374, 12000, 1.0, 2., 50, 50))           
# #     # print("stop")
# #     # for kernel in getCriticalKernelsRelativePaths():
# #     #     spice.furnsh(kernel)
# #     #     print(kernel)

# #     # # Define the frames and the time (UTC or TDB)
# #     # time = 0.0# spice.str2et("2001-01-01T12:00:00.000:TDB")  # Example time
# #     # to_frame = "ITRF93"  # Earth-fixed frame
# #     # from_frame = "J2000"        # Inertial frame (e.g., J2000)

# #     # # Get the rotation matrix from Earth-fixed to inertial frame
# #     # rotation_matrix = spice.pxform(from_frame, to_frame, time)
# #     # inertialPos = [7100.0, 0.0, 1300.0]
# #     # fixed = rotation_matrix@inertialPos
# #     # expected = [1256.53281141092, 6987.960367867953, 1299.821214996343]
# #     # print(fixed)
# #     # diff = [fixed[0]-expected[0], fixed[1]-expected[1], fixed[2]-expected[2]]
# #     # print("diff ")
# #     # print(diff)

# #     # t = sy.Symbol('t', real=True)
# #     # x,y,z = sy.symbols('x,y,z', real=True)
# #     # xf,yf,zf = sy.symbols('x_f,y_f,z_f', real=True)
# #     # i2fSymbol = rotationMatrixFunction.makeSymbolicMatrix("R", rotationMatrixFunction.matrixNameMode.xyz, [t])
# #     # expr = i2fSymbol * sy.Matrix([[x],[y], [z]])
# #     # subsDict = {}
# #     # helper = OdeLambdifyHelper(t, [x,y,z], expr, [], subsDict )
    
    
# #     # rotHelperInertialToFixed = rotationMatrixFunction("J2000", "ITRF93")
# #     # rotHelperInertialToFixed.populateRedirectionDictWithCallbacks(i2fSymbol, helper.FunctionRedirectionDictionary, subsDict)
# #     # odeCallback = helper.CreateSimpleCallbackForSolveIvp()

# #     # fixedEvaluated = odeCallback(0.0, inertialPos)
# #     # print(fixedEvaluated)

# #     # fromSpice = spice.mxv(rotation_matrix, inertialPos)
# #     # print(fromSpice)


# #     # print("done")
# #     # from sympy.vector import CoordSys3D, Del
# #     # fixed = CoordSys3D("R")
# #     # r = sy.Symbol('r', real=True, positive=True)
# #     # lat = sy.Symbol(r'\gamma', real=True)
# #     # lon = sy.Symbol(r'\lambda', real=True)
# #     # pos = r*sy.cos(lat)*sy.cos(lon)*fixed.i + r*sy.cos(lat)*sy.sin(lon)*fixed.j + r*sy.sin(lat)*fixed.k
# #     # pos2 = x*fixed.i + y*fixed.j + z*fixed.k
    
# #     # potX = sy.Symbol(r'\frac{dU}{dr}')
# #     # potY = sy.Symbol(r'\frac{dU}{dg}')
# #     # potZ = sy.Symbol(r'\frac{dU}{dl}')

# #     # rMagExpr = pos2.magnitude()
# #     # latExpr = sy.asin(z/rMagExpr)
# #     # lonExpr = sy.atan2(y, x)

# #     # vecDiffX = (rMagExpr).diff(x)*potX+(latExpr).diff(x).simplify()*potY+(lonExpr).diff(x)*potZ
# #     # display(vecDiffX.subs(rMagExpr, r))

# #     # vecDiffY = (rMagExpr).diff(y)*potX+(latExpr).diff(y).simplify()*potY+(lonExpr).diff(y)*potZ
# #     # display(vecDiffY.subs(rMagExpr, r))

# #     # vecDiffZ = (rMagExpr).diff(z)*potX+(latExpr).diff(z).simplify()*potY+(lonExpr).diff(z)*potZ
# #     # display(vecDiffZ.subs(rMagExpr, r))

# #     # drdr = pos.diff(r)
# #     # dlatdr = pos.diff(lat)
# #     # dlondr = pos.diff(lon)

# #     # display(drdr)
# #     # display(dlatdr)
# #     # display(dlondr)
# #     # display(drdr+dlatdr+dlondr)

# #     # display(Del().gradient(pos))
# #     # display(Del().gradient(pos).doit())



# #     # latexpr = (1/(sy.sqrt(x*x+y*y)))*(sy.Matrix([0,0,1]) - sy.Matrix([x,y,z])*z/(r*r))
# #     # display(latexpr[2].simplify())

    



# #     #fixed_norm = math.sqrt(fixed[0]**2+fixed[1]**2+fixed[2]**2)
# #     #print(str(180*math.asin(1300000/fixed_norm)/math.pi))
# #     #print(str(180*math.asin(fixed[2]/fixed_norm)/math.pi))
# #     #rotation_matrix = spice.tipbod(to_frame, 399, 0.0)
    
# #     # # rotation_matrix is a 3x3 matrix
# #     # print(rotation_matrix)
# #     # recGeo = spice.recgeo(fixed, 6378.1363, 0.0033527)
# #     # print(recGeo)
# #     # print(str(180*recGeo[1]/math.pi))

# #     # recLat = spice.reclat(fixed)
# #     # print(recLat)
# #     # print(str(180*recLat[2]/math.pi))

# #     # minAngle = 0
# #     # maxAngle = 2*math.pi
# #     # rad = 7000000.0
# #     # numSteps = 100
# #     # step = (maxAngle-minAngle)/numSteps
# #     # angles = []
# #     # lats = []
    

# #     # radii = spice.bodvrd("EARTH", "RADII", 3)[1]
# #     # re = radii[0]
# #     # rp = radii[2]
# #     # f = (re-rp)/re
# #     # geodeticLat = math.asin(spice.recgeo(fixed, re, f)[1])*180.0/math.pi
# #     # print(str(geodeticLat))

# #     # for i in range(0, 100):
# #     #     angle = i*step
# #     #     x = rad*math.cos(angle)
# #     #     y = rad*math.sin(angle)
    
# #     #     fixed = rotation_matrix@[x, y, 0.0]
        
# #     #     lat = 180*math.asin(fixed[2]/fixed_norm)/math.pi
    
# #     #     angles.append(angle*180.0/math.pi)
# #     #     lats.append(lat)

# #     # plt.plot(angles, lats)  
        
# #     # # naming the x axis  
# #     # plt.xlabel('inertial angle')  
# #     # # naming the y axis  
# #     # plt.ylabel('latitude')  
        
# #     # # giving a title to my graph  
# #     # plt.title('Around')  
        
# #     # # function to show the plot  
# #     # plt.show()  
# #     # spice.kclear()

# #     # print(spice.spkezr( "Earth", 0, "IAU_Earth", 'NONE', "301" ))
# #     #testValidation()
# #     ##pass
    
# # #%%

# # def R_xx(t):
# #     return spice.sxform("J2000", "ITRF93", t)[0:3,0:3][0,0]

# # def R_xy(t):
# #     return spice.sxform("J2000", "ITRF93", t)[0:3,0:3][0,1]

# # def R_xz(t):
# #     return spice.sxform("J2000", "ITRF93", t)[0:3,0:3][0,2]

# # def R_yx(t):
# #     return spice.sxform("J2000", "ITRF93", t)[0:3,0:3][1,0]

# # def R_yy(t):
# #     return spice.sxform("J2000", "ITRF93", t)[0:3,0:3][1,1]

# # def R_yz(t):
# #     return spice.sxform("J2000", "ITRF93", t)[0:3,0:3][1,2]

# # def R_zx(t):
# #     return spice.sxform("J2000", "ITRF93", t)[0:3,0:3][2,0]

# # def R_zy(t):
# #     return spice.sxform("J2000", "ITRF93", t)[0:3,0:3][2,1]

# # def R_zz(t):
# #     return spice.sxform("J2000", "ITRF93", t)[0:3,0:3][2,2]




# # def RI_xx(t):
# #     return spice.sxform("ITRF93", "J2000", t)[0:3,0:3][0,0]

# # def RI_xy(t):
# #     return spice.sxform("ITRF93", "J2000", t)[0:3,0:3][0,1]

# # def RI_xz(t):
# #     return spice.sxform("ITRF93", "J2000", t)[0:3,0:3][0,2]

# # def RI_yx(t):
# #     return spice.sxform("ITRF93", "J2000", t)[0:3,0:3][1,0]

# # def RI_yy(t):
# #     return spice.sxform("ITRF93", "J2000", t)[0:3,0:3][1,1]

# # def RI_yz(t):
# #     return spice.sxform("ITRF93", "J2000", t)[0:3,0:3][1,2]

# # def RI_zx(t):
# #     return spice.sxform("ITRF93", "J2000", t)[0:3,0:3][2,0]

# # def RI_zy(t):
# #     return spice.sxform("ITRF93", "J2000", t)[0:3,0:3][2,1]

# # def RI_zz(t):
# #     return spice.sxform("ITRF93", "J2000", t)[0:3,0:3][2,2]


# # def odefunc2(t, _Dummy_41):
# #     [x, y, z, vx, vy, vz] = _Dummy_41
# #     x0 = 398600441500000.0*(z**2 + y**2 + x**2)**(-1.5)
# #     x1 = math.sqrt(5)
# #     z_f = z*R_zz(t) + y*R_zy(t) + x*R_zx(t)
# #     x3 = z_f**2
# #     x_f = z*R_xz(t) + y*R_xy(t) + x*R_xx(t)
# #     x5 = x_f**2
# #     y_f = z*R_yz(t) + y*R_yy(t) + x*R_yx(t)
# #     x7 = y_f**2
# #     x8 = x5 + x7
# #     x9 = x3 + x8
# #     x10 = x9**(-2.0)
# #     x11 = x9**(-1.0)
# #     x12 = x11*x3
# #     x13 = x9**(-3.0)
# #     x14 = math.sqrt(7)
# #     x15 = x9**(-5/2)
# #     x16 = 1/math.sqrt(x9)
# #     x17 = x16*z_f
# #     x18 = x9**(-3/2)
# #     x19 = x18*z_f**3
# #     x20 = 1 - x12
# #     x21 = math.sqrt(x20)
# #     x22 = x11*x5
# #     x23 = x11*x7 + x22
# #     x24 = math.sqrt(x23)
# #     x25 = x24**(-1.0)
# #     x26 = x16*x25
# #     x27 = x26*x_f
# #     x28 = math.sqrt(15)
# #     x29 = 6.23292133333333e-11*x28
# #     x30 = 3.984267e-10*x16*x25*x28*y_f - x27*x29
# #     x31 = 4.86459424599602e+28*x10
# #     x32 = 3 - 3*x12
# #     x33 = x23**(-1.0)
# #     x34 = 2*x33
# #     x35 = -x22*x34 + 1
# #     x36 = -x35
# #     x37 = x28*x36
# #     x38 = x11*x_f*y_f
# #     x39 = x33*x38
# #     x40 = x28*x39
# #     x41 = x32*(4.0651395e-7*x37 - 4.667031e-7*x40)
# #     x42 = math.sqrt(105)
# #     x43 = x36*x42
# #     x44 = x39*x42
# #     x45 = 3.0146954e-8*x43 - 4.12820393333333e-8*x44
# #     x46 = (15/2)*x12 - 3/2
# #     x47 = math.sqrt(42)
# #     x48 = 3.38066616666667e-7*x47
# #     x49 = x26*y_f
# #     x50 = 4.14677733333333e-8*x47
# #     x51 = x27*x48 + x49*x50
# #     x52 = x1*x36
# #     x53 = x1*x39
# #     x54 = 3.5034932e-8*x52 + 1.3257378e-7*x53
# #     x55 = (105/2)*x12 - 15/2
# #     x56 = -15/2*x17 + (35/2)*x19
# #     x57 = math.sqrt(10)
# #     x58 = x27*x57
# #     x59 = -1.42026798e-7*x49*x57 - 1.60910385e-7*x58
# #     x60 = x20**(3/2)
# #     x61 = math.sqrt(70)
# #     x62 = 4*x18*x5*y_f/x23**(3/2) - x49
# #     x63 = x61*x62
# #     x64 = 2*x27
# #     x65 = -x27 + x36*x64
# #     x66 = x61*x65
# #     x67 = 2.35672816666667e-8*x63 + 1.20192313333333e-8*x66
# #     x68 = 7.07327228571429e-9*x61*x65 - 1.43578507142857e-9*x63
# #     x69 = x20**2
# #     x70 = math.sqrt(35)
# #     x71 = x70*(2*x16*x25*x_f*x62 - x34*x38)
# #     x72 = x70*(x35 + x64*x65)
# #     x73 = x13*x69*(1.103019e-9*x71 - 6.73173214285714e-10*x72)
# #     x74 = 2.35526817030442e+25*x1*x10*((3/2)*x12 - 1/2) - 6.20540902903166e+36*x13*z_f*x20*x45 - 3.29824538203455e+42*x13*x20*x54*x55 + 3.29824538203455e+42*x13*x21*x56*x59 - 5.34457573142779e+36*x13*((35/8)*x10*z_f**4 - 15/4*x12 + 3/8) - 3.95955728052958e+29*x14*x15*(-3/2*x17 + (5/2)*x19) + 1.4593782737988e+29*x15*z_f*x21*x30 + 4.13693935268778e+35*x15*x21*x46*x51 + 6.20540902903166e+36*x15*x60*x67 + 3.46315765113628e+44*z_f*x60*x68/x9**(7/2) - x31*x41 - 3.46315765113628e+44*x73
# #     x75 = math.sqrt(x8)
# #     x76 = x10*x24
# #     x77 = x15*x24
# #     x78 = 1.03423483817194e+35*x76
# #     x79 = z_f*x25
# #     x80 = x21**(-1.0)
# #     x81 = 6.5964907640691e+41*x77
# #     x82 = x11*(-2.35526817030442e+25*x1*x10*z_f*x24 + 3.24306283066401e+28*x10*x41*x79 + 9.89889320132394e+28*x14*x46*x76 - 4.65405677177375e+36*x15*x60*x67*x79 + 1.62153141533201e+28*x18*x24*x30*(3*x11*x3*x80 - 3*x21) + x45*x78*(15 - 45*x12) + x51*x78*(x16*z_f*x46*x80 - 15*x17*x21) + x54*x81*(105*x17*x20 - 2*x17*x55) + 1.06891514628556e+36*x56*x77 + x59*x81*(x16*z_f*x56*x80 - x21*x55) + x68*x81*(315*x11*x21*x3 - 105*x60) + 2.77052612090902e+44*x73*x79)
# #     accel_fixed_z = x17*x74 + x75*x82
# #     x84 = (-1.03423483817194e+35*x10*x21*x46*(x27*x50 - x48*x49) - 1.55135225725792e+36*x10*x60*(7.0701845e-8*x61*x65 - 3.6057694e-8*x63) - 6.92631530227256e+43*x13*z_f*x60*(-2.12198168571429e-8*x63 - 4.30735521428571e-9*x66) + 1.55135225725792e+36*x15*z_f*x20*(-4.12820393333333e-8*x43 - 1.20587816e-7*x44) + 6.5964907640691e+41*x15*x20*x55*(1.3257378e-7*x52 - 1.40139728e-7*x53) - 6.5964907640691e+41*x15*x21*x56*(1.60910385e-7*x16*x25*x57*y_f - 1.42026798e-7*x58) + 6.92631530227256e+43*x15*x69*(2.69269285714286e-9*x71 + 4.412076e-9*x72) + 1.62153141533201e+28*x18*x32*(-4.667031e-7*x37 - 1.6260558e-6*x40) - z_f*x21*x31*(3.984267e-10*x27*x28 + x29*x49))/x8
# #     x85 = z_f*x82/x75
# #     accel_fixed_y = x16*y_f*x74 + x_f*x84 - y_f*x85
# #     accel_fixed_x = x16*x_f*x74 - x_f*x85 - y_f*x84
# #     return [vx, 
# #             vy,
# #             vz, 
# #             -x*x0 + accel_fixed_z*RI_xz(t) + accel_fixed_y*RI_xy(t) + accel_fixed_x*RI_xx(t), 
# #             -y*x0 + accel_fixed_z*RI_yz(t) + accel_fixed_y*RI_yy(t) + accel_fixed_x*RI_yx(t), 
# #             -z*x0 + accel_fixed_z*RI_zz(t) + accel_fixed_y*RI_zy(t) + accel_fixed_x*RI_zx(t)]




# # for kernel in getCriticalKernelsRelativePaths():
# #     spice.furnsh(kernel)
# #     print(kernel)
# # odefunc2(0.0,  [7100000.0, 0.0, 1300000.0,   0.0, 7350.0, 1000.0])

# # #%%
# # import builtins
# # def getC(n, m):
# #     return 2#csValues[n][m]

# # def getS(n, m):
# #     return 2#ssValues[n][m]
# # from math import cos, sin
# # def _lambdifygenerated(r_c, r, lat, lon, n_m, m_m):
# #     return (builtins.sum((r_c/r)**n[-1]*(cFunc(n[-1], m[-1])*cos(lon*m[-1]) + sFunc(n[-1], m[-1])*sin(lon*m[-1]))*(n[-1] + 1)*sy.assoc_legendre(n[-1], m[-1], sin(lat)) for m in range(-1, 0+1) for n in range(n_m, 2+1)))

# # print(_lambdifygenerated(6000, 12000, 1.2, 2.0, 4, 4))



# #%%
# import sympy as sy
# from IPython.display import display

# # normal symbols
# x = sy.Symbol('x', real=True)
# e = sy.Symbol('e', real=True, positive=True)

# # indexed symbols
# i = sy.Symbol('i', integer=True)
# iMax = sy.Symbol("i_m", integer=True, positive=True)
# xOfI = sy.Indexed('x',i)

# # the expression
# s = sy.Sum((xOfI+e*sy.sin(xOfI)+(e*sy.sin(xOfI))**2),(i,1,iMax))
# display(s)

# # lambdify normally, it works
# callback = sy.lambdify([x, e, iMax], s)
# answer = callback([5, 6, 7, 8, 9], 0.01, 4)
# print(answer)

# # lambdify with cse=True, and it fails
# callbackWithCse = sy.lambdify([x, e, iMax], s, cse=True)
# answer2 = callbackWithCse([5, 6, 7, 8, 9], 0.01, 4)
# print(answer2)