import pytest
import math
import sympy as sy
from pyeq2orb.ForceModels.CircularRestricted3BodyProblem import convertInertialToRotationalCr3bpState, convertRotationalCr3bpStateToInertial, scaleNormalizedStateToUnnormalized, scaleUnNormalizedStateToNormalized


def test_convertInertialToRotationalCr3bpState():
    inertialState = [1.02134, 0, -0.18162, 0, -0.10176+1.02134, 9.76561e-07] 
    expectedRotationalState =[1.02134, 0, -0.18162, 0, -0.10176, 9.76561e-07] 
    actualRototationlState = list(convertInertialToRotationalCr3bpState(0.0, *inertialState))

    for i in range(0, 6):
        pytest.approx(expectedRotationalState[i], actualRototationlState[0], 0.00001)


def test_convertRotationalCr3bpStateToInertial():
    expectedInertial = [1.02134, 0, -0.18162, 0, -0.10176+1.02134, 9.76561e-07] 
    rotationalState =[1.02134, 0, -0.18162, 0, -0.10176, 9.76561e-07] 
    actualInertailState = list(convertRotationalCr3bpStateToInertial(0.0, *rotationalState))

    for i in range(0, 6):
        pytest.approx(expectedInertial[i], actualInertailState[0], 0.00001)


def test_scaleNormalizedStateToUnnormalized():
    normalized = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    L = 3.850e5
    V=1.025
    #T = 2.361e6
    #muR = 0.0125
    muEarth = 398600.4418
    expectedUnnormalized = [1.0*L, 2.0*L, 3.0*L, 0.1*V, 0.2*V, 0.3*V]
    unNormalized = scaleNormalizedStateToUnnormalized(muEarth, L, *normalized)
    for i in range(0, 6):
        pytest.approx(unNormalized[i], expectedUnnormalized[0], 0.00001)

def test_scaleUnNormalizedStateToNormalized():
    expectedNormalized = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    L = 3.850e5
    V=1.025
    #T = 2.361e6
    #muR = 0.0125
    muEarth = 398600.4418
    unnormalized = [1.0*L, 2.0*L, 3.0*L, 0.1*V, 0.2*V, 0.3*V]
    actualUnnormalized = scaleUnNormalizedStateToNormalized(muEarth, L, *unnormalized)
    for i in range(0, 6):
        pytest.approx(actualUnnormalized[i], expectedNormalized[0], 0.00001)        