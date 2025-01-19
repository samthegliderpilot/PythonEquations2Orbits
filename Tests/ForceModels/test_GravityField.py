import pytest
import math
import sympy as sy
from pyeq2orb.ForceModels.GravityField import gravityField
import os

def testReadingInFile():
    fileToRead = os.path.normpath(os.path.join(os.path.dirname(__file__), "../testData/EGM96.cof"))
    data = gravityField.readFromCoefFile(fileToRead)
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