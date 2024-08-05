import pytest

def assertAlmostEquals(val1, val2, places, msg = ""):
    delta= 1.0*(10**-1*places)
    assert pytest.approx(val1, delta) == val2, msg