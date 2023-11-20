#%%
import sympy as sy
import math
from IPython.display import display, Latex
from collections import OrderedDict
sy.init_printing()
import scipyPaperPrinter as jh #type: ignore
import numpy as np
sy.init_printing(use_unicode=True, wrap_line=False)

jh.printMarkdown("# Converting between celestial (inertial) and terrestial (CB fixed) coordinates")
jh.printMarkdown("Converting between these two reference frames is a fundamental part of astrodynamcis.  If you ever care about the ground track, latitude and longitude, or GEO satellites, this conversion is always being performed.  Having it done in an analytical and symbolic fashion is useful in many cases.  You can't do higher-order non-spherical earth calculations without it.")
jh.printMarkdown("This is going to mostly be following Vallado's chapter 3 analysis.  Ultimitally, this paper will impliment the IAU-2006/2000 conversion.")

r_gcrf = sy.MatrixSymbol('r_{GCRF}', 3, 1,)
r_itrf = sy.MatrixSymbol('r_{ITRF}', 3, 1)

p = sy.MatrixSymbol('P(t)', 3, 3)
n = sy.MatrixSymbol('N(t)', 3, 3)
r = sy.MatrixSymbol('R(t)', 3, 3)
w = sy.MatrixSymbol('W(t)', 3, 3)

jh.printMarkdown("Fundamentally, to get the fixed (GCRF) vector from an inertial vector (ITRF), the rotation is:")
jh.showEquation(r_gcrf, p*n*r*w*r_itrf)

jh.printMarkdown("Where P and N are the precession and nutation matrices at a given date.  R is the sideral-rotation matrix at a date, and W is the polar-motion matrix, again at a specific date.  These are known as the reduction formulas.")
jh.printMarkdown("Vallado notes that it is very hard to find a truly inertial frame.  Most frames are only quasi-inertial, or as I've told a few people, inertial enough.")
jh.printMarkdown("Note that for many operations, the level of precesion of accounting for EOP data and going through the entire IAU reduction may be more precise than we care to be.  I will attempt to design my code to make it easy to insert simpler matrices.")

jh.printMarkdown("### Time")
jh.printMarkdown("For simplicity, unless a specific calculation needs a different time standard, all times in my library will be TT.")
jh.printMarkdown("For the conversions: going from a UTC date:")
jh.printMarkdown("$$TT = UTC+\Delta{AT}+32.184 seconds$$")
jh.printMarkdown("Where $\Delta{AT}$ is the number of leap seconds, and is included in EOP data.")
jh.printMarkdown("Technically, many of the calculations require Barycentric dynamic time, TDB.  However, we are going to adopt the suggestion of his sources that TDB and TT are close enough.")
jh.printMarkdown("We require Julian centuries from some epoch in the TT time standard, $T_{TT}$.  For this, the calculation is:")
t_tt = sy.Symbol('T_{TT}', real=True, positive=True)
jDate_tt = sy.Symbol("JD_{TT}", real=True, positive=True)
t_ttEq = sy.Eq(t_tt, (jDate_tt-245545.0)/(36525))
jh.showEquation(t_ttEq)
jh.printMarkdown("With sympy's default simplification....")
jh.printMarkdown("### Fundamental Arguments")



jh.printMarkdown(r'Many concepts in the next few parts will apply to many of the models leading up to the 2006/2000 model, however specific values may vary from model to model.  We will need to evaluate the mean anomalies of the Moon and Sun, $M_{moon}$, and $M_{\odot}$. These are evaluated as polynomials with the Julian century T_${TT}$ as follows:')
# note that markdown doesn't have the full LaTeX symbols available to us
def makePoly(x, coeffs):
    xTerm = 1
    expr = 0
    for i in range(0, len(coeffs)):
        expr = expr + coeffs[i]*xTerm
        xTerm = xTerm*x
    return expr

lunarMExp = makePoly(t_tt, [485868.249036, 1717915923.2178, 31.8792, 0.051635, -0.00024470 ])
lunarMuExp = makePoly(t_tt, [335779.526232, 1739527262.8478, 12.7512, -0.001037, 0.00000417])
lunarOmegaExp = makePoly(t_tt, [450160.398036, -6962890.5431, 7.4722, 0.007702, -0.00005939])

jh.showEquation("M_{moon}", lunarMExp)
#TODO, the rest

#%%
jh.printMarkdown("The first matrix we will evaluate is the W matrix.  For this, we will need EOP data to get values of $x_p$ and $y_p$.")
s = sy.Symbol('s^{\'}')
xp = sy.Symbol('x_p')
yp = sy.Symbol('y_p')
from pyeq2orb.Coordinates.RotationMatrix import RotAboutXValladoConvention, RotAboutYValladoConvention, RotAboutZValladoConvention
wMat = RotAboutZValladoConvention(-s)*RotAboutYValladoConvention(xp)*RotAboutXValladoConvention(yp)
jh.showEquation("W", wMat)
jh.printMarkdown("We need to evaluate the Chandlre wobble, $a_c$ and annual wobble, $a_s$.  Their average values are:")
ac = sy.Symbol('a_c')
aa = sy.Symbol('a_a')
acVal = 0.26
aaVal = 0.12
jh.showEquation(ac, acVal)
jh.showEquation(aa, aaVal)
jh.printMarkdown("In arcseconds.  Evaluating $s^{\'}$, the high precision expression is:")
sPreciseEq = sy.Eq(s, -0.0015*((ac**2)/1.2+aa**2)*t_tt)
jh.showEquation(sPreciseEq)
jh.printMarkdown("But an approximate expression is:")
sApprx = sy.Eq(s, -0.000047*t_tt)
jh.showEquation(sApprx)
jh.printMarkdown("Note that the constant values in front of each RHS are also in arcseconds.")

def arcSecToRadian(val):
    return val*sy.pi/(180*36000)

def evaluateWPrecise(tt, acArcSec, aaArcSec, xpArcSec, ypArcSec):
    sPrime = -0.0015*((acArcSec**2)/1.2+aaArcSec**2)*tt
    wMat = RotAboutZValladoConvention(-1*arcSecToRadian(sPrime))*RotAboutYValladoConvention(arcSecToRadian(xpArcSec))*RotAboutXValladoConvention(arcSecToRadian(ypArcSec))
    return wMat


#%%
jh.printMarkdown("The Earth Rotation angle is pretty straight forward:")
ear = sy.Symbol(r'\theta_{ERA}', real=True)
jDateUt1 = sy.Symbol('J_{UT1}')
earExp = 2*sy.pi*0.7790572732640+1.00273781191135448*(jDateUt1-2451545.0)
jh.showEquation(ear, earExp)
jh.showEquation(ear, earExp.evalf())
jh.printMarkdown("Notice that this doesn't use TT but UT1.  That conversion must be performed.")
jDateInTermsOfT_TT = sy.solve(t_ttEq, jDate_tt)[0]
earExpLow = earExp.subs(jDateUt1,jDateInTermsOfT_TT )
rMat = RotAboutZValladoConvention(-1*earExp.subs(jDateUt1, jDateInTermsOfT_TT))

#%%
jh.printMarkdown("For low fidelity work (which is likely good enough for most of the work I care to do), we just need simplified expressions for the precession, nutation, Earth rotation, and pole wobble")
jh.printMarkdown("For the wobble, this causes an error in the location of the pole of up to about 30 meters. Looking at the W matrix, it is almost identity.  For moderate precision work, we can ignore it.")


jh.printMarkdown("For the Earth Rotation Angle, we will use the same definition as above, but we will use the TT time instead of UT1 (note that it is not $T_{TT}$, but the raw Julian Date).")
jDateInTermsOfT_TT = sy.solve(t_ttEq, jDate_tt)[0]
earExpLow = earExp.subs(jDateUt1,jDateInTermsOfT_TT )
def lowFidelityInertialToFixedMatrix(currentJulianCenturies):
    # ignore w
    p = None
    n = None
    r = rMat.subs(t_tt, currentJulianCenturies)

def lowFidelityInertialToFixedVelocityMatrix(t_tt):
    pass