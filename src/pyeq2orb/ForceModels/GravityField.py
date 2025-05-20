from typing import List, Optional, Tuple
import sympy as sy
from sympy.vector import CoordSys3D, Del

class gravityField:
    @staticmethod
    def readFromCoefFile(coefFile : str, desiredOrder : int = -1, desiredDegree  :int = -1)->"gravityField":
        cData = []
        cData.append([0])
        cData.append([0,0])

        sData = []
        sData.append([0])
        sData.append([0,0])

        with open(coefFile, 'r') as f:
            lastDegree = "1"
            for line in f:
                if line.startswith("COMMENT") or "#" in line:
                    continue
                if line.startswith("POTFIELD"):
                    mu = float(line[18:39]) # there is other data in this line, not sure what it is
                if line.startswith("RECOEF"):
                    # trusting that the coefficients are in order
                    if line[8:11].strip() != lastDegree:
                        lastDegree = line[8:11].strip()
                        cDataForThisDegree = []
                        sDataForThisDegree = []
                        cData.append(cDataForThisDegree)
                        sData.append(sDataForThisDegree)
                    sVal = 0.0
                    cVal = float(line[17:38].strip())
                    if line[11:14].strip() != '0':
                        sVal = float(line[38:-1].strip())

                    cDataForThisDegree.append(cVal)
                    sDataForThisDegree.append(sVal)
        return gravityField(cData, sData, mu)

    def __init__(self, cData : List[List[float]], sData:List[List[float]], mu:float):
        self._mu = mu
        self._cData = cData
        self._sData = sData

    def getC(self, n, m)->float:
        return self._cData[n][m]
    
    def getS(self, n, m)->float:
        return self._sData[n][m]

    def maxDegree(self)->int:
        return len(self._data)
    
    def maxOrder(self)->int:
        return len(self._data[-1])
    

def makeConstantForSphericalHarmonicCoefficient(name, n, m):
    #if not name in self.coefficientCache:
    #    self.coefficientCache[name] = []
    coef = sy.Symbol(name + "{^{" + str(m) + "}_" + str(n) + "}", real=True)#, commutative=False)
    #self.coefficientCache[name].append(coef)
    return coef

def legendre_functions_goddard_single(n, m, expr:sy.Expr):
    #coef = sy.Function("P" + "{^{" + str(m) + "}_" + str(n) + "}", real=True)(list(expr.free_symbols)[0])#, commutative=False)
    #return coef
    if m > n:
        return 1#TODO: is this ok?
    pVal = sy.assoc_legendre(n, m, expr)
    return pVal

def _cosMLon(m, cosLon):
    #return sy.cos(m*sy.Symbol(r'\lambda', real=True))
    if m == 0:
        return 1
    if m == 1:
        return cosLon
    return 2*cosLon*_cosMLon(m-1, cosLon) - _cosMLon(m-2, cosLon)

def _sinMLon(m, cosLon, sinLon):
    #return sy.sin(m*sy.Symbol(r'\lambda', real=True))
    if m == 0:
        return 0
    if m == 1:
        return sinLon
    return 2*cosLon*_sinMLon(m-1, cosLon, sinLon) - _sinMLon(m-2, cosLon, sinLon)

def _mTanLat(m, lat):
    return (m-1)*sy.tan(lat) + sy.tan(lat) # this does seem like a kind of silly recursion thing...




def makePotential(n_max: int, m_max :int, muSy : sy.Symbol, rSy : sy.Expr, rCbSy :sy.Symbol, lat : sy.Symbol, lon :sy.Expr):
    overallTerms = []
    rCbDivR = rCbSy/rSy
    rCbDivRToN = rCbDivR
    sinLon = sy.sin(lon)
    cosLon = sy.cos(lon)
    sinLat = sy.sin(lat)
    
    for n in range(2, n_max+1):
        mTerms = []
        rCbDivRToN=rCbDivRToN*rCbDivR
        pN0 = legendre_functions_goddard_single(n, 0, sinLat)
        c = makeConstantForSphericalHarmonicCoefficient("C", n, 0)
        firstTerm = c *rCbDivRToN * pN0
        for m in range(1, n+1): # M may start at 1, but it terms in Pcache start at 0
            pNM = legendre_functions_goddard_single(n, m, sinLat)
            sNM = makeConstantForSphericalHarmonicCoefficient("S", n, m)
            cNM = makeConstantForSphericalHarmonicCoefficient("C", n, m)
            innerTerm = rCbDivRToN * pNM * (sNM* _sinMLon(m, cosLon, sinLon)+cNM*_cosMLon(m, cosLon))
            mTerms.append(innerTerm)
        mTerms.reverse()
        totalTerm = firstTerm + sum(mTerms)
        overallTerms.append(totalTerm)
    overallTerms.reverse()
    thePotential = muSy/rSy * sum(overallTerms)

    return thePotential    

def makeDerivativeOfAccelerationTerms(n_max: int, m_max :int, mu : sy.Symbol, rSy : sy.Expr, rCbSy :sy.Symbol, lat : sy.Symbol, lon :sy.Expr) -> Tuple[List[List[sy.Expr]], List[List[sy.Expr]],List[List[sy.Expr]]]:
    legrande_dummy_variable = sy.Symbol('BAD', real=True)
    delPotDelR = []
    delPotDelLat = []
    delPotDelLon = []
    sinLat = sy.sin(lat)
    cosLat = sy.cos(lat)
    sinLon = sy.sin(lon)
    cosLon = sy.cos(lon)

    xLeadingTerm = -mu/(rSy**2)
    yLeadingTerm = mu/(rSy)
    zLeadingTerm = mu/(rSy)

    for n in range(2, n_max+1):
        
        
        rTermsInner = []
        latTermsInner = []
        lonTermsInner = []
        
        xMult = ((rCbSy/rSy)**n)*((n+1))
        yMult = ((rCbSy/rSy)**n)
        zMult = ((rCbSy/rSy)**n)
        for m in range(0, n+1): # M may start at 1, but it terms in Pcache start at 0            
            
            pNM = legendre_functions_goddard_single(n, m, sinLat)
            # both sources say that we need P_n^{m+1}, but doing that has problems with n=m. Digging deeper, this is from the derivative of the potential WRT latitude, and it looks like the PN_mP1 - m*tan(m*lat)*PNM term is just d_PNM/d_lat simplified

            # WHEN I WRITE THE PAPER!!!!
            # There was a challange where the equations in Vallado and the GTDS spec have a problem.  Del_psi/del_lat has a term that looks like:
            # P_n^m+1 - m*tan(lat)*P_n^m
            # BUT, the summation has m to go n, so that first term with m+1 can't be evaluated
            # This lead me to learn more about the Associated legrande function/polynominal, and what is going on is
            # The potential has a term P_n^m(sin(lat))
            # We want the derivative of the potential WRT latitude
            # So del_psi/del_lat = del_psi/del_x * del_x/del_lat
            # del_x/del_lat = del (sin(lat))/del_lat =cos(lat)
            # and if you simplify the recursive expression for the derivative of del P_n^m/del_x on https://en.wikipedia.org/wiki/Associated_Legendre_polynomials
            # that has P_n^m+1 in it and simplify, it will come out to P_n^m+1 - m*tan(lat)*P_n^m 
            # SO... cool brah, but we are still stuck with not being able to evaluate P_n^m+1. But, we got sympy!  So if we need to evaluate the derivative of
            # del_psi/del_lat, we can use sympy to evaluate del_psi/del_lat with del_psi/del_x * del_x/del_lat and it will match
            dpmn_dx = legendre_functions_goddard_single(n, m, legrande_dummy_variable).diff(legrande_dummy_variable)
            dSinLat_dLat = cosLat
            dpmn_dLat = dpmn_dx.subs(legrande_dummy_variable, sinLat)*dSinLat_dLat
            #dpmn_dLat_alt=dpmn_dLat
            if n == m:
                dpmn_dLat=m*(sinLat/cosLat)*pNM
            
            #dpmn_dLat = legendre_functions_goddard_single(n, m+1, sinLat) -m*sy.tan(lat)*pNM
                           
            # scaling
            k = 2
            if m == 0:
                k = 1
            sNM = makeConstantForSphericalHarmonicCoefficient("S", n, m)/ (sy.sqrt(sy.factorial(n+m)/(sy.factorial(n-m) *k* (2*n+1))))
            cNM = makeConstantForSphericalHarmonicCoefficient("C", n, m)/ (sy.sqrt(sy.factorial(n+m)/(sy.factorial(n-m) *k* (2*n+1))))
            
            cosMLon = _cosMLon(m, cosLon)
            sinMLon = _sinMLon(m, cosLon, sinLon)
            
            #mtanLat = _mTanLat(m, lat)

            rTermsInner.append(xLeadingTerm*xMult*(cNM*cosMLon + sNM*sinMLon)*pNM)
            latTermsInner.append(yLeadingTerm*yMult*(cNM*cosMLon + sNM*sinMLon)*(dpmn_dLat))
            lonTermsInner.append(zLeadingTerm*zMult*(m*(sNM*cosMLon-cNM*sinMLon)*(pNM)))
            
        rTermsInner.reverse()
        latTermsInner.reverse()
        lonTermsInner.reverse()
        
        delPotDelR.append(rTermsInner)
        delPotDelLat.append(latTermsInner)
        delPotDelLon.append(lonTermsInner)
        
    delPotDelR.reverse()
    delPotDelLat.reverse()
    delPotDelLon.reverse()

    return [delPotDelR, delPotDelLat, delPotDelLon]
        
def makeAccelerationMatrixFromPotential(n_max: int, m_max :int, mu : sy.Symbol, rSy : sy.Expr, rCbSy :sy.Symbol, lat : sy.Symbol, lon :sy.Expr)->sy.Matrix:
    diffPotentialTerms = makeDerivativeOfAccelerationTerms(n_max, m_max, mu, rSy, rCbSy, lat, lon)


    termsForMatrix = []
    for list in diffPotentialTerms:
        thisExpr = 0
        for innerList in list:
            for term in innerList:
                thisExpr=thisExpr+term
        termsForMatrix.append(thisExpr)
    return sy.Matrix([termsForMatrix]).transpose()


def makeAccelerationMatrixFromPotentialAsSum(nSy: sy.Symbol, mSy :sy.Symbol, nMaxSy: sy.Symbol, mMaxSy :sy.Symbol, mu : sy.Symbol, rSy : sy.Expr, rCbSy :sy.Symbol, lat : sy.Symbol, lon :sy.Expr, cs : list[list[sy.Symbol]], ss : list[list[sy.Symbol]])->sy.Matrix:
    delUDelR = (-mu/(rSy**2)) * sy.Sum( sy.Sum(sy.Pow(rCbSy/rSy), nSy)*(nSy+1)*sy.assoc_legendre(nSy, mSy, sy.sin(lat)*(cs[nSy][mSy]*sy.cos(mSy*lon) + ss[nSy][mSy]*sy.sin(mSy*lon)), (mSy, mMaxSy, 0, -1) ), (nSy, nMaxSy, 0, -1))
    delUDelLat = (-mu/(rSy**2)) * sy.Sum( sy.Sum(sy.Pow(rCbSy/rSy), nSy)*(nSy+1)*sy.assoc_legendre(nSy, mSy, sy.sin(lat)*(cs[nSy][mSy]*sy.cos(mSy*lon) + ss[nSy][mSy]*sy.sin(mSy*lon)), (mSy, mMaxSy, 0, -1) ), (nSy, nMaxSy, 0, -1))
    delUDelLon = (-mu/(rSy**2)) * sy.Sum( sy.Sum(sy.Pow(rCbSy/rSy), nSy)*(nSy+1)*sy.assoc_legendre(nSy, mSy, sy.sin(lat)*(cs[nSy][mSy]*sy.cos(mSy*lon) + ss[nSy][mSy]*sy.sin(mSy*lon)), (mSy, mMaxSy, 0, -1) ), (nSy, nMaxSy, 0, -1))

    return sy.Matrix([[delUDelR], [delUDelLat], [delUDelLon]])


def createSphericalHarmonicGravityAcceleration_org(diffPotentialMatrix : sy.Matrix, r_fixed : sy.Matrix, rSy : sy.Symbol):#, lat : sy.Symbol, lon : sy.Symbol):
    delXbDelRb = sy.Matrix([[1,0,0]])
    delYbDelRb = sy.Matrix([[0,1,0]])
    delZbDelRb = sy.Matrix([[0,0,1]])

    xyMag = r_fixed[0]**2 + r_fixed[1]**2
    rMag = rSy
    delRDelRb = r_fixed.transpose() / rSy
    delLatDelR = (1/sy.sqrt(xyMag))*(delZbDelRb - r_fixed.transpose()*r_fixed[2]/ (rMag**2))
    delLonDelR = (1/xyMag) * (r_fixed[0]*delYbDelRb - r_fixed[1]*delXbDelRb)

    accelxb = diffPotentialMatrix[0]*delRDelRb.transpose()[0,0]
    accelyb = diffPotentialMatrix[1]*delLatDelR.transpose()[0,0]
    accelzb = diffPotentialMatrix[2]*delLonDelR.transpose()[0,0]

    return sy.Matrix([[accelxb],[accelyb], [accelzb]])

def createSphericalHarmonicGravityAcceleration(diffPotentialMatrix : sy.Matrix,  r_fixed : sy.Matrix,  rSy : sy.Symbol, mu  :sy.Symbol):#, lat : sy.Symbol, lon : sy.Symbol):
    # delXbDelRb = sy.Matrix([[1,0,0]])
    # delYbDelRb = sy.Matrix([[0,1,0]])
    # delZbDelRb = sy.Matrix([[0,0,1]])

    delPhiDelR = diffPotentialMatrix[0]
    delPhiDelLat = diffPotentialMatrix[1]
    delPhiDelLon = diffPotentialMatrix[2]

    xb = r_fixed[0]
    yb = r_fixed[1]
    zb = r_fixed[2]

    xyMag2 = (xb**2) + (yb**2)

    # Vallado has an extra term that looks like, but isn't, 2 body gravity at the end?!?
    accelxb = ((1.0/rSy) * delPhiDelR  - (zb/((rSy**2) * sy.sqrt(xyMag2))) * delPhiDelLat) * xb - (1.0/xyMag2)* delPhiDelLon*yb# - mu/(rSy**2)
    accelyb = ((1.0/rSy) * delPhiDelR  - (zb/((rSy**2) * sy.sqrt(xyMag2))) * delPhiDelLat) * yb + (1.0/xyMag2)* delPhiDelLon*xb# - mu/(rSy**2)
    accelzb = ((1.0/rSy) * delPhiDelR)*zb + (sy.sqrt(xyMag2)/(rSy**2))*delPhiDelLat # - mu/(rSy**2)

    accelVector = sy.Matrix([[accelxb], [accelyb], [accelzb]])
    return accelVector
    #return sy.Matrix([[accelxb],[accelyb], [accelzb]])


def makeOverallAccelerationExpression(n_max: int, m_max :int, mu : sy.Symbol, rSy : sy.Expr, rCbSy :sy.Symbol, lat : sy.Symbol, lon :sy.Expr, r_fixed : sy.Matrix)->sy.Matrix:
    accelMatrix = makeAccelerationMatrixFromPotential(n_max, m_max, mu, rSy, rCbSy, lat, lon)

    # potential =makePotential(n_max, m_max, mu, rSy, rCbSy, lat, lon)
    # dPotdr = potential.diff(rSy)
    # dPotdLat = potential.diff(lat)
    # dPotDLon = potential.diff(lon)
    # accelMatrix = sy.Matrix([[dPotdr, dPotdLat, dPotDLon]]).transpose()

    return altConversionToXyz(accelMatrix, r_fixed, rSy, mu)


def makeOverallAccelerationExpressionAsSums(nSy : sy.Symbol, mSy : sy.Symbol, n_max: sy.Symbol, m_max :sy.Symbol, mu : sy.Symbol, rSy : sy.Expr, rCbSy :sy.Symbol, lat : sy.Symbol, lon :sy.Expr, r_fixed : sy.Matrix, c : List[List[sy.Symbol]], s : List[List[sy.Symbol]])->sy.Matrix:
    accelMatrix = makeAccelerationMatrixFromPotentialAsSum(nSy, mSy, n_max, m_max, mu, rSy, rCbSy, lat, lon, c, s)

    # potential =makePotential(n_max, m_max, mu, rSy, rCbSy, lat, lon)
    # dPotdr = potential.diff(rSy)
    # dPotdLat = potential.diff(lat)
    # dPotDLon = potential.diff(lon)
    # accelMatrix = sy.Matrix([[dPotdr, dPotdLat, dPotDLon]]).transpose()

    return altConversionToXyz(accelMatrix, r_fixed, rSy, mu)


def altConversionToXyz(diffPotentialMatrix : sy.Matrix,  r_fixed : sy.Matrix,  rSy : sy.Symbol, mu  :sy.Symbol):#, lat : sy.Symbol, lon : sy.Symbol):
    
    x = r_fixed[0]
    y = r_fixed[1]
    z = r_fixed[2]
    
    fixed = CoordSys3D("R")
    pos2 = x*fixed.i + y*fixed.j + z*fixed.k
    
    potX = diffPotentialMatrix[0]#sy.Symbol(r'\frac{dU}{dr}')
    potY = diffPotentialMatrix[1]#sy.Symbol(r'\frac{dU}{dg}')
    potZ = diffPotentialMatrix[2]#sy.Symbol(r'\frac{dU}{dl}')

    rMagExpr = pos2.magnitude()
    latExpr = sy.asin(z/rMagExpr)
    lonExpr = sy.atan2(y, x)

    vecDiffX = (rMagExpr).diff(x)*potX+(latExpr).diff(x).simplify()*potY+(lonExpr).diff(x)*potZ
    #display(vecDiffX.subs(rMagExpr, r))

    vecDiffY = (rMagExpr).diff(y)*potX+(latExpr).diff(y).simplify()*potY+(lonExpr).diff(y)*potZ
    #display(vecDiffY.subs(rMagExpr, r))

    vecDiffZ = (rMagExpr).diff(z)*potX+(latExpr).diff(z).simplify()*potY+(lonExpr).diff(z)*potZ
    #display(vecDiffZ.subs(rMagExpr, r))    

    accelVector = sy.Matrix([[vecDiffX], [vecDiffY], [vecDiffZ]]) 
    return accelVector