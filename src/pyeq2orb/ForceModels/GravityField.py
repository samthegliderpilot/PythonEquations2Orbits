from typing import List

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
                    mu = float(line.split(' ')[2]) # there is other data in this line, not sure what it is
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