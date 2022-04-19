import sympy as sy

class ScalingHelper :
    # not sure how much of this I will need, and what form it should take, but bringing it over just in case
    # def __init__(self, x : list, xDot : list, ts : sy.Symbol, x0BcVals : dict, xfBcVals : dict, termVec : Vector, scaleVector : dict, subsDict : dict) :
    #     scale = False
    #     for i in scaleVector :
    #         if(i != 1) :
    #             scale = True
    #             break
    #     originalX = x
    #     if(scale) :            
    #         # scale the problem and recreate our vectors
    #         x, xDot = ScalingHelper.scaleStates(x, xDot, ts, scaleVector)
    #         stateVarScaleMap = {}
    #         for i in range(0, len(x)) :
    #             stateVarScaleMap[originalX[i]] = x[i]
    #         numScaleVector = {}

    #         for i in range(0, len(scaleVector)) :
    #             numScaleVector[originalX[i]] = (scaleVector[originalX[i]])
    #             if(hasattr(numScaleVector[originalX[i]], "subs")) :
    #                 numScaleVector[originalX[i]] = numScaleVector[originalX[i]].subs(subsDict)
    #         x0BcVals = ScalingHelper.simpleScale(x0BcVals, numScaleVector, stateVarScaleMap)
    #         xfBcVals = ScalingHelper.simpleScale(xfBcVals, numScaleVector, stateVarScaleMap)
                
    #         newTermVec = Vector.zeros(len(termVec))
    #         for i in range(0, len(termVec)):
    #             termExp = termVec[i]
    #             newTermVec[i] = ScalingHelper.substituteScaleVector(termExp, originalX, numScaleVector, x)
    #         termVec = newTermVec
    #     else :
    #         numScaleVector = scaleVector
    #     self.X = x
    #     self.XDot = xDot
    #     self.NumericalScaleVector = numScaleVector
    #     self.X0BoundaryConditions = x0BcVals
    #     self.XFinalBoundaryConditions = xfBcVals
    #     self.TerminalVector = termVec
    #     self.OriginalX = originalX

    # @staticmethod
    # def scaleStates(x, xDot, ts, scaleVectorMap) :
    #     scale = False
    #     for i in scaleVectorMap :
    #         scale = scaleVectorMap[i] != 1.0
    #         if(scale) :
    #             break
    #     if(not scale):
    #         return [x, xDot]        
    #     xBar = []
    #     xDotScaled = []
    #     if scale :        
    #         for i in range(0, len(x)) :                
    #             #TODO: Transfer other assumptions?                
    #             xBar.append(sy.Function(r'\bar{' + str(replaceFuncWithArglessSymbol(x[i]))+ "}", real=x[i].is_real, positive=x[i].is_positive)(ts))
    #         subsToScaled = {}
    #         for i in range(0, len(x)) :
    #             subsToScaled[x[i]] = xBar[i]*scaleVectorMap[x[i]]

    #         for i in range(0, len(xDot)) :
    #             xDotScaled.append(xDot[i].subs(subsToScaled)/scaleVectorMap[x[i]])       
    #     return [Vector.fromArray(xBar), Vector.fromArray(xDotScaled)]

    # @staticmethod
    # def simpleDescale(vec, scale) :
    #     ans = vec.copy()
    #     for i in range(0, len(scale)) :
    #         ans[i] = ans[i]*scale[i]
    #     return ans

    # @staticmethod 
    # def simpleScale(vecMap, scale, newStateSymMap) :
    #     ans = {}
    #     for key in vecMap :
    #         ans[newStateSymMap[key]] = (vecMap[key]/scale[key])
    #         if(hasattr(ans[newStateSymMap[key]], "subs")) :
    #             ans[newStateSymMap[key]] = ans[newStateSymMap[key]].subs(newStateSymMap)
    #     return ans

    # @staticmethod
    # def substituteScaleVector(expres, originalState, scaleVector, scaledStateSymbols) :
    #     newExp = expres
    #     for i in range(0, len(originalState)) :
    #         newExp = newExp.subs(originalState[i],  scaledStateSymbols[i]*scaleVector[originalState[i]])
    #     return newExp

    @staticmethod
    def simpleScale(expression, symbolToReplace, replacementExpression) :
        return expression.subs(symbolToReplace, replacementExpression)

    @staticmethod
    def scaleExpressionsByFinalTime(expressions, t : sy.Symbol, tf: sy.Symbol, tau : sy.Symbol) :
        """Takes the sympy expression (or expressions) and scales them by time using the tf and tau values.  
        The intend is that tf and tau are sympy Symbols, but if you can make the duck typing work, go for it

        Args:
            expression: the expressions to scale by time
            t (sy.Symbol): The time symbol to replace
            tf (sy.Symbol): The final time symbol
            tau (sy.Symbol): The normalized fraction of completed time Symbol

        Returns:
            Updated expressions identical to the ones passed in, but scaled by time.
        """
        # dt/dTau is just tf, so we multiply every equation by tf (per chain rule), and we replace every t with tau
        if(hasattr(expressions, "__len__")) :
            ans = []
            for equ in expressions :
                ans.append(tf*equ.subs(t, tau))
        else :
            ans = tf*expressions.subs(t, tau)
        return ans