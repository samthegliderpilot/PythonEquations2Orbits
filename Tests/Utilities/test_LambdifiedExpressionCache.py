import unittest
from pyeq2orb.Utilities.LambdifiedExpressionCache import CacheKey, ExpressionCache #type: ignore
from os import path
from os import remove
import sympy as sy
from sympy.abc import x, y, z

def testBasicOperations() :

    timesCalled = 0
    expectedCreatedObject = "I have been called!"
    def ObjectCreatorCallback() :
        nonlocal timesCalled
        timesCalled = timesCalled+1
        return expectedCreatedObject
    
    key = CacheKey("caller", 0)
    filePath = "TestBasicCacheOperation.pickle"
    if path.isfile(filePath):
        remove(filePath)
    myCache = ExpressionCache(filePath)
    with myCache :
        actualOBject = myCache.GetOrCreateAndCacheObject(key, ObjectCreatorCallback)
        assert actualOBject== expectedCreatedObject
        assert 1== timesCalled

        secondTime = myCache.GetOrCreateAndCacheObject(key, ObjectCreatorCallback)
        assert secondTime== expectedCreatedObject
        assert 1== timesCalled

def testBasicOperationWithSympy():
    timesCalled = 0

    def ObjectCreatorCallback() :
        nonlocal timesCalled
        timesCalled = timesCalled+1
        f = sy.lambdify([x, y], x + y)
        return f
    
    key = CacheKey("caller", x+y)
    filePath = "TestBasicCacheOperation2.pickle"
    if path.isfile(filePath):
        remove(filePath)
    with ExpressionCache(filePath) as myCache :
        actualObject = myCache.GetOrCreateAndCacheObject(key, ObjectCreatorCallback)
        assert actualObject(5,6)== 11
        assert 1== timesCalled

        secondTime = myCache.GetOrCreateAndCacheObject(key, ObjectCreatorCallback)
        assert secondTime(5,6)== 11
        assert 1== timesCalled
    
    with ExpressionCache(filePath) as myCache2 :
        actualObject = myCache2.GetOrCreateAndCacheObject(key, ObjectCreatorCallback)
        assert actualObject(5,6)== 11
        assert 1== timesCalled

def testInvalidatingOnSecondGet():
    timesCalled = 0

    def ObjectCreatorCallback() :
        nonlocal timesCalled
        timesCalled = timesCalled+1
        f = sy.lambdify([x, y], x + y)
        return f
    
    def ObjectCreatorCallback2() :
        nonlocal timesCalled
        timesCalled = timesCalled+1
        f = sy.lambdify([x, y], 2*x + y)
        return f        

    key = CacheKey("caller", x+y)
    filePath = "TestBasicCacheOperation2.pickle"
    if path.isfile(filePath):
        remove(filePath)
    with ExpressionCache(filePath) as myCache :
        actualObject = myCache.GetOrCreateAndCacheObject(key, ObjectCreatorCallback)
        assert actualObject(5,6)== 11
    
    key2 = CacheKey("caller", 2*x+y)        
    with ExpressionCache(filePath) as myCache2 :
        actualObject2 = myCache2.GetOrCreateAndCacheObject(key2, ObjectCreatorCallback2)
        assert actualObject2(5,6)== 16
        assert 2== timesCalled  
        assert myCache2.GetObject(key) is None