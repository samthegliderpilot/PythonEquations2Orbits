import pyeq2orb.Utilities.SolutionDictionaryFunctions as DictionaryHelpers #type: ignore
import pytest

def testGettingInitialState() :
    someDictionary =  {}
    someDictionary["x"] = [1,2,3]
    someDictionary['u'] = [4,5,6]

    initialState = DictionaryHelpers.GetInitialStateDictionary(someDictionary)
    assert someDictionary['x'][0]== initialState['x'], "x"
    assert someDictionary['u'][0]== initialState['u'], "u"

def testGettingFinalState() :
    someDictionary =  {}
    someDictionary["x"] = [1,2,3]
    someDictionary['u'] = [4,5,6]

    finalState = DictionaryHelpers.GetFinalStateDictionary(someDictionary)
    assert someDictionary['x'][-1]== finalState['x'], "x"
    assert someDictionary['u'][-1]== finalState['u'], "u"       

def testGettingStateAtIndex() :
    someDictionary =  {}
    someDictionary["x"] = [1,2,3]
    someDictionary['u'] = [4,5,6]

    finalState = DictionaryHelpers.GetValueFromStateDictionaryAtIndex(someDictionary, 1)
    assert someDictionary['x'][1]== finalState['x'], "x"
    assert someDictionary['u'][1]== finalState['u'], "u"   