import pyeq2orb.Utilities.SolutionDictionaryFunctions as DictionaryHelpers
import unittest

class testSolutionDictionaryFunctions(unittest.TestCase) :
    def testGettingInitialState(self) :
        someDictionary =  {}
        someDictionary["x"] = [1,2,3]
        someDictionary['u'] = [4,5,6]

        initialState = DictionaryHelpers.GetInitialStateDictionary(someDictionary)
        self.assertEqual(someDictionary['x'][0], initialState['x'], msg="x")
        self.assertEqual(someDictionary['u'][0], initialState['u'], msg="u")

    def testGettingFinalState(self) :
        someDictionary =  {}
        someDictionary["x"] = [1,2,3]
        someDictionary['u'] = [4,5,6]

        finalState = DictionaryHelpers.GetFinalStateDictionary(someDictionary)
        self.assertEqual(someDictionary['x'][-1], finalState['x'], msg="x")
        self.assertEqual(someDictionary['u'][-1], finalState['u'], msg="u")        

    def testGettingStateAtIndex(self) :
        someDictionary =  {}
        someDictionary["x"] = [1,2,3]
        someDictionary['u'] = [4,5,6]

        finalState = DictionaryHelpers.GetValueFromStateDictionaryAtIndex(someDictionary, 1)
        self.assertEqual(someDictionary['x'][1], finalState['x'], msg="x")
        self.assertEqual(someDictionary['u'][1], finalState['u'], msg="u")            