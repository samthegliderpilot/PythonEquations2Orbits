import pickle
import dill #type: ignore
from typing import Dict, Any, cast
from dataclasses import dataclass
from os.path import isfile

@dataclass(frozen=True)
class CacheKey :
    Id : str 
    EqualityObject: object # often this is the expression you want to lambdify

class ExpressionCache:
    def __init__(self, filePathString : str):
        self._filePath = filePathString
        self._readInDictionary = {} #type: Dict[CacheKey, Any]
        #self._readInFile = self.ReloadFile()

    def __enter__(self):
        self.ReloadFile()
        return self

    def __exit__(self, type, value, traceback):
        self.SaveCache()

    def ReloadFile(self):
        filePathString = self._filePath        
        if not isfile(filePathString):
            return
        with open(filePathString, 'rb') as handle:
            self._readInDictionary = pickle.load(handle)

    def GetOrCreateAndCacheObject(self, key : CacheKey, creatorCallback):
        actualExistingKey = self.GetExistingKeyByIdOrNull(key.Id)
        if actualExistingKey == key: # the deep equality should happen here...
            #print("Found key, returning existing item")
            return self._readInDictionary[actualExistingKey]
        
        #print("did not find key or it was unequal")
        # either the object hasn't been added, or it doesn't equal the desired key
        createdObject = creatorCallback()
        if actualExistingKey != None: # remove existing item if not equal
            #print("different key was found")
            self.RemoveObjectByKey(cast(CacheKey, actualExistingKey)) # not sure why the cast is necessary, we check for None right above it...
 
        self.SetObject(key, createdObject) # now that it is removed, add the created object
        return createdObject
        
    def SetObject(self, key :CacheKey, item : Any):
        self._readInDictionary[key] = item

    def GetObject(self, key:CacheKey):
        if key  in self._readInDictionary.keys():
            return self._readInDictionary[key]
        return None

    def GetObjectById(self, id:str):
        existingKey = self.GetExistingKeyByIdOrNull(id)
        if existingKey == None:
            return None
        return self._readInDictionary[existingKey]

    def GetExistingKeyByIdOrNull(self, id: str):
        for existingKey in self._readInDictionary.keys():
            if id == existingKey.Id:
                return existingKey
        return None

    def IsKeyRegistered(self, key:CacheKey):
        return key in self._readInDictionary == None

    def IsIdRegistered(self, id:str):
        return self.GetExistingKeyByIdOrNull(id)==None

    def RemoveObjectById(self, id:str):
        existingKey = self.GetExistingKeyByIdOrNull(id)
        if existingKey != None:
            del self._readInDictionary[existingKey]

    def RemoveObjectByKey(self, key:CacheKey):
        if key in self._readInDictionary.keys():
            del self._readInDictionary[key]    

    def SaveCache(self):
        with open(self._filePath, 'wb') as handle:
            dill.settings['recurse'] = True
            dill.dump(self._readInDictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)        
    