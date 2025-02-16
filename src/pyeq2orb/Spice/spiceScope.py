from typing import List, Optional
import os
import spiceypy as spice

class spiceScope:
    #_standardGregorianFormat = "DD Mon YYYY-HH:MM:SC.######"

    def __init__(self, kernelPaths : List[str], baseDirectory : Optional[str]):
        self._kernelPaths :List[str] = []
        self._baseDirectory = baseDirectory
        if self._baseDirectory is not None:
            for partialPath in kernelPaths:
                self._kernelPaths.append(os.path.join(self._baseDirectory, partialPath))
        else:
            self._kernelPaths = kernelPaths

    def __enter__(self):
        for kernel in self._kernelPaths:
            spice.furnsh(kernel)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for kernel in self._kernelPaths:
            spice.unload(kernel)

    @staticmethod
    def etToUtc(et: float)->str:
        return spice.et2utc(et, 'C', 6)

    @staticmethod
    def utcToEt(utc : str)->float:
        return spice.utc2et(utc)
