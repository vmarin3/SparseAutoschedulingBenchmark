from .AbstractWrapper import AbstractWrapper
from ..BinsparseFormat import BinsparseFormat
import numpy as np

class NumpyWrapper(AbstractWrapper):
    def __init__(self):
        pass

    def from_benchmark(self, array):
        if array.data["format"] == "dense":
            data = np.array(array.data["values"]).reshape(array.data["shape"])
            return data
        if array.data["format"] == "COO":
            I = []
            i = 0
            while "indices_" + str(i) in array.data:
                I.append(array.data["indices_" + str(i)])
                i += 1
            V = array.data["values"]
            shape = array.data["shape"]
            data = np.zeros(shape, dtype=V.dtype)
            data[tuple(I)] = V
            return data
        raise ValueError("Unsupported format: " + array.data["format"])

    def to_benchmark(self, array):
        return BinsparseFormat.from_numpy(array)

    def lazy(self, array):
        return array

    def compute(self, array):
        return array
    
    def __getattr__(self, name):
        return getattr(np, name)
