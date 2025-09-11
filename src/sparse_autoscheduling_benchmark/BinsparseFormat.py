import numpy as np

class BinsparseFormat:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def from_numpy(array : np.ndarray):
        data = dict()
        data["format"] = "dense"
        data["shape"] = array.shape
        data["values"] = array.flatten()
        return BinsparseFormat(data)

    @staticmethod
    def from_coo(I, V, shape):
        data = dict()
        data["format"] = "COO"
        for i in range(len(I)):
            data["indices_" + str(i)] = I[i]
        data["values"] = V
        data["shape"] = shape
        return BinsparseFormat(data)
