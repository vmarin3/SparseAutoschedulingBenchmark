from pyparsing import Any
import numpy as np

class BinsparseFormat:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def from_numpy(array : np.ndarray) -> "BinsparseFormat":
        data: dict[str, Any] = {}
        data["format"] = "dense"
        data["shape"] = array.shape
        data["values"] = array.flatten()
        return BinsparseFormat(data)

    @staticmethod
    def from_coo(I: tuple[np.ndarray, ...], V: np.ndarray, shape: tuple[int, ...]) -> "BinsparseFormat":
        data: dict[str, Any] = {}
        data["format"] = "COO"
        for i in range(len(I)):
            data["indices_" + str(i)] = I[i]
        data["values"] = V
        data["shape"] = shape
        return BinsparseFormat(data)
