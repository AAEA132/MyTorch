from mytorch import Tensor
import numpy as np

def MeanSquaredError(preds: Tensor, actual: Tensor):
    "TODO: implement Mean Squared Error loss"
    error = preds - actual
    squared_error = error**2
    sum = squared_error.sum()
    size = Tensor(np.array(squared_error.data.size,dtype=np.float64))
    size = size**-1
    return sum * size
