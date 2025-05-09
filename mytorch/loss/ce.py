from mytorch import Tensor
from mytorch.activation import softmax
import numpy as np

def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
    "TODO: implement Categorical Cross Entropy loss"
    entropy = -(label * preds.log()).sum()
    return entropy 
