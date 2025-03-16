import numpy as np
import torch as tp


# Dot Product using numpy of two vector or two matrices/tensors.

np1 = np.array([1, 2, 3, 4])
np2 = np.array([0, 1, 0, -2])


print(np.dot(np1, np2))

# Dot Product using Pytorch of two vector or two matrices/tensors.
np1 = tp.tensor([1, 2, 3, 4])

np2 = tp.tensor([0, 1, 0, -2])


print(tp.dot(np1, np2))
