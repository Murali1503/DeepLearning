#!/usr/bin/env python3

import scipy.stats as stats
import numpy as np
import torch as t

# arr1 = np.array([1,23,4,0,5,10])
arr1 = np.array([1])
arr2 = np.array([90,1,4,6,2,5])

T_value,p_value = stats.ttest_ind(arr1,arr2)

print(T_value)
print(p_value)

print(t.tensor(arr1))
