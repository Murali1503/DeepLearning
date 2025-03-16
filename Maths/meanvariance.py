

import numpy as np

arr = np.array([1,2,3,4,5,6,7,8])
n = len(arr)

mean = np.sum(arr) / n

print(f"Numpy mean {np.mean(arr)}")
print(f"Regular mean { mean }")


var = (1/n-1 * np.sum( arr-mean )**2)

print("Variance"+str(np.var(arr,ddof=1)))
