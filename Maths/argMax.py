import torch as tc
import numpy as np

#Min/Max function gives min/max values where as argMax/Min give position of the value where they are located

#Min/Min and args in a single vector or single matrix
min = np.array([1,-1,0,3,2,6])
max = np.array([1,-1,0,3,2,6])


print(f"Maximum value { np.max(max) }")
print(f"Minimum value { np.min(min) }")


argMin = np.argmin(min)
argMax = np.argmax(max)

print(f"Position of the argMin: {argMin}")
print(f"Position of the argMax: {argMax}")

#Min/Max and args in a n-d array or mxn matrix
mat = np.array([[ 1,-2,0],
                [0,2,-1],
                [ 1,-1,1] ])

print(mat)
mmin = np.min(mat)
mmax = np.max(mat)

argmmin = np.argmin(mat,axis=0) #gets the position of min value from each column (and return the row number)
argmmin2 = np.argmin(mat,axis=1) #gets the position of min value from each row (and return the column number)


print(f"min value = {mmin}")
print(f"min value = {mmax}")

print(f"Position of the argmin value = {argmmin}")
print(f"Position of the argmin value = {argmmin2}")


#Using torch
tmat = tc.tensor(mat)

print(tmat)

tmin = tc.min(tmat)
tmax = tc.max(tmat)

print(f"Minimum value = {tmin}")
print(f"Maximum value = {tmax}")

tmatmin = tc.min(tmat,0)
tmatmin2 = tc.min(tmat,1)

print(tmatmin.values)
print(tmatmin2.indices)
