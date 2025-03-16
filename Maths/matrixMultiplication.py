
import torch as tc
import numpy as np


#Matrix Multipication using numpy
A = np.random.randn(3,4)
B = np.random.randn(4,5)
c1 =np.random.randn(4,3)

#A@B is same as np.matmul(dm1,dm2)
print(np.matmul(A,B))

print(np.round(A@B,2))

print(np.round(c1.T@B,2))


#Matrix multiplication using pytorch
C = tc.randn(3,4)
D = tc.randn(4,5)

mat = tc.tensor(A,dtype=tc.float)
print(tc.matmul(C,mat.T))
