import torch as tc
import numpy as np
import torch.nn as tn
import matplotlib.pyplot as pt


#Soft max

z = [1,2,3,4]
#Soft-Max using numpy

num = np.exp(z)
den = np.sum(np.exp(z))
sigma = num/den

print(f"Numerator: {num}")
print(f"Denominator: {den}")

print(f"Sigma: {sigma}")
print(f"Sum sigma: {np.sum(sigma)}")


#With some random Integers
rand = np.random.randint(low=-5,high=5,size=25)
#This says to generate random numbers(Integers) from range (-5 to +5) with a array size of 25
print(rand)

randnum = np.exp(rand)
randden = np.sum(np.exp(rand))
randsig = randnum/randden

print(np.sum(randsig))

# pt.plot(rand,randsig,'ko')
# pt.xlabel("Orginal number (randnum)")
# pt.ylabel("SoftMaxed")
# pt.plot()


#SoftMax using Pytorch
tensor = tc.tensor(z).float()
softfun = tn.Softmax(dim=0)
res = softfun(tensor)


print(res)
