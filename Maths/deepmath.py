import numpy as np
import torch as tc


##Transpose a vector using numpy
npArr = np.array([[1, 2, 3, 4]])

print(npArr), print(" ")

npTArr = np.transpose(npArr)

print(npTArr), print(" ")


##Transpose a vector using torch
tArr = tc.tensor([[1, 2, 3, 4]])

print(tArr), print(" ")

TtArr = tArr.T  ##tc.transpose or tArr.T to do transpose operation on vector

print(TtArr), print(" ")


##Multi dimension array or ND-Array
npNDArr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

print(npNDArr), print(" ")

print(np.transpose(npNDArr)), print(" ")


tNDArr = tc.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

print(tNDArr), print(" ")

print(tNDArr.T), print(" ")
