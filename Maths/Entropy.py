import numpy as np
import torch as t
import torch.nn.functional as tc

p = 0.25

ntropy = -p * np.log(p) - (1 - p) * np.log(1 - p)  # probability of the event is happening and the probability is not happening.
#
# print(entropy)


p = [
    0.25,
    0.75,
]  # for a probability of single event we need to implement the converse of it

H = 0
for i in p:
    H -= i * np.log(i)
print(f"Probability of the single event {H}")


# Cross-Entropy
p = [1, 0]  # Probability of the given image is cat.
q = [0.75, 0.25]  # Probability of the model that it can predict the given image is cat

H = 0
for i in range(len(p)):
    H -= p[i] * np.log(q[i])
print(f"Cross-entropy { H }")

# Cross-Entropy using pytorch
p_entropy = t.tensor(p).float()
q_entropy = t.tensor(q).float()

h = tc.binary_cross_entropy(p_entropy, q_entropy)

print("\nCross Entropy using torch")
print(h)
