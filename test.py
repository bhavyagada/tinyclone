import numpy as np
from tensor import Tensor

x = np.random.randn(1, 3)
W = np.random.randn(3, 3)
npout = x.dot(W)
print("numpy")
print(x, end="\n")
print(W, end="\n")
print(npout, end="\n")

x = Tensor(x)
W = Tensor(W)
tinyout = x.dot(W)
print("tinygrad")
print(x.data, end="\n")
print(W.data, end="\n")
print(tinyout.data, end="\n")

assert np.array_equiv(npout, tinyout.data), "dot results do not match"

x = np.random.randn(1, 3)
x_tensor = Tensor(x)
print(x, end="\n")
print(x_tensor.data, end="\n")

np_sum = np.sum(x)
print("numpy sum")
print(np_sum, end="\n")

tiny_sum = x_tensor.sum()
print("tinygrad sum")
print(tiny_sum.data, end="\n")

assert np.array_equal(np_sum, tiny_sum.data), "Sum results do not match!"

np_relu = np.maximum(x, 0)
print("numpy ReLU")
print(np_relu, end="\n")

tiny_relu = x_tensor.relu()
print("tinygrad ReLU")
print(tiny_relu.data, end="\n")

assert np.array_equal(np_relu, tiny_relu.data), "ReLU results do not match!"
