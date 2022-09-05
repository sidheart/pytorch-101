# Tutorial: https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
import torch
import numpy as np

# tensors can be made from raw arrays
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# lots of functions for transforming tensors
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
x_complex = torch.rand_like(x_data, dtype=torch.complex64)
print(f"Complex Tensor: \n {x_complex} \n")

# you can generate tensors just from dimensions
shape = (2,3,4)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
complex_tensor = torch.complex(ones_tensor, zeros_tensor) # so complex but also so real
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
print(f"Complex Tensor: \n {complex_tensor}") 
# tensors have useful attributes
print(f"Shape of ones_tensor: {ones_tensor.shape}")
print(f"Datatype of complex tensor: {complex_tensor.dtype}")
print(f"Device complex tensor is stored on: {complex_tensor.device}")

# lots of useful operations on tensors https://pytorch.org/docs/stable/torch.html
tensor = torch.ones(4, 4)
if torch.cuda.is_available():
    tensor = tensor.to("cuda") # move tensor to the GPU if available
print(f"Device tensor is stored on: {tensor.device}") # cuda:0 on my machine ;)
print(f"Trace of tensor is {torch.trace(tensor)}")
# ezpz slicing and indexing
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0 # sets the whole 1st (0-indexed) column to 0
print(tensor)
# joining
t1 = torch.cat([tensor, tensor, tensor], dim=1) # extends
print(f"Cat tensor t1: \n {t1}")
t2 = torch.stack([tensor, tensor, tensor]) # dimensional
print(f"Stacked tensor t2: \n {t2}")

# Arithmetic, basically just use @ for matrix multiply and * for element-wise multiply
print(f"Tensor T is: \n {tensor.T}") # T = transpose, NEAT!
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
# Single element tensors can be converted back into Python types
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# Efficient in-place & side effect operations are suffixed with _
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)
tensor.t_() # god I love transpose
print(tensor)