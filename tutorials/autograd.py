# https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True) # param 1
b = torch.randn(3, requires_grad=True) # param 2
z = torch.matmul(x, w)+b 
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
print(f"Loss is {loss}") # Loss is a scalar here
# w and b are the params to optimize
# w.grad = (d_loss / d_w) and b.grad = (d_loss / d_b)
# using gradients lets us find local minima for the loss function!
loss.backward()
print(w.grad)
print(b.grad)

# Once we've trained a model or want to fine-tune it by freezing parameters, we can disable gradient tracking
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

# In the example above, our loss function had a scalar value so things were easy
# If the loss function output is a tensor, then the gradient has to be a Jacobian matrix
# Pytorch calculates this ezpz for us
inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nSecond call\n{inp.grad}") # gradients change here because Pytorch accumulates, IRL an optimizer will zero out gradients for us automatically
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")
