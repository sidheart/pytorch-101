# Tutorial: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# FashionMNIST labels
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# my machine has cuda ;)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Flatten takes the 2D image and lays its pixels out in a 1D contiguous array
# Sequential applies a sequence of transforms in order
# Each Linear(x, y) step takes x input features and outputs a tensor of y output features
# ReLU means rectified linear activation function if you aren't a fucking nerd it just means F(x) = max(0, x)
# This class doesn't do shit, but if you train it on the dataset it eventually will
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() 
        # Sequential just applys t
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Define the model and put it on the GPU
model = NeuralNetwork().to(device)

# Use the model to get a prediction
x = torch.rand(1, 28, 28, device=device)
logits = model(x)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {labels_map[y_pred.item()]}")

# Each model is built of layers and many layers are parameterized, i.e. have associated weights and biases that are optimized during training
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")