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

# ReLU means rectified linear activation function
# if you aren't a fucking nerd it just means F(x) = max(0, x)
# Each Linear(x, y) step takes x input features and outputs a tensor of y output features
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
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
print(model)

# Use the model to get a prediction
x = torch.rand(1, 28, 28, device=device)
logits = model(x)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {labels_map[y_pred.item()]}")
