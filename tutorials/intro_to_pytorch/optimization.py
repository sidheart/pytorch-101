# Tutorial: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# my machine has cuda ;)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters are adjustable parameters that let you control the model optimization process
learning_rate = 1e-3 # how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.
batch_size = 64 # the number of data samples propagated through the network before the parameters are updated
epochs = 10 # the number times to iterate over the dataset

# This is just copypasta from dataset.py and neural_network.py 
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

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

model = NeuralNetwork().to(device)

# Optimization consist of a train and a validation/test loop
# Train iterates over the training data, and Validation/Test iterates over the test data to see if performance has improved
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Use the GPU baby
        X = X.to(device)
        y = y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Don't compute gradients during testing, we only need the forward loop so no need to waste GPU ;)
    with torch.no_grad():
        for X, y in dataloader:
            # Use the GPU baby
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Loss functions give us a measure of how good/bad the model's prediction was
loss_fn = nn.CrossEntropyLoss()
# Optimization algorithms define how to adjust model parameters to reduce model error in each training step
# The algorithm below is Stochastic Gradient Descent, PyTorch offers lots more https://pytorch.org/docs/stable/optim.html
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# Optimize the model
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")