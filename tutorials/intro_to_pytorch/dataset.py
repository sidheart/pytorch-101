# Tutorial: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

show_dataset_sample_plot = False
show_dataloader_sample_plot = True


# load the data into a dataset
# you can make custom Pytorch Dataset classes by implementing three functions: __init__, __len__, and __getitem__.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# display data from the dataset
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
if (show_dataset_sample_plot):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

# dataloaders let you iterate over datasets in batches, you can distribute and parallelize this
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
train_features, train_labels = next(iter(train_dataloader)) # these are tensors
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
if show_dataloader_sample_plot:
    img = train_features[0].squeeze()
    label = train_labels[0] # this is a single element tensor for some reason
    plt.title(labels_map[label.item()])
    plt.imshow(img, cmap="gray")
    plt.show()
