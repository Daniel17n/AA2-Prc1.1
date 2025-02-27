import math
import os
import pandas as pd
import numpy as np
import cv2
import sklearn.model_selection
import matplotlib.pyplot as plt
import torch
from torch import nn
from IPython.display import display, HTML

def set_device ():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    torch.set_default_device(device)
    print(f"\nUsing {torch.device(device)} device")
    return torch.device(device)

def load_samples (directory, size=None):
    data = pd.read_csv(directory, sep='\t', header=None) 
    
    X = []
    Y = []
    for i in range(len(data)):
        # print('.', end='')
        # carga la imagen
        imgname = data.iloc[i,0]
        img = cv2.imread(imgname)
        if size != None:
            img = cv2.resize(img, size)
        X.append(img)
        Y.append(data.iloc[i,1])

    return (np.array(X), np.array(Y))

class ConvolutionalNN1(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.LazyConv2d(16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # nn.LazyConv2d(16, kernel_size=3, stride=1),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),

            nn.Flatten()
        )

        # Capa densa (fully connected)
        self.fc = nn.Sequential(
            nn.LazyLinear(256),
            nn.LazyLinear(1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.fc(x)
        return x
    
class ConvolutionalNN2(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.LazyConv2d(16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.LazyConv2d(16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten()
        )

        # Capa densa (fully connected)
        self.fc = nn.Sequential(
            nn.LazyLinear(256),
            nn.LazyLinear(1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.fc(x)
        return x

class ConvolutionalNN3(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.LazyConv2d(16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.LazyConv2d(16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.LazyConv2d(16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten()
        )

        # Capa densa (fully connected)
        self.fc = nn.Sequential(
            nn.LazyLinear(256),
            nn.LazyLinear(1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.fc(x)
        return x

def custom_min_mse_loss(input: torch.Tensor, target: torch.Tensor):
    diff = input - target
    min_diff = torch.min(diff, 1 - diff)
    loss = min_diff ** 2
    return loss.mean()

def train(X_images, Y_angles, model, loss_fn, optimizer, epochs=1000, device='cuda', trace=100, label="Model"):
    model = model.to(device)
    model.train()
    X_gpu, Y_gpu = X_images.to(device), Y_angles.to(device)

    epochs_list = []
    loss_list = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()  # Reset gradients
        pred = model(X_gpu)  # Forward pass
        loss = loss_fn(pred, Y_gpu)  # Compute loss

        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters

        if (epoch + 1) % trace == 0:  # Store loss every `trace` epochs
            loss_value = math.sqrt(loss.item())  # Root square of loss
            current = epoch + 1
            epochs_list.append(current)
            loss_list.append(loss_value)

            print(f"{label} | loss_sqrt: {loss_value:>7f} | [{current:>5d}/{epochs:>5d}]")

    return epochs_list, loss_list

def plotGraph(X, Y, model, optimizer, device, label, color):
    epochs, loss = train(X.to(torch.float32).permute(0,3,1,2), Y.to(torch.float32), model.to(torch.float32), custom_min_mse_loss, optimizer, epochs=200, device=device, trace=1, label=label)
    plt.plot(epochs, loss, linestyle='-', linewidth=2, color=color, label=label)
    min_loss, min_epoch = min(loss), epochs[np.argmin(loss)]
    plt.scatter(min_epoch, min_loss, color=color, marker='o', label=f"Min {label}: {min_loss:.4f} at epoch {min_epoch}")

def main():
    display(HTML("<style>.container { width:100% !important; }</style>"))
    X, Y = load_samples("dataset_cleaned.csv",(64,64))
    print(f"\nX shape: {X.shape}")
    print(f"\nY shape: {Y.shape}")
    
    # get train and test sets
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.10, random_state=42) # this random_state is similar to the bootstrap results
    
    device = set_device()
    X = torch.from_numpy(X_train)
    Y = torch.from_numpy(Y_train)
    model1 = ConvolutionalNN1()
    model2 = ConvolutionalNN2()
    model3 = ConvolutionalNN3()
    optimizer1 = torch.optim.Adam(model1.parameters())
    optimizer2 = torch.optim.Adam(model2.parameters())
    optimizer3 = torch.optim.Adam(model3.parameters())

    plt.figure(figsize=(12, 8), dpi=300)

    plotGraph(X, Y, model1, optimizer1, device, "2 Layer Model", 'r')
    plotGraph(X, Y, model2, optimizer2, device, "3 Layer Model", 'g')
    plotGraph(X, Y, model3, optimizer3, device, "4 Layer Model", 'b')

    # Etiquetas y formato
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss (sqrt)", fontsize=14)
    plt.title("Training Loss Progress for Different Models", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Guardar la imagen
    plt.savefig("models_loss_comparison.png", bbox_inches='tight')

    # HACER EL TEST CON X_TEST Y Y_TEST

if __name__ == "__main__":
    main()