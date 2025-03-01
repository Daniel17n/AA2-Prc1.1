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
        imgname = data.iloc[i,0]
        img = cv2.imread(imgname)
        if size != None:
            img = cv2.resize(img, size)
        X.append(img)
        Y.append(data.iloc[i,1])

    return (np.array(X), np.array(Y))

class ConvolutionalNN3(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            # 64 x 64
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 32 x 32
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 16 x 16
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),  # Num_features * ancho_img * alto_img
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_layers = self.layers(x)
        output = self.fc(x_layers)
        return output

def custom_loss(y_pred, y_true):
    diff = y_pred - y_true
    # actual_diff = torch.where(diff > 0.5, 1 - diff, diff)
    loss = torch.mean(diff ** 2) # Mean Square Error de manual
    return loss

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
            loss_value = math.sqrt(loss)  # Root square of loss
            current = epoch + 1
            epochs_list.append(current)
            loss_list.append(loss_value)

            print(f"loss_sqrt: {loss_value:>7f} | [{current:>5d}/{epochs:>5d}]")

    return epochs_list, loss_list

def train_and_graph(X, Y, model, optimizer, device, color, label="Model"):
    epochs, loss = train(
        X.to(torch.float32).permute(0,3,1,2), 
        Y.to(torch.float32), 
        model=model.to(torch.float32), 
        loss_fn=custom_loss, 
        optimizer=optimizer, 
        epochs=200, 
        device=device, 
        trace=1, 
        label=label)

    plt.plot(epochs, loss, linestyle='-', linewidth=2, color=color, label=label)
    min_loss, min_epoch = min(loss), epochs[np.argmin(loss)]
    plt.scatter(min_epoch, min_loss, color=color, marker='o', label=f"Min {label}: {min_loss:.4f} at epoch {min_epoch}")

def main():
    # display(HTML("<style>.container { width:100% !important; }</style>"))
    X, Y = load_samples("dataset_cleaned.csv", (64,64) )
    print(f"\nX shape: {X.shape}")
    print(f"\nY shape: {Y.shape}")
    
    # get train, validation and test sets
    X_rest, X_test, Y_rest, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.10, random_state=42) # this random_state is similar to the bootstrap results
    X_train, X_validation, Y_train, Y_validation = sklearn.model_selection.train_test_split(X_rest, Y_rest, test_size=0.10, random_state=42) # this random_state is similar to the bootstrap results
    
    device = set_device()
    X_train_torch = torch.from_numpy(X_train).float()
    Y_train_torch = torch.from_numpy(Y_train).float()
    model3 = ConvolutionalNN3()
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=1e-3)
    plt.figure(figsize=(12, 8), dpi=300)

    train_and_graph(X_train_torch, Y_train_torch, model3, optimizer3, device, 'b')

    # Etiquetas y formato
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss (sqrt)", fontsize=14)
    plt.title("Training Loss Progress", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Guardar la imagen
    plt.savefig("model_loss.png", bbox_inches='tight')

    # HACER EL TEST CON X_TEST Y Y_TEST

if __name__ == "__main__":
    main()