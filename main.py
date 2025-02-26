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

N = 16

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

    print(data)
    
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

display(HTML("<style>.container { width:100% !important; }</style>"))

X, Y = load_samples("dataset_cleaned.csv",(64,64)) # for model that require higher than 64x64 resolution

print("\nX shape: ")
print(X.shape)
print("\nY shape: ")
print(Y.shape)

# get train and test sets
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.10, random_state=42) # this random_state is similar to the bootstrap results

# convertir tensores numpy a pytorch: torch.from_numpy()
# convertir datos torch a numpy: X.cpu().numpy()
# enviar datos a la GPU: X.to(device)

device = set_device()

X = torch.from_numpy(X)
Y = torch.from_numpy(Y)

# DEFINE MODEL
# The model is based on a VGG architecture, and is composed of the following layers:
#
#   - Input layer       (linear activation)          (Image dimensions 64x64X3)
#
#   - Convolution layer (8 filters, 3x3, stride 1)   (Image dimensions 64x64x8)
#   - Batch Normalization
#   - Convolution layer (8 filters, 3x3, stride 1)   (Image dimensions 64x64x8)
#   - Batch Normalization
#   - MaxPool layer     (2x2, stride 2)              (Image dimensions 32x32x8)
#
#   - Convolution layer (16 filters, 3x3, stride 1)  (Image dimensions 32x32x16)
#   - Batch Normalization
#   - Convolution layer (16 filters, 3x3, stride 1)  (Image dimensions 32x32x16)
#   - Batch Normalization
#   - MaxPooling layer  (2x2, stride 2)              (Image dimensions 16x16x16)
#
#   - Convolution layer (32 filters, 3x3, stride 1)  (Image dimensions 16x16x32)
#   - Batch Normalization
#   - Convolution layer (32 filters, 3x3, stride 1)  (Image dimensions 16x16x32)
#   - Batch Normalization
#   - Convolution layer (32 filters, 3x3, stride 1)  (Image dimensions 16x16x32)
#   - Batch Normalization
#   - MaxPooling layer  (2x2, stride 2)              (Image dimensions 8x8x32)
#
#   - Convolution layer (64 filters, 3x3, stride 1)  (Image dimensions 8x8x64)
#   - Batch Normalization
#   - Convolution layer (64 filters, 3x3, stride 1)  (Image dimensions 8x8x64)
#   - Batch Normalization
#   - Convolution layer (64 filters, 3x3, stride 1)  (Image dimensions 8x8x64)
#   - Batch Normalization
#   - MaxPooling layer  (2x2, stride 2)              (Image dimensions 4x4x64)
#
#   - Convolution layer (64 filters, 3x3, stride 1)  (Image dimensions 4x4x64)
#   - Batch Normalization
#   - Convolution layer (64 filters, 3x3, stride 1)  (Image dimensions 4x4x64)
#   - Batch Normalization
#   - Convolution layer (64 filters, 3x3, stride 1)  (Image dimensions 4x4x64)
#   - Batch Normalization
#   - MaxPooling layer  (2x2, stride 2)              (Image dimensions 2x2x64)
#
#   - Full Connected    (256 neurons)
#   - Full Connected    (256 neurons)
#
#   - Output layer      (linear activation)

class ConvolutionalNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 8, kernel_size=3, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.LazyConv2d(8, kernel_size=3, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten()
        )

        # Capa densa (fully connected)
        self.fc = nn.Sequential(
            nn.LazyLinear(128),
            nn.LazyLinear(1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.fc(x)
        return x

def train (X_images, Y_angles, model, loss_fn, optimizer, epochs=1000, device='cuda', trace=100):
    size = len(X_images)
    model = model.to(device)
    model.train()
    X_gpu, Y_gpu = X_images.to(device), Y_angles.to(device)
    for epoch in range(epochs):
        optimizer.zero_grad() # reset gradients

        pred = model(X_gpu) # propagate

        loss = loss_fn((pred + 1)/2, Y_gpu) # prediction error
        
        for item in loss:
            loss_root = math.sqrt(item)
            if loss_root > 0.5:
                item = (1 - loss_root)*(1 - loss_root)

        loss.backward() # back propagation
        optimizer.step() # update parameters

        if (epoch + 1) % trace == 0: # traces
            loss, current = loss.item(), epoch + 1
            print(f"loss: {loss:>7f}  [{current:>5d} /{epochs:>5d}]")

model = ConvolutionalNN()
optimizer = torch.optim.Adam(model.parameters())
train(X.to(torch.float32).permute(0,3,1,2), Y.to(torch.float32), model.to(torch.float32), nn.MSELoss(), optimizer, epochs=5000, trace=1)
