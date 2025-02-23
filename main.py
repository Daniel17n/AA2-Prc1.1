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
    if torch.backends.mps.is_available(): # para usar con mac silicon
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    torch.set_default_device(device)
    print(f"Using {torch.device(device)} device")
    return torch.device(device)

def load_samples (directory, size=None):
    data = pd.read_csv(directory, sep='\t', header=None) 

    print(data)
    
    X = []
    Y = []
    for i in range(len(data)):
        print('.', end='')
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

print(X.shape)
print(Y.shape)

# get train and test sets
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.10, random_state=42) # this random_state is similar to the bootstrap results

# imprime algunos ejemplos
fig, axs = plt.subplots(N//4,4,figsize=(10, 10*N//16))
axs = axs.flatten()
plt.axis('off')

im0 = 567
for i in range(N):
    axs[i].axis('off')
    axs[i].imshow(X[im0+i])
    #axs[i].title.set_text(f"$angle$ = {(Y[im0+i]):.3f}")
    axs[i].title.set_text(f"$angle$ = {(360*Y[im0+i]):.1f}°")

plt.savefig("test.png", bbox_inches='tight')

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
    def __init__(self, nh, no):
        super().__init__()

        self.layers = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloque 2
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloque 3
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloque 4
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloque 5
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten()
        )

        # Capa densa (fully connected)
        self.fc = nn.Sequential(
            nn.Linear(64 * (2**nh) * (2**nh), 256),
            nn.ReLU(),
            nn.Linear(256, no)
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.fc(x)
        return x

# Prueba con una imagen de 64x64
model = ConvolutionalNN(nh=1, no=10)
x = torch.randn(1, 3, 64, 64)  
output = model(x)
print(output.shape)  # Debe ser [1, 10]
