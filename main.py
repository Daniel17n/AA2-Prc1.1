import os
import pandas as pd
import numpy as np
import cv2
import sklearn.model_selection
import matplotlib.pyplot as plt
import torch
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

class ConvolutionalNN (nn.Module):
    def __init__(self, ni, nh, no):
        super.__init__()
        self.layers = nn.Sequential(
            nn.Linear(ni, nh),
            nn.Conv2d(1,1)
            nn.Linear(nh, no)
        )

