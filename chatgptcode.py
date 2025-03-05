import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sklearn.model_selection

# Configuración
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001


# Dataset personalizado
class ArrowDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        angle = self.data.iloc[idx]['angle']

        # Leer imagen
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0  # Normalización [0, 1]
        img = np.expand_dims(img, axis=0)  # Añadir canal

        # Normalizar ángulo entre 0 y 1
        angle_normalized = angle

        return torch.tensor(img, dtype=torch.float32), torch.tensor(angle_normalized, dtype=torch.float32)


# Modelo CNN
class ArrowAngleCNN(nn.Module):
    def __init__(self):
        super(ArrowAngleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid()  # Salida normalizada entre 0 y 1
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x.squeeze()


# Función de pérdida personalizada considerando el error circular
def circular_mse_loss(y_pred, y_true):
    diff = torch.abs(y_pred - y_true)
    diff = torch.minimum(diff, 1 - diff)  # Manejar el cierre del círculo (0 == 1)
    return torch.mean(diff ** 2)


# Cargar datos# Cargar CSV separando por tabulador
df = pd.read_csv('dataset_cleaned.csv', sep='\t', header=None, names=['image_path', 'angle'])
train_df, val_df = sklearn.model_selection.train_test_split(df, test_size=0.2, random_state=42)

train_dataset = ArrowDataset(train_df)
val_dataset = ArrowDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Entrenamiento
model = ArrowAngleCNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses, val_losses = [], []
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, angles in train_loader:
        imgs, angles = imgs.to(DEVICE), angles.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs).squeeze()
        loss = criterion(outputs, angles)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, angles in val_loader:
            imgs, angles = imgs.to(DEVICE), angles.to(DEVICE)
            outputs = model(imgs)
            loss = circular_mse_loss(outputs, angles)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {math.sqrt(train_loss):.8f} | Val Loss: {math.sqrt(val_loss):.8f}")
    print("-" * 50)



# Visualizar pérdidas
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig("model_loss.png", bbox_inches='tight')