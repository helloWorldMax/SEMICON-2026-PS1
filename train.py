import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import ImageRestorationCNN
from dataset import ImageRestorationDataset


# Paths
DEGRADED_DIR = "data/degraded"
CLEAN_DIR = "data/clean"

# Training settings
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Dataset
dataset = ImageRestorationDataset(
    degraded_dir=DEGRADED_DIR,
    clean_dir=CLEAN_DIR
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# Model
model = ImageRestorationCNN().to(device)

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# Training
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0.0

    for degraded, clean in dataloader:

        degraded = degraded.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()

        restored = model(degraded)

        loss = criterion(restored, clean)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.6f}"
    )


# Create model directory
os.makedirs("models", exist_ok=True)

# Save trained model
torch.save(
    model.state_dict(),
    "models/image_restoration_cnn.pth"
)

print("Training completed.")
print("Model saved to models/image_restoration_cnn.pth")
