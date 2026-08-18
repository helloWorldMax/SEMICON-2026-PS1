import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ImageRestorationDataset
from model import ImageRestorationCNN


NOISY_DIR = "train/NoisyLR"
GT_DIR = "train/GT"

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


dataset = ImageRestorationDataset(
    NOISY_DIR,
    GT_DIR
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


model = ImageRestorationCNN().to(device)

criterion = nn.L1Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


print("Device:", device)
print("Training samples:", len(dataset))


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for noisy, gt in loader:

        noisy = noisy.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()

        output = model(noisy)

        loss = criterion(output, gt)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.6f}"
    )


torch.save(
    model.state_dict(),
    "restoration_model.pth"
)

print("Training completed.")
print("Model saved as restoration_model.pth")
