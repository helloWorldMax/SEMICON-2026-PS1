import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageRestorationDataset(Dataset):
    def __init__(self, degraded_dir, clean_dir):
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir

        self.images = sorted([
            f for f in os.listdir(degraded_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ])

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        filename = self.images[index]

        degraded_path = os.path.join(self.degraded_dir, filename)
        clean_path = os.path.join(self.clean_dir, filename)

        degraded = Image.open(degraded_path).convert("L")
        clean = Image.open(clean_path).convert("L")

        degraded = self.transform(degraded)
        clean = self.transform(clean)

        return degraded, clean
