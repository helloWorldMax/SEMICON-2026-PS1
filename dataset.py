import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ImageRestorationDataset(Dataset):

    def __init__(self, noisy_dir, gt_dir):

        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir

        self.files = sorted([
            f for f in os.listdir(noisy_dir)
            if f.endswith(".npy")
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]

        noisy_path = os.path.join(
            self.noisy_dir,
            filename
        )

        gt_path = os.path.join(
            self.gt_dir,
            filename
        )

        noisy = np.load(noisy_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)

        noisy = torch.from_numpy(noisy).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)

        noisy = F.interpolate(
            noisy.unsqueeze(0),
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return noisy, gt
