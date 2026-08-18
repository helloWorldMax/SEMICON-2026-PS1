import os
import numpy as np
import torch

from model import ImageRestorationCNN


TEST_DIR = "Test_NoisyLR/NoisyLR"
OUTPUT_DIR = "outputs"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


os.makedirs(OUTPUT_DIR, exist_ok=True)


model = ImageRestorationCNN().to(device)

model.load_state_dict(
    torch.load(
        "restoration_model.pth",
        map_location=device
    )
)

model.eval()


files = sorted([
    f for f in os.listdir(TEST_DIR)
    if f.endswith(".npy")
])


print("Test samples:", len(files))
print("Device:", device)


with torch.no_grad():

    for filename in files:

        path = os.path.join(
            TEST_DIR,
            filename
        )

        noisy = np.load(path).astype(
            np.float32
        )

        noisy = torch.from_numpy(
            noisy
        ).unsqueeze(0).unsqueeze(0)

        noisy = torch.nn.functional.interpolate(
            noisy,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        noisy = noisy.to(device)

        output = model(noisy)

        output = output.squeeze().cpu().numpy()

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        np.save(
            output_path,
            output
        )


print("Testing completed.")
print("Outputs saved to:", OUTPUT_DIR)
