import os
import torch
from PIL import Image
from torchvision import transforms

from model import ImageRestorationCNN


INPUT_DIR = "data/test"
OUTPUT_DIR = "outputs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load model
model = ImageRestorationCNN().to(device)

model.load_state_dict(
    torch.load(
        "models/image_restoration_cnn.pth",
        map_location=device
    )
)

model.eval()


transform = transforms.ToTensor()
to_image = transforms.ToPILImage()


# Process test images
for filename in os.listdir(INPUT_DIR):

    if not filename.lower().endswith(
        (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    ):
        continue

    input_path = os.path.join(INPUT_DIR, filename)

    image = Image.open(input_path).convert("L")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        restored = model(image_tensor)

    restored = torch.clamp(restored, 0, 1)

    output_image = to_image(restored.squeeze(0).cpu())

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    output_image.save(output_path)

    print("Restored:", filename)

print("Testing completed.")
