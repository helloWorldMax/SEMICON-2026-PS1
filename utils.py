import torch


def calculate_psnr(output, target):
    mse = torch.mean((output - target) ** 2)

    if mse == 0:
        return float("inf")

    max_pixel = 1.0

    psnr = 10 * torch.log10(
        (max_pixel ** 2) / mse
    )

    return psnr.item()
