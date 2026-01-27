import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms


def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406],
                        device=img.device,
                        dtype=img.dtype)
    std  = torch.tensor([0.229, 0.224, 0.225],
                        device=img.device,
                        dtype=img.dtype)

    mean = mean[:, None, None]
    std  = std[:, None, None]

    return img * std + mean


def show_image(dataloader):
    image, mask = next(iter(dataloader))

    img = image.clone()   # [3, H, W]

    # Denormalize
    img = denormalize(img)

    # CHW -> HWC
    img = img.permute(1, 2, 0)

    # Clamp về [0,1] để imshow không clip
    img = img.clamp(0, 1)

    mask = mask.squeeze(0)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")

    plt.show()


