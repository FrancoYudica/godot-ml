import argparse
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image


# Transformations


def apply_warm_tone(pixels: torch.Tensor) -> torch.Tensor:
    warm = pixels.clone()
    warm[:, 0] = torch.clamp(pixels[:, 0] * 1.2, 0, 1)
    warm[:, 1] = torch.clamp(pixels[:, 1] * 1.05, 0, 1)
    warm[:, 2] = torch.clamp(pixels[:, 2] * 0.7, 0, 1)
    return warm


def apply_grayscale(pixels: torch.Tensor) -> torch.Tensor:
    luma = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
    return luma.unsqueeze(1).repeat(1, 3)


def apply_sepia(pixels: torch.Tensor) -> torch.Tensor:
    r = torch.clamp(
        pixels[:, 0] * 0.393 + pixels[:, 1] * 0.769 + pixels[:, 2] * 0.189, 0, 1
    )
    g = torch.clamp(
        pixels[:, 0] * 0.349 + pixels[:, 1] * 0.686 + pixels[:, 2] * 0.168, 0, 1
    )
    b = torch.clamp(
        pixels[:, 0] * 0.272 + pixels[:, 1] * 0.534 + pixels[:, 2] * 0.131, 0, 1
    )
    return torch.stack([r, g, b], dim=1)


def apply_reinhard(pixels: torch.Tensor) -> torch.Tensor:
    return pixels / (1 + pixels)


def apply_invert(pixels: torch.Tensor) -> torch.Tensor:
    return 1.0 - pixels


def apply_gamma(pixels: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    return torch.clamp(pixels ** (1.0 / gamma), 0, 1)


TRANSFORMATIONS: dict[str, Callable] = {
    "warm_tone": apply_warm_tone,
    "grayscale": apply_grayscale,
    "sepia": apply_sepia,
    "reinhard": apply_reinhard,
    "invert": apply_invert,
    "gamma": apply_gamma,
}


class ColorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Data
# Training uses [N, 3, 1, 1] tensors so Conv2d receives the right shape.
# Transforms still operate on [N, 3] pixels via squeeze/unsqueeze.


def make_dataset(
    transform: Callable, num_samples: int
) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = torch.rand(num_samples, 3)
    targets = transform(pixels)
    return pixels.unsqueeze(-1).unsqueeze(-1), targets.unsqueeze(-1).unsqueeze(-1)


# Training


def train(
    transform: Callable,
    epochs: int,
    num_samples: int,
    batch_size: int,
    lr: float,
) -> ColorMLP:
    pixels, targets = make_dataset(transform, num_samples)

    model = ColorMLP()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pixels, targets),
        batch_size=batch_size,
        shuffle=True,
    )

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_pixels, batch_targets in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_pixels), batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} - avg loss: {total_loss / len(loader):.6f}")

    return model


# Preview


def save_comparison(
    model: ColorMLP, transform: Callable, image_path: str, out_path: str
) -> None:
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    pixels_np = np.array(img).astype(np.float32) / 255.0

    x = torch.from_numpy(pixels_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    with torch.no_grad():
        model_out = (model(x).squeeze(0).permute(1, 2, 0).numpy() * 255).astype(
            np.uint8
        )

    pixels_flat = torch.from_numpy(pixels_np).reshape(-1, 3)
    analytical = (
        (transform(pixels_flat).numpy() * 255).astype(np.uint8).reshape(h, w, 3)
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.imshow(np.array(img))
    ax1.set_title("Original")
    ax1.axis("off")
    ax2.imshow(analytical)
    ax2.set_title("Analytical")
    ax2.axis("off")
    ax3.imshow(model_out)
    ax3.set_title("Model")
    ax3.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Preview saved to {out_path}")


# ONNX Export


def export_onnx(model: ColorMLP, out_path: str) -> None:
    model.eval()
    dummy = torch.zeros(1, 3, 64, 64)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["pixels"],
        output_names=["result"],
        dynamic_axes={
            "pixels": {0: "batch", 2: "height", 3: "width"},
            "result": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=11,
        dynamo=False,
    )
    print(f"Model exported to {out_path}")


# Entry point


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a color-transform MLP and export to ONNX"
    )
    p.add_argument(
        "--transform",
        choices=TRANSFORMATIONS.keys(),
        default="invert",
        help="Color transformation to approximate (default: invert)",
    )
    p.add_argument(
        "--epochs", type=int, default=100, help="Training epochs (default: 400)"
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=100_000,
        help="Random training pixels (default: 100 000)",
    )
    p.add_argument(
        "--batch-size", type=int, default=1024, help="Batch size (default: 1024)"
    )
    p.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)"
    )
    p.add_argument(
        "--output",
        default="",
        help="Output ONNX path (default: color_mlp_<transform>.onnx)",
    )
    p.add_argument(
        "--preview",
        metavar="IMAGE",
        help="Path to an image; saves a side-by-side comparison PNG next to the ONNX file",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    transform = TRANSFORMATIONS[args.transform]

    print(f"Training ColorMLP  transform={args.transform}  epochs={args.epochs}")
    model = train(
        transform,
        epochs=args.epochs,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    export_filename = (
        args.output if len(args.output) > 0 else f"color_mlp_{args.transform}.onnx"
    )

    export_onnx(model, export_filename)

    if args.preview:
        preview_path = export_filename.replace(".onnx", "_preview.png")
        save_comparison(model, transform, args.preview, preview_path)


if __name__ == "__main__":
    main()
