import matplotlib.pyplot as plt
import torch


def save_image_batch(
    images: torch.Tensor,
    iter: int,
    out_dir: str = "training",
) -> None:
    rows = 2
    cols = 2
    batch_size: int = images.shape[0] ** 0.5
    max_size: int = min()
    fig, axs = plt.subplots(rows, cols, figsize=(10, 8))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(images[i].detach().cpu().permute(1, 2, 0).numpy(), cmap="gray")
        ax.axis("off")
    plt.savefig(f"training/iter_{iter}.png")
    plt.close()


def print_train_time(start: float, end: float, device: torch.device | None = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def _gradient_penalty(
    model: torch.nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device,
):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty
