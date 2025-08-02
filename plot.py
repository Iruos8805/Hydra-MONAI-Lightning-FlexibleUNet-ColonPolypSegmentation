import matplotlib.pyplot as plt
import torch


def visualize_batch(inputs, preds, targets, num_samples=5):
    """
    Visualizes a batch of predictions alongside inputs and targets.

    Args:
        inputs (Tensor): Shape (N, C, H, W)
        preds (Tensor): Shape (N, 1, H, W)
        targets (Tensor): Shape (N, 1, H, W)
        num_samples (int): Number of samples to visualize
    """
    # Make sure everything is in CPU and numpy
    inputs = inputs[:num_samples]
    preds = preds[:num_samples]
    targets = targets[:num_samples]

    inputs = inputs.numpy()
    preds = preds.numpy()
    targets = targets.numpy()

    for i in range(num_samples):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # Input image: assume it's grayscale or RGB
        img = inputs[i]
        if img.shape[0] == 1:  # single channel
            axs[0].imshow(img[0], cmap="gray")
        else:
            axs[0].imshow(img.transpose(1, 2, 0))
        axs[0].set_title("Input")

        axs[1].imshow(targets[i][0], cmap="gray")
        axs[1].set_title("Ground Truth")

        axs[2].imshow(preds[i][0], cmap="gray")
        axs[2].set_title("Prediction")

        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
