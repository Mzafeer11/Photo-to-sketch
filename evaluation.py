import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def tensor_to_image(tensor):
    image = tensor.cpu().detach().numpy().transpose(1, 2, 0)
    image = (image * 0.5 + 0.5) * 255
    return image.astype(np.uint8)

def calculate_metrics(real, fake):
    real_img = tensor_to_image(real)
    fake_img = tensor_to_image(fake)
    
    # SSIM and PSNR
    s = ssim(real_img, fake_img, multichannel=True, data_range=255)
    p = psnr(real_img, fake_img, data_range=255)
    
    return s, p

def visualize_results(real_A, fake_B, rec_A, real_B, fake_A, rec_B, n_samples=5):
    fig, axes = plt.subplots(6, n_samples, figsize=(15, 18))
    
    titles = ["Input Sketch", "Generated Photo", "Reconstructed Sketch", 
              "Input Photo", "Generated Sketch", "Reconstructed Photo"]
    
    for i in range(n_samples):
        imgs = [real_A[i], fake_B[i], rec_A[i], real_B[i], fake_A[i], rec_B[i]]
        for j, img in enumerate(imgs):
            axes[j, i].imshow(tensor_to_image(img))
            axes[j, i].axis("off")
            if i == 0:
                axes[j, i].set_ylabel(titles[j], size='large')

    plt.tight_layout()
    plt.show()

def plot_loss_curves(gen_losses, disc_losses, cycle_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Training Loss Curves")
    plt.plot(gen_losses, label="Generator Loss")
    plt.plot(disc_losses, label="Discriminator Loss")
    plt.plot(cycle_losses, label="Cycle Consistency Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
