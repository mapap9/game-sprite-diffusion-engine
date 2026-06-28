import torch
import matplotlib.pyplot as plt
from model import PixelArtUNet
from train_engine import NoiseScheduler

def generate_pixel_art_grid(checkpoint_path, num_samples=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔮 Initializing Reverse Sampler via Device: {device}")
    
    # 1. Initialize and load the trained model weights
    model = PixelArtUNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 2. Sync the noise scheduler to training parameters
    scheduler = NoiseScheduler(timesteps=300)
    betas = scheduler.betas.to(device)
    alphas = scheduler.alphas.to(device)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    
    # 3. Generate absolute white Gaussian noise to start the reverse loop [B, 3, 32, 32]
    x = torch.randn(num_samples, 3, 32, 32).to(device)
    
    # ─── REVERSE DDPM SAMPLING LOOP ───
    print("⏳ Denoising latent space blocks...")
    with torch.no_grad():
        for t in reversed(range(0, scheduler.timesteps)):
            # Formulate the timestep tensor for the current batch size
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # Predict the noise matrix currently present in the tensor
            pred_noise = model(x, t_tensor)
            
            # Extract scheduling coefficients for step t
            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]
            beta_t = betas[t]
            
            # Standard DDPM Reverse Step Formula:
            # Reconstruct the denoised image estimate at step t-1
            sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod_t)
            x_prev_mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / sqrt_one_minus_alpha_cumprod) * pred_noise)
            
            if t > 0:
                # Inject stochastic variance to keep the color boundaries dynamic
                noise = torch.randn_like(x).to(device)
                sigma_t = torch.sqrt(beta_t)
                x = x_prev_mean + sigma_t * noise
            else:
                x = x_prev_mean

    # ─── POST-PROCESS AND RENDER GRID ───
    # Map from symmetric [-1, 1] space back to raw continuous [0, 1] floats
    x = (x + 1.0) / 2.0
    x = torch.clamp(x, 0.0, 1.0)
    
    # Move to CPU and shift axes to standard image format (Batch, Height, Width, Channels)
    samples = x.cpu().permute(0, 2, 3, 1).numpy()
    
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    fig.patch.set_facecolor('#111111') # Dark theme backdrop to emphasize pixel art edges
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i])
        ax.axis('off')
        
    plt.tight_layout()
    output_name = f"generated_{checkpoint_path.replace('.pt', '.png')}"
    plt.savefig(output_name, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"🎨 Grid generation complete! Saved visual artifact to: {output_name}")

if __name__ == "__main__":
    # Test your latest milestone checkpoint
    generate_pixel_art_grid("pixel_art_unet_epoch_1000.pt")
