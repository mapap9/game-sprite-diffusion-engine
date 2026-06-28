import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from model import PixelArtUNet

# ─── CUSTOM PIXEL ART DATASET CLASS ───
class PixelArtDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.png')]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.file_list)  # ✅ Fixed: Returns the total integer count

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.file_list[idx])
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)

# ─── NOISE SCHEDULE PRIMITIVES (DDPM) ───
class NoiseScheduler:
    def __init__(self, timesteps=300, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        # Linear variance schedule matching standard diffusion baselines
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_start, t, device):
        """Injects Gaussian noise corresponding to step t into the clean RGB tensor x_start."""
        noise = torch.randn_like(x_start).to(device)

        # Move the schedule array to the active device before indexing
        alphas_cumprod_dev = self.alphas_cumprod.to(device)

        # Gather coefficients cleanly on the correct device
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod_dev[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod_dev[t]).view(-1, 1, 1, 1)

        # Forward process equation
        x_noisy = sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
        return x_noisy, noise

# ─── MAIN TRAINING EXECUTION ───
def run_training():
    # 1. Hardware Profiling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Launching Pixel Art Diffusion Engine on target device: {device}")

    # 2. Pipeline Initialization
    dataset = PixelArtDataset(folder_path="data/processed")
    # Using single-threaded worker configuration to prevent Python 3.14 spawn deadlocks
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=0)
    
    model = PixelArtUNet().to(device)
    scheduler = NoiseScheduler(timesteps=300)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()

    epochs = 1000 # Color synthesis requires higher epoch depths to crystalize gradients
    print(f"📦 Staged {len(dataset)} unique RGB assets. Executing {epochs} optimization steps...")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch in progress_bar:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Sample random timesteps uniformly across the batch size
            t = torch.randint(0, scheduler.timesteps, (batch.size(0),), device=device).long()
            
            # Formulate the noisy spatial representations
            x_noisy, true_noise = scheduler.add_noise(batch, t, device)
            
            # Predict the noise matrix via the multi-stage UNet
            pred_noise = model(x_noisy, t)
            
            # Evaluate parametric vector MSE loss
            loss = criterion(pred_noise, true_noise)
            loss.backward()
            
            # Gradient clipping to maintain network stability across deep channels
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Checkpoint persistence layer
        if epoch % 10 == 0 or epoch == epochs:
            checkpoint_path = f"pixel_art_unet_epoch_{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Checkpoint secured: {checkpoint_path} | Avg Loss: {epoch_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    run_training()
