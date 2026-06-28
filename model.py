import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlockRGB(nn.Module):
    """
    A deep spatial residual block that fuses continuous multi-channel color maps
    with temporal time embeddings.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Time projection layer to inject temporal step variables into this block
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # Shortcut pathway to handle residual gradient flow
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            
    def forward(self, x, t_emb):
        h = F.silu(self.conv1(x))
        
        # Project and inject the time step context directly into the spatial features
        time_proj = self.time_mlp(t_emb).view(t_emb.size(0), -1, 1, 1)
        h = h + time_proj
        
        h = self.conv2(F.silu(h))
        return h + self.shortcut(x)

class PixelArtUNet(nn.Module):
    """
    3-Channel Generative U-Net Engine optimized for 32x32 RGB asset synthesis.
    """
    def __init__(self, time_emb_dim=128):
        super().__init__()
        
        # Sinusoidal time step projection block
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # ─── ENCODER PATHWAY ───
        self.init_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1) # Entry point: 3 RGB channels
        self.down1 = ResidualBlockRGB(64, 128, time_emb_dim)
        self.down2 = ResidualBlockRGB(128, 256, time_emb_dim)
        
        # ─── BOTTLENECK LAYER ───
        self.bottleneck = ResidualBlockRGB(256, 256, time_emb_dim)
        
        # ─── DECODER PATHWAY ───
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_block2 = ResidualBlockRGB(384, 128, time_emb_dim) #  Ensure this says 384

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_block1 = ResidualBlockRGB(192, 64, time_emb_dim)  #  Ensure this says 192
        
        # Final output layer mapping hidden tracking vectors back to raw RGB space
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def _get_time_embedding(self, timesteps, dim=128):
        """Generates continuous sinusoidal embedding vectors for time progression modeling."""
        half_dim = dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        args = timesteps.unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, x, t):
        # 1. Compute temporal embedding vector
        t_emb = self._get_time_embedding(t, dim=128)
        t_emb = self.time_mlp(t_emb)
        
        # 2. Forward execution through Encoder (saving intermediate skip connections)
        x1 = self.init_conv(x)              # Shape: [B, 64, 32, 32]
        x2 = self.down1(x1, t_emb)          # Shape: [B, 128, 32, 32]
        x2_down = F.max_pool2d(x2, 2)       # Shape: [B, 128, 16, 16]
        
        x3 = self.down2(x2_down, t_emb)     # Shape: [B, 256, 16, 16]
        x3_down = F.max_pool2d(x3, 2)       # Shape: [B, 256, 8, 8]
        
        # 3. Bottleneck Processing
        b = self.bottleneck(x3_down, t_emb) # Shape: [B, 256, 8, 8]
        
        # 4. Decoder Execution with Skip Connections
        u2 = self.up2(b)                    # Upsample back to [B, 128, 16, 16]
        u2_cat = torch.cat([u2, x3], dim=1) # Cat along channel axis -> [B, 256, 16, 16]
        u2_out = self.up_block2(u2_cat, t_emb)
        
        u1 = self.up1(u2_out)               # Upsample back to [B, 64, 32, 32]
        u1_cat = torch.cat([u1, x2], dim=1) # Cat along channel axis -> [B, 128, 32, 32]
        u1_out = self.up_block1(u1_cat, t_emb)
        
        # 5. Project to raw output RGB channels
        return self.final_conv(u1_out)      # Shape: [B, 3, 32, 32]
