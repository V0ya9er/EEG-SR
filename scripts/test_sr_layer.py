import torch
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modules.sr_layer import (
    GaussianNoise, UniformNoise, AlphaStableNoise,
    AdditiveSR, BistableSR, TristableSR
)

def test_sr_layer():
    print("Testing SR Layer...")
    
    # Create dummy input: (Batch, Channels, Time)
    batch_size = 2
    channels = 3
    time_steps = 100
    x = torch.sin(torch.linspace(0, 10, time_steps)).unsqueeze(0).unsqueeze(0).repeat(batch_size, channels, 1)
    
    print(f"Input shape: {x.shape}")
    
    # Test Noise Sources
    print("\n--- Testing Noise Sources ---")
    device = torch.device('cpu')
    shape = x.shape
    
    gaussian = GaussianNoise()
    noise_g = gaussian(shape, device)
    print(f"Gaussian Noise shape: {noise_g.shape}, Mean: {noise_g.mean():.4f}, Std: {noise_g.std():.4f}")
    
    uniform = UniformNoise()
    noise_u = uniform(shape, device)
    print(f"Uniform Noise shape: {noise_u.shape}, Mean: {noise_u.mean():.4f}, Range: [{noise_u.min():.4f}, {noise_u.max():.4f}]")
    
    alpha = AlphaStableNoise(alpha=1.5)
    noise_a = alpha(shape, device)
    print(f"Alpha Stable Noise shape: {noise_a.shape}, Mean: {noise_a.mean():.4f}")

    # Test SR Mechanisms
    print("\n--- Testing SR Mechanisms ---")
    
    # Additive SR
    sr_add = AdditiveSR(noise_source=gaussian, intensity=0.5)
    out_add = sr_add(x)
    print(f"Additive SR Output shape: {out_add.shape}")
    assert out_add.shape == x.shape
    
    # Bistable SR
    sr_bi = BistableSR(noise_source=gaussian, intensity=0.5, dt=0.01)
    out_bi = sr_bi(x)
    print(f"Bistable SR Output shape: {out_bi.shape}")
    assert out_bi.shape == x.shape
    
    # Tristable SR
    sr_tri = TristableSR(noise_source=gaussian, intensity=0.5, dt=0.01)
    out_tri = sr_tri(x)
    print(f"Tristable SR Output shape: {out_tri.shape}")
    assert out_tri.shape == x.shape
    
    print("\nAll SR Layer tests passed!")

if __name__ == "__main__":
    test_sr_layer()