import torch
import torch.nn as nn
import math

class NoiseSource(nn.Module):
    """Base class for noise generation."""
    def forward(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

class GaussianNoise(NoiseSource):
    def __init__(self, **kwargs):  # 接受 Hydra 配置中的额外字段（如 name）
        super().__init__()
    
    def forward(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        return torch.randn(shape, device=device)

class UniformNoise(NoiseSource):
    def __init__(self, **kwargs):  # 接受 Hydra 配置中的额外字段（如 name）
        super().__init__()
    
    def forward(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        return torch.rand(shape, device=device) - 0.5

class AlphaStableNoise(NoiseSource):
    """
    Alpha-stable noise generator using Chambers-Mallows-Stuck method.
    Currently supports symmetric alpha-stable distributions (beta=0).
    """
    def __init__(self, alpha=1.5, beta=0.0, **kwargs):  # 接受 Hydra 配置中的额外字段（如 name）
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        # Generate random variables
        # U ~ Uniform(-pi/2, pi/2)
        u = (torch.rand(shape, device=device) - 0.5) * math.pi
        # W ~ Exponential(1)
        w = -torch.log(torch.rand(shape, device=device) + 1e-8) # Add epsilon to avoid log(0)

        if self.alpha == 1:
            # Cauchy distribution
            return torch.tan(u)
        
        # Symmetric alpha-stable (beta=0)
        # X = (sin(alpha*U) / (cos(U))^(1/alpha)) * (cos((1-alpha)*U) / W)^((1-alpha)/alpha)
        
        # Note: For general (alpha, beta), the formula is more complex.
        # Assuming beta=0 for simplicity as it's common in signal processing.
        
        term1 = torch.sin(self.alpha * u) / (torch.pow(torch.cos(u), 1.0 / self.alpha) + 1e-8)
        term2 = torch.pow(torch.cos((1.0 - self.alpha) * u) / w, (1.0 - self.alpha) / self.alpha)
        
        return term1 * term2


class PoissonNoise(NoiseSource):
    """
    Poisson noise generator.
    Generates noise based on Poisson distribution, normalized to zero mean.
    """
    def __init__(self, lam=1.0, **kwargs):  # 接受 Hydra 配置中的额外字段（如 name）
        super().__init__()
        self.lam = lam  # Rate parameter (lambda) of Poisson distribution

    def forward(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        # Generate Poisson samples and normalize to zero mean, unit variance
        poisson_samples = torch.poisson(torch.full(shape, self.lam, device=device))
        # Normalize: (X - mean) / std, where mean = lam, std = sqrt(lam)
        normalized = (poisson_samples - self.lam) / (math.sqrt(self.lam) + 1e-8)
        return normalized


class ColoredNoise(NoiseSource):
    """
    Colored noise generator (1/f^beta noise).
    - beta=0: White noise
    - beta=1: Pink noise (1/f)
    - beta=2: Red/Brown noise (1/f^2)
    
    Uses spectral shaping in frequency domain.
    """
    def __init__(self, beta=1.0, **kwargs):  # 接受 Hydra 配置中的额外字段（如 name）
        super().__init__()
        self.beta = beta  # Spectral exponent

    def forward(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        # For 3D input (batch, channels, time), we apply to time dimension
        # For other shapes, apply to last dimension
        
        # Generate white noise in frequency domain
        n_samples = shape[-1]
        white_noise = torch.randn(shape, device=device)
        
        # Transform to frequency domain
        fft = torch.fft.rfft(white_noise, dim=-1)
        
        # Create frequency scaling (1/f^(beta/2) for amplitude)
        n_freqs = fft.shape[-1]
        freqs = torch.arange(n_freqs, device=device, dtype=torch.float32)
        freqs[0] = 1e-8  # Avoid division by zero at DC
        
        # Scale amplitude by 1/f^(beta/2) (power spectrum scales as 1/f^beta)
        scaling = 1.0 / (freqs ** (self.beta / 2.0))
        scaling = scaling / scaling.max()  # Normalize
        
        # Apply scaling
        fft_scaled = fft * scaling
        
        # Transform back to time domain
        colored = torch.fft.irfft(fft_scaled, n=n_samples, dim=-1)
        
        # Normalize to unit variance
        std = colored.std(dim=-1, keepdim=True) + 1e-8
        colored = colored / std
        
        return colored

class SRLayer(nn.Module):
    def __init__(self, noise_source: NoiseSource, intensity: float = 1.0, **kwargs):  # 接受额外参数
        super().__init__()
        self.noise_source = noise_source
        self.intensity = intensity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class AdditiveSR(SRLayer):
    """
    Additive Stochastic Resonance: Output = Input + D * Noise
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = self.noise_source(x.shape, x.device)
        return x + self.intensity * noise

class BistableSR(SRLayer):
    """
    Bistable Stochastic Resonance using Runge-Kutta method for SDE.
    System: dx/dt = ax - bx^3 + Input + Noise
    """
    def __init__(self, noise_source: NoiseSource, intensity: float = 1.0, a=1.0, b=1.0, dt=0.01, **kwargs):
        super().__init__(noise_source, intensity, **kwargs)
        self.a = a
        self.b = b
        self.dt = dt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Channels, Time)
        # We integrate over the Time dimension.
        
        batch_size, channels, time_steps = x.shape
        dt = self.dt
        sqrt_dt = math.sqrt(dt)
        
        # Pre-generate noise for the entire sequence
        # Noise term in SDE: intensity * dW, where dW ~ N(0, dt) -> N(0,1) * sqrt(dt)
        # We assume noise_source returns standard distribution samples (like N(0,1))
        noise = self.noise_source(x.shape, x.device)
        
        # Initial state
        state = torch.zeros((batch_size, channels), device=x.device)
        outputs = []
        
        # Runge-Kutta 4 integration for drift, Euler-Maruyama for diffusion
        for t in range(time_steps):
            inp = x[..., t]
            n = noise[..., t]
            
            # Drift function f(y) = ax - bx^3 + Input
            def f(y, i_val):
                return self.a * y - self.b * (y ** 3) + i_val
            
            k1 = f(state, inp)
            k2 = f(state + 0.5 * dt * k1, inp)
            k3 = f(state + 0.5 * dt * k2, inp)
            k4 = f(state + dt * k3, inp)
            
            # Update state
            # Deterministic part: (dt/6) * (k1 + 2k2 + 2k3 + k4)
            # Stochastic part: intensity * noise * sqrt(dt)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4) + self.intensity * n * sqrt_dt
            
            outputs.append(state)
            
        return torch.stack(outputs, dim=-1)

class TristableSR(SRLayer):
    """
    Tristable Stochastic Resonance using Runge-Kutta method for SDE.
    Potential: U(x) = (a/2)x^2 - (b/4)x^4 + (c/6)x^6
    System: dx/dt = -U'(x) + Input + Noise
            dx/dt = -ax + bx^3 - cx^5 + Input + Noise
    """
    def __init__(self, noise_source: NoiseSource, intensity: float = 1.0, a=1.0, b=1.0, c=1.0, dt=0.01, **kwargs):
        super().__init__(noise_source, intensity, **kwargs)
        self.a = a
        self.b = b
        self.c = c
        self.dt = dt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, time_steps = x.shape
        dt = self.dt
        sqrt_dt = math.sqrt(dt)
        
        noise = self.noise_source(x.shape, x.device)
        state = torch.zeros((batch_size, channels), device=x.device)
        outputs = []
        
        for t in range(time_steps):
            inp = x[..., t]
            n = noise[..., t]
            
            # Drift function f(y) = -ax + bx^3 - cx^5 + Input
            def f(y, i_val):
                return -self.a * y + self.b * (y ** 3) - self.c * (y ** 5) + i_val
            
            k1 = f(state, inp)
            k2 = f(state + 0.5 * dt * k1, inp)
            k3 = f(state + 0.5 * dt * k2, inp)
            k4 = f(state + dt * k3, inp)
            
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4) + self.intensity * n * sqrt_dt
            
            outputs.append(state)
            
        return torch.stack(outputs, dim=-1)