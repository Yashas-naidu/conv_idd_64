import torch
from torch import nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) loss for images in range [0, 1].
    Expects inputs shaped as [B, S, C, H, W] or [B, C, H, W].
    Returns mean(1 - SSIM).
    """

    def __init__(self, window_size: int = 11, channel: int = 3, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.C1 = C1
        self.C2 = C2
        self.register_buffer('gaussian_window', self._create_gaussian_window(window_size))

    def _create_gaussian_window(self, window_size: int) -> torch.Tensor:
        sigma = 1.5
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).unsqueeze(0)
        window_2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)  # [1,1,ws,ws]
        return window_2d

    def _ssim_2d(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x,y: [B, C, H, W] in [0,1]
        b, c, h, w = x.shape
        if c != self.channel:
            # support arbitrary channels by updating groups
            channel = c
        else:
            channel = self.channel

        window = self.gaussian_window.expand(channel, 1, self.window_size, self.window_size)

        padding = self.window_size // 2
        mu_x = F.conv2d(x, window, padding=padding, groups=channel)
        mu_y = F.conv2d(y, window, padding=padding, groups=channel)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, padding=padding, groups=channel) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=padding, groups=channel) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channel) - mu_xy

        C1 = self.C1
        C2 = self.C2

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12)
        return ssim_map.mean(dim=[1, 2, 3])  # per-sample SSIM

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Accept [B,S,C,H,W] or [B,C,H,W]
        if pred.dim() == 5:
            b, s, c, h, w = pred.shape
            pred_4d = pred.reshape(b * s, c, h, w)
            target_4d = target.reshape(b * s, c, h, w)
        elif pred.dim() == 4:
            pred_4d = pred
            target_4d = target
        else:
            raise ValueError(f"Unsupported tensor shape for SSIMLoss: {pred.shape}")

        pred_4d = pred_4d.clamp(0.0, 1.0)
        target_4d = target_4d.clamp(0.0, 1.0)

        ssim_per_sample = self._ssim_2d(pred_4d, target_4d)
        loss = 1.0 - ssim_per_sample
        return loss.mean()


