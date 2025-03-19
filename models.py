from typing import Tuple
from torch import nn
from torch.nn import functional as F
from utils import PositionEmbed


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super().__init__()
        output_padding = 0 if stride == 1 else 1
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size,
                        stride=stride, padding=kernel_size//2, output_padding=output_padding),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class Encoder(nn.Module):
    def __init__(self, channels: Tuple[int, ...], strides: Tuple[int, ...], kernel_size):
        super().__init__()
        modules = []
        channel = 3
        for ch, s in zip(channels, strides):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(channel, ch, kernel_size, stride=s, padding=kernel_size//2),
                    nn.ReLU(),
                )
            )
            channel = ch
        self.conv = nn.Sequential(*modules)
        
    def forward(self, x):
        """
        input:
            x: input image, [B, 3, H, W]
        output:
            feature_map: [B, C, H_enc, W_enc]
        """
        x = self.conv(x)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        resolution: Tuple[int, int],
        init_resolution: Tuple[int, int],
        slot_size: int,
        kernel_size: int,
        channels: Tuple[int, ...],
        strides: Tuple[int, ...],
    ):
        super().__init__()
        self.resolution = resolution
        self.init_resolution = init_resolution

        self.pos_emb = PositionEmbed(slot_size, init_resolution)

        modules = []
        channel = slot_size
        for ch, s in zip(channels, strides):
            modules.append(
                nn.Sequential(
                    ConvTransposeBlock(
                    in_ch=channel, out_ch=ch, kernel_size=kernel_size, stride=s),
                )
            )
            channel = ch
        modules.append(
            nn.Sequential(
                ConvTransposeBlock(
                    in_ch=channel, out_ch=channel, kernel_size=kernel_size, stride=1),
                nn.ConvTranspose2d(
                    channel, 4, kernel_size=3, stride=1, padding=1)
                )
            )
        
        self.conv = nn.Sequential(*modules)

        
    def forward(self, slots):
        B, K, D = slots.shape
        slots = self.broadcast(slots)
        slots = self.pos_emb(slots)
        out = self.conv(slots)
        masks_logits = out[:, 3:].reshape(B, K, 1, self.resolution[0], self.resolution[1])
        masks = F.softmax(masks_logits, dim=1)
        recons = out[:, :3].reshape(B, K, 3, self.resolution[0], self.resolution[1])
        return masks, recons
    
    def broadcast(self, slots):
        B, K, D = slots.shape
        H, W = self.init_resolution
        return slots.reshape(B*K, D, 1, 1).repeat(1, 1, H, W)