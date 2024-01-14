import torch
import torch.nn as nn
import torchvision.transforms as T
from timm.models.vision_transformer import vit_base_patch16_224

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from vit_pytorch import ViT


class VideoViT(nn.Module):
    def __init__(self):
        super(VideoViT, self).__init__()

        # ViT model
        self.vit = ViT(image_size=224, patch_size=16, num_classes=50, dim=256, depth=2, heads=4, mlp_dim=256)
        # Output layer
        self.output_layer = nn.Linear(1024, 50)

    def forward(self, x):
        batch_size, num_frames, channels, H, W = x.shape

        x = x.view(batch_size * num_frames, channels, H, W)

        x = self.vit(x)

        x = x.view(batch_size, num_frames, -1)

        return x
