#NOTE: Work in progress

from typing import Tuple, Union
import torch
import torch.nn as nn
from torchvision.models import resnet34
from transformers import CLIPModel, CLIPProcessor

class VisionEncoder(nn.Module):
    """Base class for vision encoders"""
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class ResNetEncoder(VisionEncoder):
    """ResNet34 encoder implementation"""
    def __init__(self, latent_dim: int = 512):
        super().__init__(latent_dim)
        # Load pretrained ResNet34
        resnet = resnet34(pretrained=True)
        # Remove the final classification layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        # Add projection if needed
        self.projection = (
            nn.Linear(512, latent_dim) if latent_dim != 512 else nn.Identity()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return self.projection(x)

class CLIPEncoder(VisionEncoder):
    """CLIP encoder implementation"""
    def __init__(self, latent_dim: int = 512, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__(latent_dim)
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.projection = (
            nn.Linear(self.model.config.vision_config.hidden_size, latent_dim) 
            if latent_dim != self.model.config.vision_config.hidden_size 
            else nn.Identity()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process images using CLIP's processor
        inputs = self.processor(images=x, return_tensors="pt")
        features = self.model.get_image_features(**inputs)
        return self.projection(features)

def create_encoder(
    encoder_type: str,
    latent_dim: int = 512,
    **kwargs
) -> VisionEncoder:
    """Factory function to create vision encoders"""
    encoders = {
        "resnet34": ResNetEncoder,
        "clip": CLIPEncoder
    }
    
    if encoder_type not in encoders:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
        
    return encoders[encoder_type](latent_dim=latent_dim, **kwargs) 