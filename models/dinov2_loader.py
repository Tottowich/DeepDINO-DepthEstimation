import urllib.request
from functools import partial
from typing import Optional, Union

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from dinov2.eval.depth.models import build_depther
from mmcv.runner import load_checkpoint

import math
import itertools

class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output

class DinoV2Loader:
    """
    A class to load DinoV2 backbones and associated models for depth estimation.
    
    Attributes:
    -----------
    backbone_size: str
        The size of the backbone, can be "small", "base", "large", or "giant".
    backbone_model: nn.Module
        The loaded backbone model.
    head_type: str
        The type of the head, can be "linear", "linear4", or "dpt".
    head_dataset: str
        The dataset for the head, can be "nyu" or "kitti".
    model: nn.Module
        The loaded DinoV2 model for depth estimation.
    """
    def __init__(self, 
                 backbone_size: str = "small", 
                 head_type: str = "dpt", 
                 head_dataset: str = "nyu") -> None:
        """
        Initialize the DinoV2Loader object.
        
        Parameters:
        -----------
        backbone_size: str, default="small"
            The size of the backbone to load. Can be one of "small", "base", "large", or "giant".
        head_type: str, default="dpt"
            The type of the head to use. Can be one of "linear", "linear4", or "dpt".
        head_dataset: str, default="nyu"
            The dataset for the head. Can be one of "nyu" or "kitti".
        """
        self.backbone_size = backbone_size.lower()
        self.head_type = head_type
        self.head_dataset = head_dataset

        self.head_checkpoint_url: Optional[str] = None
        self.head_config_url: Optional[str] = None

    def load_model(self, device=torch.device("cuda:0")) -> nn.Module:

        self.backbone_model = self._load_backbone().to(device)
        self.model = self._load_model()
        load_checkpoint(self.model, self.head_checkpoint_url, map_location="cpu")
        self.model.eval()
        self.model.to(device)
        return self.model
        
    def _load_backbone(self) -> nn.Module:
        """
        Load the DinoV2 backbone model.
        
        Returns:
        --------
        nn.Module
            The loaded backbone model.
        """
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        self.backbone_arch = backbone_archs[self.backbone_size]
        backbone_name = f"dinov2_{self.backbone_arch}"
        return torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name).eval()
    
    def _load_model(self) -> nn.Module:
        """
        Load the DinoV2 model for depth estimation.
        
        Returns:
        --------
        nn.Module
            The loaded DinoV2 model.
        """
        # Load config and create depther model
        cfg_str = self._load_config_from_url()
        cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
        depther = build_depther(cfg.model)
        
        # Customize the backbone forward function
        depther.backbone.forward = partial(
            self.backbone_model.get_intermediate_layers,
            n=cfg.model.backbone.out_indices,
            reshape=True,
            return_class_token=cfg.model.backbone.output_cls_token,
            norm=cfg.model.backbone.final_norm
        )
        if hasattr(self.backbone_model, "patch_size"):
            depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(self.backbone_model.patch_size)(x[0]))

        return depther

    def _load_config_from_url(self) -> str:
        """
        Load the configuration string from a URL.
        
        Returns:
        --------
        str
            The loaded configuration string.
        """
        # Define URL
        DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
        backbone_name = f"dinov2_{self.backbone_arch}"
        self.head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{self.head_dataset}_{self.head_type}_config.py"
        self.head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{self.head_dataset}_{self.head_type}_head.pth"

        # Fetch content
        with urllib.request.urlopen(self.head_config_url) as f:
            return f.read().decode()