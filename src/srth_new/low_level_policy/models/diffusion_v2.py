'''
use the DiffusionPolicy class from lerobot on the cholecystectomy data.
'''
import logging
from typing import List, Literal, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F
from torchvision import transforms

import torch.nn as nn

from srth_new.general.third_party.EndoSynth.endosynth.models import (
    load as load_depth_model,
)
from srth_new.general.utils.lang_encoding import (
    encode_text,
    initialize_model_and_tokenizer,
)
from srth_new.low_level_policy.dataset.img_aug import ImageAug
from srth_new.low_level_policy.models.detr.models.backbone import build_image_backbone
from srth_new.low_level_policy.models.detr.models.detr_vae_utils import build_encoder
from srth_new.low_level_policy.models.detr.models.transformer import build_transformer
from srth_new.low_level_policy.models.detr.models.detr_vae import DETRVAE

from diffusers import DDPMScheduler
from lerobot.configs.types import PolicyFeature
from lerobot.configs.types import FeatureType
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion import processor_diffusion

from srth_new.low_level_policy.models.dvrk_policy import DVRKPolicy

from types import SimpleNamespace

log = logging.getLogger(__name__)

class DiffusionPolicyV2(DVRKPolicy):
    def __init__(
            self,
            lr: float,
            weight_decay: float,
            camera_names: List[str],
            num_queries: int,
            history_chunk_size: int,
            history_num_tokens: int,
            history_num_layers: int,
            history_num_heads: int,
            action_dim: int,
            kl_weight: float,
            use_language: bool,
            language_encoder: str,
            action_mode: Literal["hybrid_relative", "ego", "relative_endoscope"],
            norm_scheme: Literal["std", "min_max"],
            img_resize_cfg: DictConfig,
            img_backbone_cfg: DictConfig,
            transformer_cfg: DictConfig,
            encoder_cfg: DictConfig,
            img_aug_cfg: DictConfig,
            use_depth: bool = True,
            use_history: bool = True,
    ):
        """Initialize the policy, optimizer, and optional conditioning modules."""
        super().__init__(
            action_dim=action_dim,
            action_mode=action_mode,
            norm_scheme=norm_scheme,
        )

        self.lr = lr
        self.weight_decay = weight_decay

        self.camera_names = camera_names
        self.num_queries = num_queries
        self.history_chunk_size = history_chunk_size

        # self.kl_weight = kl_weight
        self.state_dim = action_dim
        self.use_language = use_language
        self.language_encoder = language_encoder

        self.img_resize_cfg = img_resize_cfg

        self.use_depth = use_depth
        self.use_history = use_history
        self.history_chunk_size = history_chunk_size
        self.use_history = history_chunk_size > 0

        # Build image backbones.
        img_backbones = []
        for _ in range(len(camera_names)):
            img_backbone = build_image_backbone(**img_backbone_cfg)  # type: ignore
            img_backbones.append(img_backbone)
        self.backbones = nn.ModuleList(img_backbones)  # to get image features

        # Build image augmentation pipeline.
        self.img_aug_dict = self._build_img_aug_dict(img_aug_cfg)

        self.use_depth = use_depth
        self.use_history = use_history

        self.optimizer = torch.optim.AdamW(
            self._get_param_dict(self, img_backbone_cfg),  # to use backbones and input proj params
            lr=lr,
            weight_decay=weight_decay,
        )






