# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Tuple

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class NucleusMoEArchConfig(DiTArchConfig):
    patch_size: int = 2
    in_channels: int = 64
    out_channels: int | None = None
    num_layers: int = 24
    attention_head_dim: int = 128
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    joint_attention_dim: int = 3584
    axes_dims_rope: Tuple[int, int, int] = (16, 56, 56)
    mlp_ratio: float = 4.0

    moe_enabled: bool = True
    dense_moe_strategy: str = "leave_first_three_and_last_block_dense"
    num_experts: int = 128
    moe_intermediate_dim: int = 1344
    capacity_factor: float = 8.0
    use_sigmoid: bool = False
    route_scale: float = 2.5

    stacked_params_mapping: list[tuple[str, str, str]] = field(default_factory=list)

    param_names_mapping: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class NucleusMoEDitConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=NucleusMoEArchConfig)

    prefix: str = "nucleusmoe"
