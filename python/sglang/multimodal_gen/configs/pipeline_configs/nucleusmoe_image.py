# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Callable

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.nucleusmoe import NucleusMoEDitConfig
from sglang.multimodal_gen.configs.models.encoders.qwen3 import Qwen3TextConfig
from sglang.multimodal_gen.configs.models.vaes.qwenimage import QwenImageVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
    shard_rotary_emb_for_sp,
)


DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant designed to generate photorealistic, "
    "ultra-high-quality images based on user prompts."
)

HIDDEN_STATE_RETURN_INDEX = -8


def nucleusmoe_preprocess_text(prompt):
    template = (
        "<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return template.format(system=DEFAULT_SYSTEM_PROMPT, user=prompt)


def nucleusmoe_postprocess_text(outputs, _text_inputs):
    hidden_states = outputs.hidden_states[HIDDEN_STATE_RETURN_INDEX]
    return hidden_states


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels_latents * 4
    )
    return latents


@dataclass
class NucleusMoEImagePipelineConfig(ImagePipelineConfig):
    """Configuration for the NucleusMoE-Image pipeline."""

    should_use_guidance: bool = False
    task_type: ModelTaskType = ModelTaskType.T2I

    vae_tiling: bool = False
    vae_sp: bool = False

    dit_config: DiTConfig = field(default_factory=NucleusMoEDitConfig)
    vae_config: VAEConfig = field(default_factory=QwenImageVAEConfig)

    enable_autocast: bool = False

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Qwen3TextConfig(),)
    )

    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))

    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (nucleusmoe_preprocess_text,)
    )

    postprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (nucleusmoe_postprocess_text,)
    )

    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [
            dict(
                padding=True,
                truncation=True,
            ),
            None,
        ]
    )

    def prepare_sigmas(self, sigmas, num_inference_steps):
        return self._prepare_sigmas(sigmas, num_inference_steps)

    def get_vae_scale_factor(self):
        return self.vae_config.arch_config.vae_scale_factor

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor
        height = 2 * (batch.height // (vae_scale_factor * 2))
        width = 2 * (batch.width // (vae_scale_factor * 2))
        num_channels_latents = self.dit_config.arch_config.in_channels // 4
        shape = (batch_size, 1, num_channels_latents, height, width)
        return shape

    def maybe_pack_latents(self, latents, batch_size, batch):
        height = 2 * (
            batch.height // (self.vae_config.arch_config.vae_scale_factor * 2)
        )
        width = 2 * (batch.width // (self.vae_config.arch_config.vae_scale_factor * 2))
        num_channels_latents = self.dit_config.arch_config.in_channels // 4
        return _pack_latents(latents, batch_size, num_channels_latents, height, width)

    def get_decode_scale_and_shift(self, device, dtype, vae):
        vae_arch_config = self.vae_config.arch_config
        scaling_factor = 1.0 / torch.tensor(
            vae_arch_config.latents_std, device=device
        ).view(1, vae_arch_config.z_dim, 1, 1, 1).to(device, dtype)
        shift_factor = (
            torch.tensor(vae_arch_config.latents_mean)
            .view(1, vae_arch_config.z_dim, 1, 1, 1)
            .to(device, dtype)
        )
        return scaling_factor, shift_factor

    @staticmethod
    def get_freqs_cis(img_shapes, txt_seq_lens, rotary_emb, device, dtype):
        img_freqs, txt_freqs = rotary_emb(img_shapes, txt_seq_lens, device=device)

        img_cos_half = img_freqs.real.to(dtype=torch.float32).contiguous()
        img_sin_half = img_freqs.imag.to(dtype=torch.float32).contiguous()
        txt_cos_half = txt_freqs.real.to(dtype=torch.float32).contiguous()
        txt_sin_half = txt_freqs.imag.to(dtype=torch.float32).contiguous()

        img_cos_sin_cache = torch.cat([img_cos_half, img_sin_half], dim=-1)
        txt_cos_sin_cache = torch.cat([txt_cos_half, txt_sin_half], dim=-1)
        return img_cos_sin_cache, txt_cos_sin_cache

    def _prepare_cond_kwargs(self, batch, prompt_embeds, rotary_emb, device, dtype):
        batch_size = prompt_embeds[0].shape[0]
        height = batch.height
        width = batch.width
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor

        img_shapes = [
            [
                (
                    1,
                    height // vae_scale_factor // 2,
                    width // vae_scale_factor // 2,
                )
            ]
        ] * batch_size
        txt_seq_lens = [prompt_embeds[0].shape[1]]

        if rotary_emb is None:
            return {
                "img_shapes": img_shapes,
                "txt_seq_lens": txt_seq_lens,
                "freqs_cis": None,
            }

        freqs_cis = self.get_freqs_cis(
            img_shapes, txt_seq_lens, rotary_emb, device, dtype
        )

        img_cache, txt_cache = freqs_cis
        img_cache = shard_rotary_emb_for_sp(img_cache)
        return {
            "txt_seq_lens": txt_seq_lens,
            "freqs_cis": (img_cache, txt_cache),
            "img_shapes": img_shapes,
        }

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._prepare_cond_kwargs(
            batch, batch.prompt_embeds, rotary_emb, device, dtype
        )

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._prepare_cond_kwargs(
            batch, batch.negative_prompt_embeds, rotary_emb, device, dtype
        )

    def post_denoising_loop(self, latents, batch):
        (
            latents,
            batch_size,
            channels,
            height,
            width,
        ) = self._unpad_and_unpack_latents(latents, batch)
        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)
        return latents
