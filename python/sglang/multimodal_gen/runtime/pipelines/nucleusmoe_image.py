# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DecodingStage,
    DenoisingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def prepare_mu(batch: Req, server_args: ServerArgs):
    height = batch.height
    width = batch.width
    vae_scale_factor = server_args.pipeline_config.vae_config.vae_scale_factor
    image_seq_len = (int(height) // vae_scale_factor // 2) * (
        int(width) // vae_scale_factor // 2
    )
    mu = calculate_shift(
        image_seq_len,
        256,
        4096,
        0.5,
        1.15,
    )
    return "mu", mu


class NucleusMoEImagePipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "NucleusMoEImagePipeline"

    _required_config_modules = [
        "text_encoder",
        "processor",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(InputValidationStage())

        # NucleusMoE uses a Qwen3VLProcessor as both processor and tokenizer.
        # The processor can tokenize text directly, so we use it in place of
        # a standalone tokenizer for the text encoding stage.
        self.add_stage(
            TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("processor")],
            ),
        )

        self.add_standard_latent_preparation_stage()
        self.add_standard_timestep_preparation_stage(
            prepare_extra_kwargs=[prepare_mu]
        )
        self.add_standard_denoising_stage()
        self.add_standard_decoding_stage()


EntryClass = NucleusMoEImagePipeline
