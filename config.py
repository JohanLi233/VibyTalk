from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class NetworkConfiguration:
    architecture_name: str
    layer_channels: List[int]
    input_resolution: int

    spectral_bins: int
    acoustic_vector_size: int
    acoustic_group_count: int
    temporal_window_length: int

    transformer_depth: int
    attention_head_count: int

    learning_rate: float
    training_batch_size: int

    PROBABILISTIC_MASKING: float = 0.5
    MASK_DIMENSION_RANGE: Tuple[float, float] = (0.1, 0.9)
    audio_learning_rate: Optional[float] = None
    adaptation_epoch_threshold: int = -1
    expansion_factor: float = 2.0
    acoustic_processing_mode: str = "mel"

    audio_conv_kernel_size: int = 5
    audio_expansion_coefficient: float = 2.0
    feedforward_scaling: float = 2.0

    convolution_stack_depth: int = 3
    conv_channel_progression: Optional[List[int]] = field(default=None)
    pretrained_encoder_path: Optional[str] = field(default=None)


MINIMAL_NETWORK = NetworkConfiguration(
    architecture_name="nano",
    layer_channels=[4, 4, 8, 16, 32],
    input_resolution=96,
    spectral_bins=80,
    acoustic_vector_size=80,
    acoustic_group_count=4,
    temporal_window_length=32,
    transformer_depth=4,
    attention_head_count=2,
    learning_rate=5e-4,
    training_batch_size=32,
    audio_learning_rate=5e-4,
    expansion_factor=1.25,
    audio_conv_kernel_size=5,
    audio_expansion_coefficient=2.0,
    feedforward_scaling=4.0,
    convolution_stack_depth=4,
    conv_channel_progression=[16, 24, 32, 40],
    adaptation_epoch_threshold=0,
)

COMPACT_NETWORK = NetworkConfiguration(
    architecture_name="tiny",
    layer_channels=[8, 8, 16, 32, 64],
    input_resolution=96,
    spectral_bins=80,
    acoustic_vector_size=80,
    acoustic_group_count=4,
    temporal_window_length=32,
    transformer_depth=4,
    attention_head_count=2,
    learning_rate=5e-4,
    training_batch_size=32,
    audio_learning_rate=5e-4,
    expansion_factor=1.25,
    audio_conv_kernel_size=5,
    audio_expansion_coefficient=2.0,
    feedforward_scaling=4.0,
    convolution_stack_depth=4,
    conv_channel_progression=[32, 48, 64, 80],
    adaptation_epoch_threshold=0,
)

STANDARD_NETWORK = NetworkConfiguration(
    architecture_name="base",
    layer_channels=[16, 16, 32, 64, 128],
    input_resolution=128,
    spectral_bins=80,
    acoustic_vector_size=80,
    acoustic_group_count=4,
    temporal_window_length=32,
    transformer_depth=4,
    attention_head_count=2,
    learning_rate=5e-4,
    training_batch_size=32,
    audio_learning_rate=5e-4,
    expansion_factor=2.0,
    audio_conv_kernel_size=5,
    audio_expansion_coefficient=2.0,
    feedforward_scaling=4.0,
    convolution_stack_depth=4,
    conv_channel_progression=[64, 96, 128, 160],
    adaptation_epoch_threshold=0,
)

NETWORK_REGISTRY = {
    "nano": MINIMAL_NETWORK,
    "tiny": COMPACT_NETWORK,
    "base": STANDARD_NETWORK,
}


def get_config(model_size: str) -> NetworkConfiguration:
    if model_size not in NETWORK_REGISTRY:
        raise ValueError(
            f"Unsupported architecture: {model_size}. Available options: {list(NETWORK_REGISTRY.keys())}"
        )
    return NETWORK_REGISTRY[model_size]


def list_available_models() -> List[str]:
    return list(NETWORK_REGISTRY.keys())


def get_border_from_crop_size(crop_size: int) -> int:
    if crop_size == 128:
        return 4
    else:
        return 3


def get_inner_size_from_crop_size(crop_size: int) -> int:
    border = get_border_from_crop_size(crop_size)
    return crop_size - 2 * border
