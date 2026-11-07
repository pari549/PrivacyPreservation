"""Image-input utilities for the independent non-recoverable pipeline."""

from src.non_recoverable.image_input import (
    DEFAULT_BACKGROUND_COLOR,
    DEFAULT_MAX_IMAGE_PIXELS,
    ImageInputError,
    ImageSizeLimitError,
    PreparedImage,
    extract_image_block_features,
    iter_image_blocks,
    load_image,
    prepare_image,
    restore_original_dimensions,
)
from src.non_recoverable.feature_matrix import (
    DEFAULT_FEATURE_NAMES,
    ImageFeatureMatrix,
    extract_feature_matrix,
)
from src.non_recoverable.optimizer import optimize_feature_weights, project_bounded_simplex
from src.non_recoverable.baseline_transformations import (
    MAX_QUANTIZATION_LEVELS,
    MAX_SMOOTHING_STRENGTH,
    MIN_QUANTIZATION_LEVELS,
    combined_transformation,
    uniform_gaussian_smoothing,
    uniform_quantization,
)
from src.non_recoverable.adaptive_transformations import (
    DEFAULT_BLOCK_SIZE,
    adaptive_combined_transformation,
    adaptive_gaussian_smoothing,
    adaptive_quantization,
)
from src.non_recoverable.transformation_parameter_codec import (
    INITIAL_8BIT_QUANTIZATION_MENU,
    DecodedTransformationConfiguration,
    ExpandedBlockParameters,
    TransformationParameterCodec,
)

__all__ = [
    "DEFAULT_BACKGROUND_COLOR",
    "DEFAULT_MAX_IMAGE_PIXELS",
    "ImageInputError",
    "ImageSizeLimitError",
    "PreparedImage",
    "extract_image_block_features",
    "iter_image_blocks",
    "load_image",
    "prepare_image",
    "restore_original_dimensions",
    "DEFAULT_FEATURE_NAMES",
    "ImageFeatureMatrix",
    "extract_feature_matrix",
    "optimize_feature_weights",
    "project_bounded_simplex",
    "MAX_QUANTIZATION_LEVELS",
    "MAX_SMOOTHING_STRENGTH",
    "MIN_QUANTIZATION_LEVELS",
    "combined_transformation",
    "uniform_gaussian_smoothing",
    "uniform_quantization",
    "DEFAULT_BLOCK_SIZE",
    "adaptive_combined_transformation",
    "adaptive_gaussian_smoothing",
    "adaptive_quantization",
    "INITIAL_8BIT_QUANTIZATION_MENU",
    "DecodedTransformationConfiguration",
    "ExpandedBlockParameters",
    "TransformationParameterCodec",
]
