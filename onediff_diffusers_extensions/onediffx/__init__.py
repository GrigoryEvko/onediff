"""
OneDiffX Diffusers Extensions - LoRA Loading Utilities

This package provides standalone LoRA loading functionality with:
- Async/await support for non-blocking loading
- Device_map support for distributed models
- Multiple loading strategies (eager, lazy, fast_single_file, parallel_async_lazy)
- Comprehensive error handling and validation
- Security features (path traversal protection, size limits)

Usage:
    from onediffx.lora.safetensors_utils import load_safetensors_robust
    from onediffx.lora.loading_strategies import get_loading_strategy

    # Async loading
    state_dict = await load_safetensors_robust(path, use_async=True)

    # Device map for memory efficiency
    state_dict = load_safetensors_robust(path, device_map={"layer1": "cuda:0", "layer2": "disk"})
"""

try:
    from ._version import version as __version__, version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

__all__ = []
