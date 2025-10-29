"""
Pytest configuration and shared fixtures for OneDiffX tests.

This module provides:
- HuggingFace model fixtures (tiny SDXL for testing)
- Synthetic LoRA file fixtures
- Shared test utilities
- Common test configurations
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict
import safetensors.torch


# ============================================================================
# HuggingFace Model Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def hf_tiny_sdxl_cache_dir(tmp_path_factory):
    """Create a temporary cache directory for HF models (session scope)"""
    cache_dir = tmp_path_factory.mktemp("hf_cache")
    yield cache_dir
    # Cleanup after all tests
    shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def hf_tiny_sdxl_model(hf_tiny_sdxl_cache_dir):
    """
    Download HuggingFace tiny SDXL model once for all tests (session scope).

    Uses 'hf-internal-testing/tiny-stable-diffusion-xl-pipe' which is:
    - ~50MB (very small, fast download)
    - Official HF testing model
    - Has same structure as real SDXL
    """
    try:
        from diffusers import DiffusionPipeline

        model_id = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"

        # Download model to cache
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir=hf_tiny_sdxl_cache_dir,
            local_files_only=False
        )

        # Return the cache directory path for safetensors files
        model_path = hf_tiny_sdxl_cache_dir / "models--hf-internal-testing--tiny-stable-diffusion-xl-pipe"
        return model_path

    except Exception as e:
        pytest.skip(f"Could not download HF tiny SDXL model: {e}")


@pytest.fixture(scope="session")
def hf_tiny_sdxl_unet_file(hf_tiny_sdxl_model):
    """Get the UNet safetensors file from tiny SDXL model"""
    # Find the unet safetensors file
    unet_files = list(hf_tiny_sdxl_model.rglob("**/unet/*.safetensors"))
    if unet_files:
        return unet_files[0]

    # If not found, skip tests
    pytest.skip("Could not find UNet safetensors file in tiny SDXL model")


# ============================================================================
# Synthetic LoRA File Fixtures
# ============================================================================

def create_synthetic_lora(path: Path, num_layers: int = 10, size: int = 128) -> Path:
    """
    Create a synthetic LoRA file with realistic structure.

    Args:
        path: Path to save the LoRA file
        num_layers: Number of layers to create (default: 10 for realistic size)
        size: Size of weight matrices (default: 128 for ~5MB files)

    Returns:
        Path to the created LoRA file

    Note:
        Default creates ~5MB files for realistic async performance testing.
        For smaller/faster tests, use synthetic_lora_small fixture.
    """
    state_dict = {}

    # Create Kohya-style LoRA structure
    for i in range(num_layers):
        # LoRA down weights
        state_dict[f"lora_unet_down_blocks_{i}_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight"] = torch.randn(size, size)
        # LoRA up weights
        state_dict[f"lora_unet_down_blocks_{i}_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight"] = torch.randn(size, size)
        # Alpha values
        state_dict[f"lora_unet_down_blocks_{i}_attentions_0_transformer_blocks_0_attn1_to_k.alpha"] = torch.tensor([4.0])

    # Save as safetensors
    safetensors.torch.save_file(state_dict, path)
    return path


@pytest.fixture
def synthetic_lora_file(tmp_path):
    """Create a single synthetic LoRA file for testing (~5MB, realistic size)"""
    lora_path = tmp_path / "test_lora.safetensors"
    return create_synthetic_lora(lora_path)  # Use defaults: num_layers=10, size=128


@pytest.fixture
def synthetic_lora_small(tmp_path):
    """Create a very small synthetic LoRA for fast tests (~50KB)"""
    lora_path = tmp_path / "test_lora_small.safetensors"
    return create_synthetic_lora(lora_path, num_layers=1, size=16)


@pytest.fixture
def synthetic_lora_batch(tmp_path) -> List[Path]:
    """Create multiple synthetic LoRA files for batch testing (~3MB each)"""
    lora_paths = []
    for i in range(4):
        lora_path = tmp_path / f"test_lora_{i}.safetensors"
        # Larger files to show async benefit: 8 layers × 128×128 = ~3MB each
        create_synthetic_lora(lora_path, num_layers=8, size=128)
        lora_paths.append(lora_path)
    return lora_paths


@pytest.fixture
def synthetic_lora_large(tmp_path):
    """Create a larger synthetic LoRA for performance testing"""
    lora_path = tmp_path / "test_lora_large.safetensors"
    return create_synthetic_lora(lora_path, num_layers=10, size=128)


# ============================================================================
# Device Map Fixtures
# ============================================================================

@pytest.fixture
def simple_device_map_cpu():
    """Simple device map pointing to CPU"""
    return "cpu"


@pytest.fixture
def simple_device_map_cuda():
    """Simple device map pointing to CUDA (skips if CUDA unavailable)"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda:0"


@pytest.fixture
def dict_device_map_cpu():
    """Dictionary device map with CPU and disk"""
    return {
        "lora_unet_down_blocks_0": "cpu",
        "lora_unet_down_blocks_1": "cpu",
        "lora_unet_down_blocks_2": "disk",  # This should be filtered out
    }


@pytest.fixture
def dict_device_map_mixed():
    """Dictionary device map with mixed devices (requires CUDA)"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    return {
        "lora_unet_down_blocks_0": "cuda:0",
        "lora_unet_down_blocks_1": "cpu",
        "lora_unet_down_blocks_2": "disk",  # This should be filtered out
    }


# ============================================================================
# Test Utilities
# ============================================================================

@pytest.fixture
def measure_memory():
    """Fixture to measure memory usage during test"""
    import gc

    def _measure():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / 1024**2  # MB
        return 0

    return _measure


@pytest.fixture
def measure_time():
    """Fixture to measure execution time"""
    import time

    class TimeContext:
        def __init__(self):
            self.elapsed = 0

        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, *args):
            self.elapsed = time.time() - self.start

    return TimeContext


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "asyncio: marks tests as async"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_cuda: marks tests that require CUDA"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests based on markers and environment"""
    skip_cuda = pytest.mark.skip(reason="CUDA not available")

    for item in items:
        # Skip CUDA tests if CUDA not available
        if "requires_cuda" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_cuda)


# ============================================================================
# Async Test Support
# ============================================================================

@pytest.fixture
def event_loop():
    """Create an event loop for async tests"""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
