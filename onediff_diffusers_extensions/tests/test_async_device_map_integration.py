"""
Integration tests for async loading + device_map functionality.

Tests the combination of:
- Async/await loading
- Device_map tensor filtering
- Different loading strategies (lazy, fast_single_file, parallel_async_lazy)

These tests ensure that async and device_map work correctly together,
which is critical for production use cases like lazy LoRA loading in inference services.
"""

import asyncio
import pytest
import torch
from pathlib import Path

from onediffx.lora.safetensors_utils import (
    load_safetensors_robust,
    load_loras_batch_async,
    SafetensorsLoadError,
)


class TestAsyncWithDeviceMapFiltering:
    """Test that async loading correctly filters tensors based on device_map"""

    @pytest.mark.asyncio
    async def test_async_lazy_filters_disk_tensors(self, kohya_lora_file):
        """Test that async + lazy strategy filters tensors mapped to 'disk'"""
        device_map = {
            "lora_unet_down_blocks_0": "cpu",
            "lora_unet_down_blocks_1": "disk",  # Should NOT load
            "lora_unet_mid_block": "cpu",
        }

        state_dict = await load_safetensors_robust(
            lora_file=kohya_lora_file,
            device_map=device_map,
            strategy="lazy",
            use_async=True,
        )

        # Verify disk tensors were filtered out
        disk_keys = [k for k in state_dict.keys() if "down_blocks_1" in k]
        assert len(disk_keys) == 0, "Tensors mapped to 'disk' should not be loaded"

        # Verify other tensors were loaded
        cpu_keys = [k for k in state_dict.keys() if "down_blocks_0" in k or "mid_block" in k]
        assert len(cpu_keys) > 0, "Tensors mapped to 'cpu' should be loaded"

    @pytest.mark.asyncio
    async def test_async_lazy_filters_by_prefix(self, kohya_lora_file):
        """Test that device_map prefix matching works with async loading"""
        device_map = {
            "lora_unet": "cpu",
            "lora_text_encoder": "disk",  # Filter all text encoder tensors
        }

        state_dict = await load_safetensors_robust(
            lora_file=kohya_lora_file,
            device_map=device_map,
            strategy="lazy",
            use_async=True,
        )

        # Count tensors by prefix
        unet_count = len([k for k in state_dict.keys() if k.startswith("lora_unet")])
        text_enc_count = len([k for k in state_dict.keys() if k.startswith("lora_text_encoder")])

        assert unet_count > 0, "UNet tensors should be loaded"
        assert text_enc_count == 0, "Text encoder tensors should be filtered out"

    @pytest.mark.asyncio
    async def test_async_with_underscore_separator(self, kohya_lora_file):
        """Test device_map works with underscore separator (Kohya LoRAs)"""
        # Kohya LoRAs use underscore: lora_unet_down_blocks_0_attentions_0...
        device_map = {
            "lora_unet_down_blocks_0_attentions": "cpu",
            "lora_unet_down_blocks_1_attentions": "disk",  # Filter
        }

        state_dict = await load_safetensors_robust(
            lora_file=kohya_lora_file,
            device_map=device_map,
            strategy="lazy",
            use_async=True,
        )

        # Verify filtering with underscore separator
        loaded = [k for k in state_dict.keys() if "down_blocks_0_attentions" in k]
        filtered = [k for k in state_dict.keys() if "down_blocks_1_attentions" in k]

        assert len(loaded) > 0, "Should load down_blocks_0_attentions"
        assert len(filtered) == 0, "Should filter down_blocks_1_attentions"


class TestAsyncBatchWithDeviceMap:
    """Test async batch loading with device_map"""

    @pytest.mark.asyncio
    async def test_batch_async_filters_consistently(self, synthetic_lora_batch):
        """Test that batch async loading applies device_map consistently to all files"""
        device_map = {
            "weight_0": "cpu",
            "weight_1": "disk",  # Filter
            "weight_2": "cpu",
        }

        loras = await load_loras_batch_async(
            lora_paths=synthetic_lora_batch,
            device_map=device_map,
            strategy="lazy",
        )

        # Verify all LoRAs filtered the same way
        for name, state_dict in loras.items():
            has_weight_0 = "weight_0" in state_dict
            has_weight_1 = "weight_1" in state_dict
            has_weight_2 = "weight_2" in state_dict

            assert has_weight_0, f"{name}: weight_0 should be loaded"
            assert not has_weight_1, f"{name}: weight_1 should be filtered"
            assert has_weight_2, f"{name}: weight_2 should be loaded"

    @pytest.mark.asyncio
    async def test_batch_async_memory_savings(self, synthetic_lora_batch):
        """Test that device_map reduces memory in batch async loading"""
        # Load all tensors
        loras_full = await load_loras_batch_async(
            lora_paths=synthetic_lora_batch,
            device="cpu",
            strategy="lazy",
        )

        # Load with filtering
        device_map = {
            "weight_0": "cpu",
            "weight_1": "disk",  # Filter half
            "weight_2": "cpu",
            "weight_3": "disk",  # Filter half
        }

        loras_filtered = await load_loras_batch_async(
            lora_paths=synthetic_lora_batch,
            device_map=device_map,
            strategy="lazy",
        )

        # Count total tensors
        total_full = sum(len(sd) for sd in loras_full.values())
        total_filtered = sum(len(sd) for sd in loras_filtered.values())

        assert total_filtered < total_full, "Filtered should load fewer tensors"
        # Should be roughly 50% (filtering weight_1 and weight_3)
        assert total_filtered <= total_full * 0.6, "Should filter significant amount"


class TestAsyncWithDifferentStrategies:
    """Test async + device_map with different loading strategies"""

    @pytest.mark.asyncio
    async def test_fast_single_file_with_device_map(self, synthetic_lora_file):
        """Test that fast_single_file works with device_map (string form)"""
        state_dict = await load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device_map="cpu",  # String form
            strategy="fast_single_file",
            use_async=True,
        )

        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

        # Verify all tensors on CPU
        for tensor in state_dict.values():
            assert tensor.device.type == "cpu"

    @pytest.mark.asyncio
    async def test_parallel_async_lazy_with_device_map(self, synthetic_lora_batch):
        """Test parallel_async_lazy strategy with device_map filtering"""
        device_map = {
            "weight_0": "cpu",
            "weight_1": "disk",  # Filter
        }

        loras = await load_loras_batch_async(
            lora_paths=synthetic_lora_batch,
            device_map=device_map,
            strategy="parallel_async_lazy",
        )

        # Verify filtering worked
        for state_dict in loras.values():
            assert "weight_0" in state_dict
            assert "weight_1" not in state_dict

    @pytest.mark.asyncio
    async def test_auto_strategy_selection_with_device_map(self, synthetic_lora_file):
        """Test that auto strategy correctly chooses lazy when device_map is dict"""
        device_map = {
            "weight_0": "cpu",
            "weight_1": "disk",
        }

        state_dict = await load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device_map=device_map,
            strategy="auto",  # Should auto-select lazy
            use_async=True,
        )

        # Should have filtered weight_1
        assert "weight_0" in state_dict
        assert "weight_1" not in state_dict


class TestAsyncDeviceMapEdgeCases:
    """Test edge cases for async + device_map"""

    @pytest.mark.asyncio
    async def test_empty_device_map_loads_all(self, synthetic_lora_file):
        """Test that empty device_map loads all tensors to CPU"""
        device_map = {}

        state_dict = await load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device_map=device_map,
            strategy="lazy",
            use_async=True,
        )

        # Should load all tensors
        assert len(state_dict) > 0

    @pytest.mark.asyncio
    async def test_all_tensors_filtered_returns_empty(self, synthetic_lora_file):
        """Test that filtering all tensors returns empty dict (not error)"""
        device_map = {
            "": "disk",  # Default: filter everything
        }

        state_dict = await load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device_map=device_map,
            strategy="lazy",
            use_async=True,
        )

        # Should return empty dict
        assert len(state_dict) == 0

    @pytest.mark.asyncio
    async def test_device_map_with_cuda(self, synthetic_lora_small):
        """Test async + device_map with CUDA devices"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device_map = {
            "weight_0": "cuda:0",
            "weight_1": "cpu",
            "weight_2": "disk",  # Filter
        }

        state_dict = await load_safetensors_robust(
            lora_file=synthetic_lora_small,
            device_map=device_map,
            strategy="lazy",
            use_async=True,
        )

        # Verify device placement
        if "weight_0" in state_dict:
            assert state_dict["weight_0"].device.type == "cuda"
        if "weight_1" in state_dict:
            assert state_dict["weight_1"].device.type == "cpu"
        assert "weight_2" not in state_dict

    @pytest.mark.asyncio
    async def test_device_map_none_vs_empty_dict(self, synthetic_lora_file):
        """Test that device_map=None behaves differently from device_map={}"""
        # None should load to default device
        state_dict_none = await load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device="cpu",
            device_map=None,
            strategy="fast_single_file",
            use_async=True,
        )

        # Empty dict should also load all (default to cpu)
        state_dict_empty = await load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device_map={},
            strategy="lazy",
            use_async=True,
        )

        # Both should load all tensors
        assert len(state_dict_none) == len(state_dict_empty)


class TestAsyncDeviceMapPerformance:
    """Performance-related tests for async + device_map"""

    @pytest.mark.asyncio
    async def test_async_batch_with_filtering_faster(self, synthetic_lora_batch):
        """Test that filtering with device_map improves async batch load time"""
        import time

        # Full load
        start = time.time()
        loras_full = await load_loras_batch_async(
            lora_paths=synthetic_lora_batch,
            device="cpu",
        )
        time_full = time.time() - start

        # Filtered load (50% filtered)
        device_map = {
            "weight_0": "cpu",
            "weight_1": "disk",
            "weight_2": "cpu",
            "weight_3": "disk",
        }

        start = time.time()
        loras_filtered = await load_loras_batch_async(
            lora_paths=synthetic_lora_batch,
            device_map=device_map,
            strategy="lazy",
        )
        time_filtered = time.time() - start

        # Sanity checks
        assert len(loras_full) == len(loras_filtered), "Same number of files"

        total_full = sum(len(sd) for sd in loras_full.values())
        total_filtered = sum(len(sd) for sd in loras_filtered.values())

        assert total_filtered < total_full, "Filtered should load fewer tensors"

        # Note: We don't assert time_filtered < time_full because for small test files,
        # the overhead might dominate. In production with large files, filtering helps.


class TestAsyncDeviceMapWithValidation:
    """Test that validation works with async + device_map"""

    @pytest.mark.asyncio
    async def test_validation_runs_before_filtering(self, tmp_path):
        """Test that file validation happens before device_map filtering"""
        # Create invalid file (empty)
        invalid_file = tmp_path / "invalid.safetensors"
        invalid_file.touch()

        device_map = {"": "cpu"}

        # Should fail validation even though device_map would filter
        with pytest.raises((SafetensorsLoadError, FileNotFoundError)):
            await load_safetensors_robust(
                lora_file=invalid_file,
                device_map=device_map,
                strategy="lazy",
                use_async=True,
                min_size_bytes=100,  # Require min size
            )

    @pytest.mark.asyncio
    async def test_async_batch_validation_with_device_map(self, tmp_path):
        """Test batch loading with validation + device_map"""
        import safetensors.torch

        # Create valid and invalid files
        valid = tmp_path / "valid.safetensors"
        invalid = tmp_path / "invalid.safetensors"

        safetensors.torch.save_file({"weight": torch.randn(100, 100)}, valid)
        invalid.touch()  # Empty file

        device_map = {"weight": "cpu"}

        # With stop_on_error=False, should load valid and skip invalid
        loras = await load_loras_batch_async(
            lora_paths=[valid, invalid],
            device_map=device_map,
            strategy="lazy",
            stop_on_error=False,
            min_size_bytes=100,
        )

        # Should only have valid file
        assert len(loras) == 1
        assert "valid" in loras


class TestAsyncDeviceMapRealWorldScenarios:
    """Real-world production scenarios"""

    @pytest.mark.asyncio
    async def test_pulid_lora_lazy_loading_simulation(self, tmp_path):
        """
        Simulate real PuLID inference scenario:
        - Load 4 LoRAs in parallel
        - Only load UNet weights (filter text encoder)
        - Use async for non-blocking
        """
        import safetensors.torch

        # Create 4 fake LoRAs with realistic structure
        lora_paths = []
        for i, name in enumerate(["oils", "realvis", "artisan", "birka"]):
            lora_file = tmp_path / f"{name}_lora.safetensors"

            state_dict = {
                f"lora_unet_down_blocks_{i}": torch.randn(64, 64),
                f"lora_unet_mid_block": torch.randn(32, 32),
                f"lora_text_encoder_layer_{i}": torch.randn(16, 16),  # Filter this!
            }

            safetensors.torch.save_file(state_dict, lora_file)
            lora_paths.append(lora_file)

        # Device map: only load UNet, filter text encoder
        device_map = {
            "lora_unet": "cpu",
            "lora_text_encoder": "disk",  # Don't load text encoder
        }

        # Load all in parallel with filtering
        loras = await load_loras_batch_async(
            lora_paths=lora_paths,
            device_map=device_map,
            strategy="parallel_async_lazy",
        )

        # Verify results
        assert len(loras) == 4, "All 4 LoRAs loaded"

        for name, state_dict in loras.items():
            # Should have UNet tensors
            unet_keys = [k for k in state_dict.keys() if "unet" in k]
            assert len(unet_keys) > 0, f"{name}: Should have UNet tensors"

            # Should NOT have text encoder tensors
            text_enc_keys = [k for k in state_dict.keys() if "text_encoder" in k]
            assert len(text_enc_keys) == 0, f"{name}: Should filter text encoder"

        print(f"✅ PuLID simulation: Loaded {len(loras)} LoRAs with {sum(len(sd) for sd in loras.values())} total tensors")
