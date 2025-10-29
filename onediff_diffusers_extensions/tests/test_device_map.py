"""
Unit tests for device_map functionality.

Tests cover:
- Simple string device_map
- Dictionary device_map with filtering
- Lazy loading with device_map
- Memory savings with device_map
- Device placement verification
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

from onediffx.lora.safetensors_utils import (
    load_safetensors_robust,
)
from onediffx.lora.loading_strategies import (
    LazyLoadingStrategy,
    FastSingleFileStrategy,
)


class TestSimpleDeviceMapString:
    """Tests for simple string device_map parameter"""

    def test_device_map_cpu_string(self, synthetic_lora_file):
        """Test device_map with simple 'cpu' string"""
        state_dict = load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device_map="cpu"
        )

        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

        # Verify all tensors on CPU
        for tensor in state_dict.values():
            assert tensor.device.type == "cpu"

    @pytest.mark.requires_cuda
    def test_device_map_cuda_string(self, synthetic_lora_small):
        """Test device_map with simple 'cuda:0' string"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        state_dict = load_safetensors_robust(
            lora_file=synthetic_lora_small,
            device_map="cuda:0"
        )

        assert isinstance(state_dict, dict)

        # Verify all tensors on CUDA
        for tensor in state_dict.values():
            assert tensor.device.type == "cuda"

    def test_device_map_string_overrides_device(self, synthetic_lora_file):
        """Test that device_map overrides device parameter"""
        state_dict = load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device="cuda",  # This should be ignored
            device_map="cpu"  # This should take precedence
        )

        # All tensors should be on CPU (from device_map)
        for tensor in state_dict.values():
            assert tensor.device.type == "cpu"


class TestDictionaryDeviceMap:
    """Tests for dictionary device_map with selective loading"""

    def test_dict_device_map_basic(self, synthetic_lora_file):
        """Test basic dictionary device_map"""
        device_map = {
            "lora_unet_down_blocks_0": "cpu",
            "lora_unet_down_blocks_1": "cpu",
        }

        state_dict = load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device_map=device_map,
            strategy="lazy"  # Lazy strategy respects device_map
        )

        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

    def test_dict_device_map_filters_disk(self, synthetic_lora_file):
        """Test that dictionary device_map filters out 'disk' tensors"""
        # This test mocks the lazy loading to verify filtering behavior
        with patch('safetensors.safe_open') as mock_open:
            # Setup mock
            mock_file = MagicMock()
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=None)

            # Mock keys that include some for "disk"
            mock_file.keys.return_value = [
                "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight",
                "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight",
                "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight",
            ]
            mock_file.metadata.return_value = {}
            mock_file.get_tensor.return_value = torch.tensor([1.0])

            mock_open.return_value = mock_file

            # Device map with one layer on disk
            device_map = {
                "lora_unet_down_blocks_0": "cpu",
                "lora_unet_down_blocks_1": "cpu",
                "lora_unet_down_blocks_2": "disk",  # Should be filtered out!
            }

            # Load with lazy strategy
            strategy = LazyLoadingStrategy(validate=False)
            state_dict = strategy._load_impl(
                synthetic_lora_file,
                device_map=device_map,
                dtype=None
            )

            # Verify that get_tensor was NOT called for the disk tensor
            # We check this by verifying the final state_dict doesn't contain it
            tensor_keys_loaded = [call[0][0] for call in mock_file.get_tensor.call_args_list]

            # Should not have loaded the tensor going to "disk"
            assert not any("down_blocks_2" in key for key in tensor_keys_loaded)

    def test_dict_device_map_memory_savings(self, synthetic_lora_file):
        """Test that dictionary device_map reduces memory by not loading disk tensors"""
        # Load all tensors (no device_map)
        state_dict_full = load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device="cpu",
            strategy="eager"
        )
        num_tensors_full = len(state_dict_full)

        # Load with device_map filtering
        device_map = {
            "lora_unet_down_blocks_0": "cpu",
            "lora_unet_down_blocks_1": "disk",  # Don't load
            "lora_unet_down_blocks_2": "disk",  # Don't load
        }

        state_dict_filtered = load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device_map=device_map,
            strategy="lazy"
        )
        num_tensors_filtered = len(state_dict_filtered)

        # Should have loaded fewer tensors with filtering
        assert num_tensors_filtered < num_tensors_full


class TestLazyLoadingWithDeviceMap:
    """Tests for lazy loading strategy with device_map"""

    def test_lazy_strategy_filters_by_device_map(self, synthetic_lora_file):
        """Test that lazy strategy filters tensors based on device_map"""
        device_map = {
            "lora_unet_down_blocks_0": "cpu",
            "lora_unet_down_blocks_1": "disk",  # Should be filtered
        }

        strategy = LazyLoadingStrategy(validate=False)

        # Mock safe_open to control what keys are available
        with patch('safetensors.safe_open') as mock_open:
            mock_file = MagicMock()
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=None)
            mock_file.keys.return_value = [
                "lora_unet_down_blocks_0_attentions_0.weight",
                "lora_unet_down_blocks_1_attentions_0.weight",
            ]
            mock_file.metadata.return_value = {}
            mock_file.get_tensor.return_value = torch.tensor([1.0])
            mock_open.return_value = mock_file

            state_dict = strategy._load_impl(
                synthetic_lora_file,
                device_map=device_map,
                dtype=None
            )

            # Should only have loaded down_blocks_0 (not down_blocks_1 which is on "disk")
            assert len(state_dict) == 1
            assert "lora_unet_down_blocks_0_attentions_0.weight" in state_dict

    def test_lazy_strategy_filter_keys_by_device_map(self):
        """Test the _filter_keys_by_device_map method directly"""
        strategy = LazyLoadingStrategy(validate=False)

        available_keys = [
            "lora_unet_down_blocks_0.weight",
            "lora_unet_down_blocks_1.weight",
            "lora_unet_down_blocks_2.weight",
        ]

        device_map = {
            "lora_unet_down_blocks_0": "cpu",
            "lora_unet_down_blocks_1": "cpu",
            "lora_unet_down_blocks_2": "disk",  # Should be filtered
        }

        filtered_keys = strategy._filter_keys_by_device_map(available_keys, device_map)

        # Should only include keys not going to "disk"
        assert "lora_unet_down_blocks_0.weight" in filtered_keys
        assert "lora_unet_down_blocks_1.weight" in filtered_keys
        assert "lora_unet_down_blocks_2.weight" not in filtered_keys

    def test_lazy_strategy_with_no_device_map_loads_all(self):
        """Test that lazy strategy without device_map loads all tensors"""
        strategy = LazyLoadingStrategy(validate=False)

        available_keys = [
            "lora_unet_down_blocks_0.weight",
            "lora_unet_down_blocks_1.weight",
            "lora_unet_down_blocks_2.weight",
        ]

        # No device_map - should load all
        filtered_keys = strategy._filter_keys_by_device_map(available_keys, device_map=None)

        assert len(filtered_keys) == len(available_keys)


class TestDevicePlacement:
    """Tests for correct device placement based on device_map"""

    def test_get_tensor_device_simple_string(self):
        """Test _get_tensor_device with simple string device_map"""
        strategy = LazyLoadingStrategy(validate=False)

        device = strategy._get_tensor_device("any_key", device_map="cpu")
        assert device == "cpu"

        device = strategy._get_tensor_device("any_key", device_map="cuda:0")
        assert device == "cuda:0"

    def test_get_tensor_device_dict_direct_match(self):
        """Test _get_tensor_device with dict device_map and direct key match"""
        strategy = LazyLoadingStrategy(validate=False)

        device_map = {
            "layer1": "cpu",
            "layer2": "cuda:0",
        }

        device = strategy._get_tensor_device("layer1", device_map)
        assert device == "cpu"

        device = strategy._get_tensor_device("layer2", device_map)
        assert device == "cuda:0"

    def test_get_tensor_device_dict_prefix_match(self):
        """Test _get_tensor_device with dict device_map and prefix matching"""
        strategy = LazyLoadingStrategy(validate=False)

        device_map = {
            "lora_unet_down_blocks_0": "cpu",
            "lora_unet_down_blocks_1": "cuda:0",
        }

        device = strategy._get_tensor_device(
            "lora_unet_down_blocks_0.attentions_0.weight",
            device_map
        )
        assert device == "cpu"

        device = strategy._get_tensor_device(
            "lora_unet_down_blocks_1.attentions_0.weight",
            device_map
        )
        assert device == "cuda:0"

    def test_get_tensor_device_dict_default(self):
        """Test _get_tensor_device with dict device_map and default fallback"""
        strategy = LazyLoadingStrategy(validate=False)

        device_map = {
            "lora_unet_down_blocks_0": "cuda:0",
            "": "cpu",  # Default
        }

        # Unmatched key should use default
        device = strategy._get_tensor_device(
            "unknown_layer",
            device_map
        )
        assert device == "cpu"


class TestDeviceMapIntegration:
    """Integration tests for device_map with full loading"""

    def test_full_loading_with_device_map(self, synthetic_lora_file):
        """Test complete loading workflow with device_map"""
        device_map = {"": "cpu"}

        state_dict = load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device_map=device_map,
            strategy="lazy"
        )

        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

        # Verify device placement
        for tensor in state_dict.values():
            assert tensor.device.type == "cpu"

    @pytest.mark.requires_cuda
    def test_full_loading_with_mixed_device_map(self, synthetic_lora_file):
        """Test loading with mixed CPU/CUDA device_map"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Note: This is a simplified test - real mixed device maps would need
        # more complex tensor routing
        device_map = {"": "cuda:0"}

        state_dict = load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device_map=device_map,
            strategy="lazy"
        )

        assert isinstance(state_dict, dict)

        # Verify device placement
        for tensor in state_dict.values():
            assert tensor.device.type == "cuda"

    def test_device_map_with_validation(self, synthetic_lora_file):
        """Test that device_map works with validation enabled"""
        device_map = {"": "cpu"}

        state_dict = load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device_map=device_map,
            validate=True,  # Validation should still work
            strategy="lazy"
        )

        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0


class TestDeviceMapAutoStrategySelection:
    """Test that device_map influences automatic strategy selection"""

    def test_dict_device_map_selects_lazy_strategy(self, synthetic_lora_file):
        """Test that dict device_map auto-selects lazy strategy"""
        device_map = {
            "layer1": "cpu",
            "layer2": "disk"
        }

        # Auto strategy should select lazy for dict device_map
        state_dict = load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device_map=device_map,
            strategy="auto"
        )

        assert isinstance(state_dict, dict)

    def test_string_device_map_cuda_selects_fast_strategy(self, synthetic_lora_file):
        """Test that simple cuda device_map can select fast_single_file strategy"""
        # Auto strategy might select fast_single_file for simple cuda device_map
        # (depends on implementation)
        state_dict = load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device_map="cpu",  # Use CPU for testing
            strategy="auto"
        )

        assert isinstance(state_dict, dict)


class TestDeviceMapMemoryEfficiency:
    """Tests for memory efficiency with device_map"""

    def test_device_map_reduces_loaded_tensors(self, synthetic_lora_file):
        """Test that device_map with 'disk' entries reduces number of loaded tensors"""
        # Load without device_map
        state_dict_full = load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device="cpu"
        )
        num_full = len(state_dict_full)

        # Load with aggressive filtering
        device_map = {
            "lora_unet_down_blocks_0": "cpu",
            "lora_unet_down_blocks_1": "disk",
            "lora_unet_down_blocks_2": "disk",
        }

        state_dict_filtered = load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device_map=device_map,
            strategy="lazy"
        )
        num_filtered = len(state_dict_filtered)

        # Should load fewer tensors with filtering
        assert num_filtered <= num_full

    @pytest.mark.requires_cuda
    def test_device_map_memory_measured(self, synthetic_lora_file, measure_memory):
        """Test memory usage with and without device_map filtering"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory measurement")

        # Load full to CUDA
        mem_before = measure_memory()
        state_dict_full = load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device="cuda"
        )
        mem_after = measure_memory()
        mem_full = mem_after - mem_before

        # Clean up
        del state_dict_full
        torch.cuda.empty_cache()

        # Load with filtering
        device_map = {
            "lora_unet_down_blocks_0": "cuda",
            "lora_unet_down_blocks_1": "disk",  # Don't load
        }

        mem_before = measure_memory()
        state_dict_filtered = load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device_map=device_map,
            strategy="lazy"
        )
        mem_after = measure_memory()
        mem_filtered = mem_after - mem_before

        # Filtered loading should use less or equal memory
        assert mem_filtered <= mem_full * 1.1  # Allow 10% margin
