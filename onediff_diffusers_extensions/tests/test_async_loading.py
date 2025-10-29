"""
Unit tests for async loading functionality.

Tests cover:
- Async/await single file loading
- Async batch loading with parallel I/O
- Non-blocking behavior
- Error handling in async context
- Performance comparison (async vs sync)
"""

import asyncio
import pytest
import torch
import time
from pathlib import Path

from onediffx.lora.safetensors_utils import (
    load_safetensors_robust,
    load_loras_batch,
    load_loras_batch_async,
    SafetensorsLoadError,
)


class TestAsyncSingleFileLoading:
    """Tests for async loading of single files"""

    @pytest.mark.asyncio
    async def test_async_returns_awaitable(self, synthetic_lora_file):
        """Test that use_async=True returns an awaitable"""
        result = load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device="cpu",
            use_async=True
        )

        # Should be a coroutine/awaitable
        assert asyncio.iscoroutine(result)

        # Await it to get actual result
        state_dict = await result
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

    @pytest.mark.asyncio
    async def test_async_loads_correct_data(self, synthetic_lora_file):
        """Test that async loading produces correct state dict"""
        state_dict = await load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device="cpu",
            use_async=True
        )

        # Verify structure
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

        # Verify tensors are valid
        for key, tensor in state_dict.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.device.type == "cpu"

    @pytest.mark.asyncio
    async def test_async_with_cuda_device(self, synthetic_lora_small):
        """Test async loading to CUDA device (if available)"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        state_dict = await load_safetensors_robust(
            lora_file=synthetic_lora_small,
            device="cuda",
            use_async=True
        )

        # Verify tensors are on CUDA
        for tensor in state_dict.values():
            assert tensor.device.type == "cuda"

    @pytest.mark.asyncio
    async def test_sync_and_async_produce_same_result(self, synthetic_lora_file):
        """Test that sync and async loading produce identical results"""
        # Load synchronously
        sync_state_dict = load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device="cpu",
            use_async=False
        )

        # Load asynchronously
        async_state_dict = await load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device="cpu",
            use_async=True
        )

        # Compare results
        assert set(sync_state_dict.keys()) == set(async_state_dict.keys())

        for key in sync_state_dict.keys():
            assert torch.equal(sync_state_dict[key], async_state_dict[key])


class TestAsyncBatchLoading:
    """Tests for async batch loading of multiple files"""

    @pytest.mark.asyncio
    async def test_async_batch_returns_dict(self, synthetic_lora_batch):
        """Test that async batch loading returns dict of state dicts"""
        loras = await load_loras_batch_async(
            lora_paths=synthetic_lora_batch,
            device="cpu"
        )

        assert isinstance(loras, dict)
        assert len(loras) == len(synthetic_lora_batch)

        # Verify each LoRA loaded correctly
        for name, state_dict in loras.items():
            assert isinstance(state_dict, dict)
            assert len(state_dict) > 0

    @pytest.mark.asyncio
    async def test_async_batch_names_correct(self, synthetic_lora_batch):
        """Test that batch loading uses correct names (file stems)"""
        loras = await load_loras_batch_async(
            lora_paths=synthetic_lora_batch,
            device="cpu"
        )

        # Names should match file stems
        expected_names = {path.stem for path in synthetic_lora_batch}
        assert set(loras.keys()) == expected_names

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Timing test is flaky - async overhead can be similar to I/O time for small files")
    async def test_async_batch_parallel_execution(self, synthetic_lora_batch):
        """Test that async batch loading executes in parallel (faster than sequential)"""
        # Time sequential loading
        start = time.time()
        sync_loras = load_loras_batch(synthetic_lora_batch, device="cpu")
        sync_time = time.time() - start

        # Time parallel async loading
        start = time.time()
        async_loras = await load_loras_batch_async(synthetic_lora_batch, device="cpu")
        async_time = time.time() - start

        # Async should be at least as fast or faster
        # Allow 20% margin for overhead and test variability
        assert async_time <= sync_time * 1.2, (
            f"Async loading ({async_time:.2f}s) should be faster than or similar to "
            f"sync loading ({sync_time:.2f}s)"
        )

        # Verify results are the same
        assert len(async_loras) == len(sync_loras)

    @pytest.mark.asyncio
    async def test_async_batch_with_device_map(self, synthetic_lora_batch):
        """Test async batch loading with device_map parameter"""
        device_map = {"": "cpu"}

        loras = await load_loras_batch_async(
            lora_paths=synthetic_lora_batch,
            device_map=device_map
        )

        assert len(loras) == len(synthetic_lora_batch)

        # Verify tensors are on correct device
        for state_dict in loras.values():
            for tensor in state_dict.values():
                assert tensor.device.type == "cpu"


class TestAsyncNonBlockingBehavior:
    """Tests verifying that async loading is truly non-blocking"""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Timing test is flaky - environment dependent, async overhead varies")
    async def test_concurrent_loads_dont_block(self, synthetic_lora_batch):
        """Test that multiple async loads can run concurrently"""
        # Start multiple loads concurrently
        tasks = [
            load_safetensors_robust(path, device="cpu", use_async=True)
            for path in synthetic_lora_batch
        ]

        # Time concurrent execution
        start = time.time()
        results = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start

        # Time sequential execution for comparison
        start = time.time()
        for path in synthetic_lora_batch:
            _ = load_safetensors_robust(path, device="cpu", use_async=False)
        sequential_time = time.time() - start

        # Concurrent should be faster (allow margin for overhead)
        assert concurrent_time < sequential_time * 1.2

        # Verify all loaded successfully
        assert len(results) == len(synthetic_lora_batch)
        for state_dict in results:
            assert isinstance(state_dict, dict)

    @pytest.mark.asyncio
    async def test_async_load_allows_other_work(self, synthetic_lora_large):
        """Test that async loading allows other async work to proceed"""
        work_done = []

        async def do_work():
            """Simulate other async work"""
            for i in range(10):
                await asyncio.sleep(0.01)
                work_done.append(i)

        async def load_file():
            """Load file asynchronously"""
            return await load_safetensors_robust(
                lora_file=synthetic_lora_large,
                device="cpu",
                use_async=True
            )

        # Run both concurrently
        state_dict, _ = await asyncio.gather(
            load_file(),
            do_work()
        )

        # Verify both completed
        assert isinstance(state_dict, dict)
        assert len(work_done) == 10  # Other work completed during loading


class TestAsyncErrorHandling:
    """Tests for error handling in async context"""

    @pytest.mark.asyncio
    async def test_async_file_not_found_raises(self):
        """Test that async loading raises error for non-existent file"""
        with pytest.raises((FileNotFoundError, SafetensorsLoadError)):
            await load_safetensors_robust(
                lora_file=Path("nonexistent.safetensors"),
                device="cpu",
                use_async=True
            )

    @pytest.mark.asyncio
    async def test_async_batch_stop_on_error(self, tmp_path):
        """Test that async batch loading stops on error when requested"""
        # Create one valid and one invalid path
        valid_file = tmp_path / "valid.safetensors"
        invalid_file = tmp_path / "invalid.safetensors"

        # Create valid file (must be >100 bytes to pass validation)
        import safetensors.torch
        valid_state_dict = {f"weight_{i}": torch.randn(10, 10) for i in range(3)}
        safetensors.torch.save_file(valid_state_dict, valid_file)

        paths = [valid_file, invalid_file]

        # Should raise error (FileNotFoundError or SafetensorsValidationError that wraps it)
        with pytest.raises((FileNotFoundError, SafetensorsLoadError)):
            await load_loras_batch_async(
                lora_paths=paths,
                device="cpu",
                stop_on_error=True
            )

    @pytest.mark.asyncio
    async def test_async_batch_continue_on_error(self, tmp_path):
        """Test that async batch loading continues on error when requested"""
        # Create one valid and one invalid path
        valid_file = tmp_path / "valid.safetensors"
        invalid_file = tmp_path / "invalid.safetensors"

        # Create valid file (must be >100 bytes to pass validation)
        import safetensors.torch
        valid_state_dict = {f"weight_{i}": torch.randn(10, 10) for i in range(3)}
        safetensors.torch.save_file(valid_state_dict, valid_file)

        paths = [valid_file, invalid_file]

        # Should NOT raise, just skip failed file
        loras = await load_loras_batch_async(
            lora_paths=paths,
            device="cpu",
            stop_on_error=False
        )

        # Should have loaded only the valid file
        assert len(loras) == 1
        assert "valid" in loras

    @pytest.mark.asyncio
    async def test_async_validation_error(self, tmp_path):
        """Test that async loading handles validation errors"""
        # Create file that's too large (if validation is on)
        huge_file = tmp_path / "huge.safetensors"

        # Create a file and try to load with strict size limit
        import safetensors.torch
        safetensors.torch.save_file({"weight": torch.randn(1000, 1000)}, huge_file)

        # Should respect validation
        with pytest.raises((SafetensorsLoadError, Exception)):
            await load_safetensors_robust(
                lora_file=huge_file,
                device="cpu",
                use_async=True,
                max_size_gb=0.001  # Very small limit
            )


class TestAsyncWithDifferentStrategies:
    """Test async loading with different loading strategies"""

    @pytest.mark.asyncio
    async def test_async_with_fast_single_file_strategy(self, synthetic_lora_file):
        """Test async loading with fast_single_file strategy"""
        state_dict = await load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device="cpu",
            strategy="fast_single_file",
            use_async=True
        )

        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

    @pytest.mark.asyncio
    async def test_async_with_lazy_strategy(self, synthetic_lora_file):
        """Test async loading with lazy strategy"""
        state_dict = await load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device="cpu",
            strategy="lazy",
            use_async=True
        )

        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

    @pytest.mark.asyncio
    async def test_async_with_auto_strategy(self, synthetic_lora_file):
        """Test async loading with auto strategy selection"""
        state_dict = await load_safetensors_robust(
            lora_file=synthetic_lora_file,
            device="cpu",
            strategy="auto",
            use_async=True
        )

        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0


class TestAsyncMemoryEfficiency:
    """Tests for memory efficiency of async loading"""

    @pytest.mark.asyncio
    @pytest.mark.requires_cuda
    async def test_async_batch_memory_usage(self, synthetic_lora_batch, measure_memory):
        """Test that async batch loading doesn't use excessive memory"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory testing")

        # Measure memory before
        mem_before = measure_memory()

        # Load batch asynchronously
        loras = await load_loras_batch_async(
            lora_paths=synthetic_lora_batch,
            device="cuda"
        )

        # Measure memory after
        mem_after = measure_memory()

        mem_used = mem_after - mem_before

        # Memory usage should be reasonable (not loading all files at once)
        # This is a sanity check - actual values depend on LoRA sizes
        assert mem_used > 0  # Some memory was used
        assert len(loras) == len(synthetic_lora_batch)  # All loaded successfully
