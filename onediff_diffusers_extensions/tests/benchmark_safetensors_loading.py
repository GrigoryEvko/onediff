"""
Performance benchmarks for safetensors loading improvements.

Compares:
- Old approach (direct safetensors.torch.load_file) vs new robust loader
- Memory usage with and without .copy()
- Streaming vs direct loading for large files
- Batch loading performance
- Caching benefits

Run with: pytest benchmark_safetensors_loading.py -v --benchmark-only
Or: python benchmark_safetensors_loading.py
"""

import time
import tempfile
from pathlib import Path
import psutil
import os

import torch
import safetensors.torch

# Import our improved utilities
from onediffx.lora.safetensors_utils import (
    load_safetensors_robust,
    load_loras_batch,
    load_safetensors_cached,
    clear_lora_cache,
)


def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def create_mock_lora_file(path: Path, size_mb: int = 50):
    """
    Create a mock safetensors file of specified size.

    Args:
        path: Path to create file at
        size_mb: Approximate size in MB
    """
    # Create tensors that sum to approximately size_mb
    num_tensors = 10
    tensor_size = (size_mb * 1024 * 1024) // (num_tensors * 4)  # 4 bytes per float32

    state_dict = {}
    for i in range(num_tensors):
        # Create square-ish tensors
        dim = int(tensor_size ** 0.5)
        state_dict[f"layer{i}.weight"] = torch.randn(dim, dim, dtype=torch.float32)

    safetensors.torch.save_file(state_dict, path)
    return path


def benchmark_loading_speed():
    """Benchmark: Loading speed comparison."""
    print("\n" + "="*80)
    print("BENCHMARK: Loading Speed (Old vs New)")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file
        test_file = Path(tmpdir) / "test.safetensors"
        create_mock_lora_file(test_file, size_mb=50)

        # Old approach
        start = time.time()
        for _ in range(5):
            state_dict = safetensors.torch.load_file(test_file, device="cpu")
        old_time = (time.time() - start) / 5

        # New approach
        start = time.time()
        for _ in range(5):
            state_dict = load_safetensors_robust(test_file, device="cpu")
        new_time = (time.time() - start) / 5

        print(f"Old approach (direct load_file):  {old_time:.3f}s per load")
        print(f"New approach (robust loader):      {new_time:.3f}s per load")
        print(f"Overhead:                          {((new_time/old_time - 1)*100):.1f}%")

        if new_time <= old_time * 1.1:
            print("✅ PASS: New approach has minimal overhead (<10%)")
        else:
            print(f"⚠️  WARNING: New approach slower by {((new_time/old_time - 1)*100):.1f}%")


def benchmark_memory_with_copy():
    """Benchmark: Memory usage with .copy() anti-pattern."""
    print("\n" + "="*80)
    print("BENCHMARK: Memory Usage (With vs Without .copy())")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple test files
        test_files = []
        for i in range(5):
            test_file = Path(tmpdir) / f"lora{i}.safetensors"
            create_mock_lora_file(test_file, size_mb=30)
            test_files.append(test_file)

        # Load all files
        loras = {}
        for f in test_files:
            loras[f.stem] = safetensors.torch.load_file(f, device="cpu")

        mem_before = get_memory_usage_mb()

        # Old approach: with .copy()
        copies = {}
        for name, lora in loras.items():
            copies[name] = lora.copy()

        mem_with_copy = get_memory_usage_mb()

        # Clear copies
        del copies

        # New approach: without .copy()
        no_copies = {}
        for name, lora in loras.items():
            no_copies[name] = lora  # No copy!

        mem_without_copy = get_memory_usage_mb()

        copy_overhead = mem_with_copy - mem_before
        no_copy_overhead = mem_without_copy - mem_before

        print(f"Memory before:                     {mem_before:.1f} MB")
        print(f"Memory with .copy():               {mem_with_copy:.1f} MB  (+{copy_overhead:.1f} MB)")
        print(f"Memory without .copy():            {mem_without_copy:.1f} MB  (+{no_copy_overhead:.1f} MB)")
        print(f"Memory saved by removing .copy():  {copy_overhead - no_copy_overhead:.1f} MB")

        if copy_overhead > no_copy_overhead * 1.5:
            print(f"✅ PASS: Removed .copy() saves {copy_overhead - no_copy_overhead:.1f} MB")
        else:
            print("⚠️  Test inconclusive (small difference)")


def benchmark_streaming_vs_direct():
    """Benchmark: Streaming vs direct loading for large files."""
    print("\n" + "="*80)
    print("BENCHMARK: Streaming vs Direct Loading (Large File)")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create large file (200MB)
        large_file = Path(tmpdir) / "large.safetensors"
        create_mock_lora_file(large_file, size_mb=200)

        # Direct loading
        start = time.time()
        state_dict = load_safetensors_robust(
            large_file,
            device="cpu",
            use_streaming=False  # Force direct
        )
        direct_time = time.time() - start
        direct_mem = get_memory_usage_mb()

        # Clear memory
        del state_dict

        # Streaming loading
        start = time.time()
        state_dict = load_safetensors_robust(
            large_file,
            device="cpu",
            use_streaming=True  # Force streaming
        )
        streaming_time = time.time() - start
        streaming_mem = get_memory_usage_mb()

        print(f"Direct loading:    {direct_time:.3f}s  (peak mem: {direct_mem:.1f} MB)")
        print(f"Streaming loading: {streaming_time:.3f}s  (peak mem: {streaming_mem:.1f} MB)")

        if streaming_time <= direct_time * 1.2:
            print(f"✅ PASS: Streaming competitive with direct ({streaming_time:.3f}s vs {direct_time:.3f}s)")
        else:
            print(f"⚠️  WARNING: Streaming significantly slower")


def benchmark_batch_loading():
    """Benchmark: Batch loading vs sequential loading."""
    print("\n" + "="*80)
    print("BENCHMARK: Batch Loading Performance")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple files
        test_files = []
        for i in range(10):
            test_file = Path(tmpdir) / f"lora{i}.safetensors"
            create_mock_lora_file(test_file, size_mb=20)
            test_files.append(test_file)

        # Sequential loading (old approach)
        start = time.time()
        loras_sequential = {}
        for f in test_files:
            loras_sequential[f.stem] = safetensors.torch.load_file(f, device="cpu")
        sequential_time = time.time() - start

        # Batch loading (new approach)
        start = time.time()
        loras_batch = load_loras_batch(test_files, device="cpu", stop_on_error=False)
        batch_time = time.time() - start

        print(f"Sequential loading: {sequential_time:.3f}s for {len(test_files)} files")
        print(f"Batch loading:      {batch_time:.3f}s for {len(test_files)} files")
        print(f"Per file (seq):     {sequential_time/len(test_files):.3f}s")
        print(f"Per file (batch):   {batch_time/len(test_files):.3f}s")

        if batch_time <= sequential_time:
            print(f"✅ PASS: Batch loading faster or equal")
        else:
            print(f"⚠️  Batch loading slower (overhead from error handling/logging)")


def benchmark_caching():
    """Benchmark: Caching benefits."""
    print("\n" + "="*80)
    print("BENCHMARK: Caching Performance")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file
        test_file = Path(tmpdir) / "test.safetensors"
        create_mock_lora_file(test_file, size_mb=50)

        # Clear cache before test
        clear_lora_cache()

        # First load (cold cache)
        start = time.time()
        state_dict1 = load_safetensors_cached(str(test_file), "cpu")
        cold_time = time.time() - start

        # Second load (warm cache)
        start = time.time()
        state_dict2 = load_safetensors_cached(str(test_file), "cpu")
        warm_time = time.time() - start

        # Third load (still cached)
        start = time.time()
        state_dict3 = load_safetensors_cached(str(test_file), "cpu")
        warm_time2 = time.time() - start

        print(f"First load (cold cache):   {cold_time:.3f}s")
        print(f"Second load (warm cache):  {warm_time:.6f}s")
        print(f"Third load (warm cache):   {warm_time2:.6f}s")
        print(f"Speedup:                   {cold_time/warm_time:.0f}x faster")

        if warm_time < cold_time * 0.01:  # Cache should be >100x faster
            print(f"✅ PASS: Caching provides {cold_time/warm_time:.0f}x speedup")
        else:
            print("⚠️  WARNING: Caching not working as expected")


def benchmark_error_handling_overhead():
    """Benchmark: Error handling overhead."""
    print("\n" + "="*80)
    print("BENCHMARK: Error Handling Overhead")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file
        test_file = Path(tmpdir) / "test.safetensors"
        create_mock_lora_file(test_file, size_mb=30)

        # Old: no validation
        start = time.time()
        for _ in range(10):
            state_dict = safetensors.torch.load_file(test_file, device="cpu")
        no_validation_time = (time.time() - start) / 10

        # New: with validation
        start = time.time()
        for _ in range(10):
            state_dict = load_safetensors_robust(test_file, device="cpu", validate=True)
        with_validation_time = (time.time() - start) / 10

        # New: without validation
        start = time.time()
        for _ in range(10):
            state_dict = load_safetensors_robust(test_file, device="cpu", validate=False)
        skip_validation_time = (time.time() - start) / 10

        print(f"No validation (old):       {no_validation_time:.3f}s")
        print(f"With validation (new):     {with_validation_time:.3f}s")
        print(f"Skip validation (new):     {skip_validation_time:.3f}s")
        print(f"Validation overhead:       {((with_validation_time - skip_validation_time) * 1000):.1f}ms")

        if with_validation_time <= no_validation_time * 1.1:
            print(f"✅ PASS: Validation overhead minimal (<10%)")
        else:
            print(f"⚠️  Validation adds {((with_validation_time/no_validation_time - 1)*100):.1f}% overhead")


def run_all_benchmarks():
    """Run all benchmarks."""
    print("\n" + "="*80)
    print("SAFETENSORS LOADING PERFORMANCE BENCHMARKS")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Memory (start): {get_memory_usage_mb():.1f} MB")

    try:
        benchmark_loading_speed()
        benchmark_memory_with_copy()
        benchmark_streaming_vs_direct()
        benchmark_batch_loading()
        benchmark_caching()
        benchmark_error_handling_overhead()

    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print(f"Memory (end): {get_memory_usage_mb():.1f} MB")
    print("="*80)


if __name__ == "__main__":
    # Check if psutil is available
    try:
        import psutil
    except ImportError:
        print("⚠️  psutil not installed. Memory benchmarks will be skipped.")
        print("   Install with: pip install psutil")
        print()

    run_all_benchmarks()
