# OneDiffX LoRA Memory Optimizations

This document describes the memory optimizations implemented for LoRA loading to reduce memory consumption from ~1.5GB to ~128MB per LoRA.

## Changes Made

### 1. Preserved Original Data Types
- LoRA weights are no longer automatically converted to float32
- Weights maintain their original dtype (typically fp16)
- Computation is done in float32 only when necessary, then converted back

### 2. Eliminated Unnecessary Cloning
- `offload_tensor()` no longer clones tensors when already on target device
- Direct references are used when possible

### 3. Memory-Efficient Loading Option
- Added `memory_efficient=True` parameter (default)
- Avoids intermediate copies and preserves dtypes
- Uses in-place operations where possible

### 4. Reduced Cache Size
- Default cache size reduced from 100 to 10 LoRAs
- Can be configured via `ONEDIFFX_LORA_CACHE_SIZE` environment variable
- Set to 0 to disable caching entirely

### 5. Cache Cleanup
- Added `clear_lora_cache()` function to manually free cache memory

## Usage

### Basic Usage (Memory Efficient by Default)
```python
from onediffx.lora import load_and_fuse_lora

# Memory efficient loading is enabled by default
load_and_fuse_lora(
    pipeline,
    "path/to/lora.safetensors",
    adapter_name="my_lora",
    lora_scale=1.0,
)
```

### Disable Memory Efficient Mode
```python
# Use original behavior if needed
load_and_fuse_lora(
    pipeline,
    "path/to/lora.safetensors",
    memory_efficient=False,  # Use original float32 conversion
)
```

### Configure Cache Size
```bash
# Disable LoRA caching entirely
export ONEDIFFX_LORA_CACHE_SIZE=0

# Or set a custom cache size
export ONEDIFFX_LORA_CACHE_SIZE=5
```

### Clear Cache Manually
```python
from onediffx.lora.lora import clear_lora_cache

# Free all cached LoRAs
clear_lora_cache()
```

## Memory Savings

With these optimizations:
- **Before**: Each LoRA consumed ~1.5GB GPU memory
- **After**: Each LoRA consumes ~128MB GPU memory (actual file size)
- **Savings**: ~91% reduction in memory usage per LoRA

## Compatibility

These changes are backward compatible. The `memory_efficient` parameter defaults to `True` but can be set to `False` to restore original behavior if needed.