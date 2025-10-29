"""
Comprehensive unit tests for onediffx.lora.safetensors_utils module.

Tests cover:
- Error handling for all exception types
- File validation logic
- Metadata inspection
- Format detection
- Batch loading
- Caching
- Memory management
- Edge cases
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import torch

from onediffx.lora.safetensors_utils import (
    # Exceptions
    SafetensorsLoadError,
    SafetensorsValidationError,
    SafetensorsCorruptedError,
    # Validation
    validate_safetensors_path,
    # Loading
    load_safetensors_robust,
    load_loras_batch,
    # Inspection
    inspect_safetensors_metadata,
    detect_lora_format_from_keys,
    # Caching
    load_safetensors_cached,
    clear_lora_cache,
    # Memory
    clear_gpu_memory_cache,
    # Constants
    DEFAULT_MAX_FILE_SIZE_GB,
    MIN_FILE_SIZE_BYTES,
    LARGE_FILE_THRESHOLD_MB,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_safetensors_file():
    """Create a temporary mock safetensors file."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        # Write minimal valid data (>100 bytes)
        f.write(b"X" * 1000)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_empty_file():
    """Create an empty safetensors file."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        temp_path = Path(f.name)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_large_file():
    """Create a large file (>100MB threshold)."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        # Write 150MB of data
        chunk_size = 1024 * 1024  # 1MB chunks
        for _ in range(150):
            f.write(b"X" * chunk_size)
        temp_path = Path(f.name)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def mock_state_dict():
    """Create a mock state dict with tensors."""
    return {
        "layer1.weight": torch.randn(10, 10),
        "layer2.weight": torch.randn(5, 5),
        "layer3.bias": torch.randn(5),
    }


# ============================================================================
# Test: Exception Classes
# ============================================================================

def test_exception_hierarchy():
    """Test exception class hierarchy."""
    assert issubclass(SafetensorsValidationError, SafetensorsLoadError)
    assert issubclass(SafetensorsCorruptedError, SafetensorsLoadError)
    assert issubclass(SafetensorsLoadError, Exception)


def test_exception_messages():
    """Test exception messages are preserved."""
    msg = "Test error message"

    exc = SafetensorsLoadError(msg)
    assert str(exc) == msg

    exc = SafetensorsValidationError(msg)
    assert str(exc) == msg

    exc = SafetensorsCorruptedError(msg)
    assert str(exc) == msg


# ============================================================================
# Test: validate_safetensors_path
# ============================================================================

def test_validate_safetensors_path_success(temp_safetensors_file):
    """Test successful path validation."""
    resolved, size_mb = validate_safetensors_path(temp_safetensors_file)

    assert resolved.exists()
    assert resolved.is_absolute()
    assert resolved.suffix == ".safetensors"
    assert size_mb > 0


def test_validate_safetensors_path_nonexistent():
    """Test validation fails for nonexistent file."""
    with pytest.raises(FileNotFoundError):
        validate_safetensors_path(Path("/nonexistent/file.safetensors"))


def test_validate_safetensors_path_empty_file(temp_empty_file):
    """Test validation fails for empty file."""
    with pytest.raises(SafetensorsValidationError, match="too small"):
        validate_safetensors_path(temp_empty_file)


def test_validate_safetensors_path_wrong_extension(temp_safetensors_file):
    """Test validation fails for wrong extension."""
    # Rename to wrong extension
    wrong_ext = temp_safetensors_file.with_suffix(".txt")
    temp_safetensors_file.rename(wrong_ext)

    try:
        with pytest.raises(SafetensorsValidationError, match="Invalid file extension"):
            validate_safetensors_path(wrong_ext)
    finally:
        if wrong_ext.exists():
            wrong_ext.unlink()


def test_validate_safetensors_path_size_limit(temp_large_file):
    """Test validation fails when file exceeds size limit."""
    with pytest.raises(SafetensorsValidationError, match="too large"):
        # Set very small limit (0.1GB = 100MB, file is 150MB)
        validate_safetensors_path(temp_large_file, max_size_gb=0.1)


def test_validate_safetensors_path_skip_existence_check(temp_safetensors_file):
    """Test validation can skip existence check."""
    # Delete the file
    temp_safetensors_file.unlink()

    # Should not raise if check_exists=False
    # (will fail with stat error, which is expected)
    with pytest.raises(SafetensorsValidationError):
        validate_safetensors_path(temp_safetensors_file, check_exists=False)


# ============================================================================
# Test: detect_lora_format_from_keys
# ============================================================================

def test_detect_lora_format_kohya():
    """Test detection of Kohya format."""
    keys = [
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight",
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight",
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.alpha",
    ]
    assert detect_lora_format_from_keys(keys) == "kohya"


def test_detect_lora_format_diffusers_old():
    """Test detection of old diffusers format."""
    keys = [
        "unet.down_blocks.0.attentions.0.to_out_lora.up.weight",
        "unet.down_blocks.0.attentions.0.to_out_lora.down.weight",
    ]
    assert detect_lora_format_from_keys(keys) == "diffusers_old"


def test_detect_lora_format_peft():
    """Test detection of PEFT format."""
    keys = [
        "unet.down_blocks.0.attentions.0.to_q.lora_A.weight",
        "unet.down_blocks.0.attentions.0.to_q.lora_B.weight",
    ]
    assert detect_lora_format_from_keys(keys) == "peft"


def test_detect_lora_format_diffusers():
    """Test detection of new diffusers format."""
    keys = [
        "unet.down_blocks.0.attentions.0.to_q.lora_linear_layer.up.weight",
        "unet.down_blocks.0.attentions.0.to_q.lora_linear_layer.down.weight",
    ]
    assert detect_lora_format_from_keys(keys) == "diffusers"


def test_detect_lora_format_unknown():
    """Test detection returns unknown for unrecognized format."""
    keys = [
        "some.random.key",
        "another.unknown.key",
    ]
    assert detect_lora_format_from_keys(keys) == "unknown"


# ============================================================================
# Test: inspect_safetensors_metadata (Mocked)
# ============================================================================

@patch('onediffx.lora.safetensors_utils.safe_open')
def test_inspect_safetensors_metadata_success(mock_safe_open, temp_safetensors_file):
    """Test metadata inspection with mocked safe_open."""
    # Setup mock
    mock_file = MagicMock()
    mock_file.metadata.return_value = {"format": "pt"}
    mock_file.keys.return_value = ["layer1.weight", "layer2.weight"]

    mock_slice = MagicMock()
    mock_slice.get_shape.return_value = [10, 10]
    mock_slice.get_dtype.return_value = "float32"
    mock_file.get_slice.return_value = mock_slice

    mock_safe_open.return_value.__enter__.return_value = mock_file

    # Test
    metadata = inspect_safetensors_metadata(temp_safetensors_file)

    assert "metadata" in metadata
    assert "keys" in metadata
    assert "num_tensors" in metadata
    assert metadata["num_tensors"] == 2
    assert len(metadata["sample_keys"]) == 2


@patch('onediffx.lora.safetensors_utils.safe_open')
def test_inspect_safetensors_metadata_corrupted(mock_safe_open, temp_safetensors_file):
    """Test metadata inspection detects corruption."""
    # Simulate corruption error
    mock_safe_open.side_effect = RuntimeError("Corrupted header")

    with pytest.raises(SafetensorsCorruptedError, match="Corrupted"):
        inspect_safetensors_metadata(temp_safetensors_file)


# ============================================================================
# Test: load_safetensors_robust (Mocked)
# ============================================================================

@patch('onediffx.lora.safetensors_utils.safetensors.torch.load_file')
def test_load_safetensors_robust_success(mock_load_file, temp_safetensors_file, mock_state_dict):
    """Test successful robust loading."""
    mock_load_file.return_value = mock_state_dict

    result = load_safetensors_robust(temp_safetensors_file, device="cuda")

    assert result == mock_state_dict
    mock_load_file.assert_called_once()


@patch('onediffx.lora.safetensors_utils.safetensors.torch.load_file')
def test_load_safetensors_robust_oom(mock_load_file, temp_safetensors_file):
    """Test OOM handling."""
    mock_load_file.side_effect = torch.cuda.OutOfMemoryError()

    with pytest.raises(torch.cuda.OutOfMemoryError, match="GPU out of memory"):
        load_safetensors_robust(temp_safetensors_file, device="cuda")


@patch('onediffx.lora.safetensors_utils.safetensors.torch.load_file')
def test_load_safetensors_robust_permission(mock_load_file, temp_safetensors_file):
    """Test permission error handling."""
    mock_load_file.side_effect = PermissionError("Access denied")

    with pytest.raises(SafetensorsLoadError, match="Permission denied"):
        load_safetensors_robust(temp_safetensors_file, device="cuda")


@patch('onediffx.lora.safetensors_utils.safetensors.torch.load_file')
def test_load_safetensors_robust_corruption(mock_load_file, temp_safetensors_file):
    """Test corrupted file detection."""
    mock_load_file.side_effect = RuntimeError("Corrupted data")

    with pytest.raises(SafetensorsCorruptedError, match="Corrupted"):
        load_safetensors_robust(temp_safetensors_file, device="cuda")


@patch('onediffx.lora.safetensors_utils._load_safetensors_streaming')
def test_load_safetensors_robust_streaming_auto(mock_streaming, temp_large_file):
    """Test automatic streaming for large files."""
    mock_streaming.return_value = {"key": torch.randn(10, 10)}

    # Large file should trigger streaming
    result = load_safetensors_robust(
        temp_large_file,
        device="cuda",
        use_streaming=None  # Auto-detect
    )

    mock_streaming.assert_called_once()


@patch('onediffx.lora.safetensors_utils._load_safetensors_streaming')
def test_load_safetensors_robust_streaming_forced(mock_streaming, temp_safetensors_file):
    """Test forced streaming for small files."""
    mock_streaming.return_value = {"key": torch.randn(10, 10)}

    # Small file but force streaming
    result = load_safetensors_robust(
        temp_safetensors_file,
        device="cuda",
        use_streaming=True  # Force
    )

    mock_streaming.assert_called_once()


# ============================================================================
# Test: load_loras_batch (Mocked)
# ============================================================================

@patch('onediffx.lora.safetensors_utils.load_safetensors_robust')
def test_load_loras_batch_success(mock_load, temp_safetensors_file):
    """Test successful batch loading."""
    mock_load.return_value = {"layer1.weight": torch.randn(10, 10)}

    paths = [temp_safetensors_file]
    result = load_loras_batch(paths, device="cuda")

    assert len(result) == 1
    assert temp_safetensors_file.stem in result


@patch('onediffx.lora.safetensors_utils.load_safetensors_robust')
def test_load_loras_batch_partial_failure(mock_load):
    """Test batch loading with partial failures."""
    # First succeeds, second fails, third succeeds
    mock_load.side_effect = [
        {"key": torch.randn(10, 10)},
        RuntimeError("Failed"),
        {"key": torch.randn(10, 10)},
    ]

    paths = [
        Path("lora1.safetensors"),
        Path("lora2.safetensors"),
        Path("lora3.safetensors"),
    ]

    # Should continue and load 2/3
    result = load_loras_batch(paths, device="cuda", stop_on_error=False)

    assert len(result) == 2  # Only successful ones


@patch('onediffx.lora.safetensors_utils.load_safetensors_robust')
def test_load_loras_batch_stop_on_error(mock_load):
    """Test batch loading stops on error when configured."""
    mock_load.side_effect = RuntimeError("Failed")

    paths = [Path("lora1.safetensors")]

    with pytest.raises(RuntimeError, match="Failed"):
        load_loras_batch(paths, device="cuda", stop_on_error=True)


# ============================================================================
# Test: Caching
# ============================================================================

@patch('onediffx.lora.safetensors_utils.load_safetensors_robust')
def test_load_safetensors_cached(mock_load, temp_safetensors_file):
    """Test LRU caching works."""
    mock_load.return_value = {"key": torch.randn(10, 10)}

    # First call
    result1 = load_safetensors_cached(str(temp_safetensors_file), "cuda")

    # Second call should use cache (mock not called again)
    result2 = load_safetensors_cached(str(temp_safetensors_file), "cuda")

    # Should be called only once due to cache
    assert mock_load.call_count == 1

    # Results should be identical (from cache)
    assert result1 is result2


def test_clear_lora_cache():
    """Test cache clearing."""
    # Should not raise
    clear_lora_cache()


# ============================================================================
# Test: Memory Management
# ============================================================================

@patch('torch.cuda.is_available')
@patch('torch.cuda.empty_cache')
def test_clear_gpu_memory_cache(mock_empty_cache, mock_is_available):
    """Test GPU memory cache clearing."""
    mock_is_available.return_value = True

    clear_gpu_memory_cache()

    mock_empty_cache.assert_called_once()


@patch('torch.cuda.is_available')
@patch('torch.cuda.empty_cache')
def test_clear_gpu_memory_cache_no_cuda(mock_empty_cache, mock_is_available):
    """Test GPU cache clearing when CUDA not available."""
    mock_is_available.return_value = False

    clear_gpu_memory_cache()

    # Should not call empty_cache if CUDA not available
    mock_empty_cache.assert_not_called()


# ============================================================================
# Test: Constants
# ============================================================================

def test_constants_values():
    """Test constant values are reasonable."""
    assert DEFAULT_MAX_FILE_SIZE_GB > 0
    assert MIN_FILE_SIZE_BYTES > 0
    assert LARGE_FILE_THRESHOLD_MB > 0
    assert DEFAULT_MAX_FILE_SIZE_GB > LARGE_FILE_THRESHOLD_MB / 1024


# ============================================================================
# Test: Edge Cases
# ============================================================================

def test_device_types_accepted():
    """Test various device type specifications work."""
    # These should all be valid (mocked to not actually load)
    with patch('onediffx.lora.safetensors_utils.safetensors.torch.load_file') as mock:
        mock.return_value = {}

        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            f.write(b"X" * 1000)
            f.flush()
            path = Path(f.name)

            # String device
            load_safetensors_robust(path, device="cuda")

            # torch.device object
            load_safetensors_robust(path, device=torch.device("cpu"))


def test_path_types_accepted():
    """Test various path type specifications work."""
    with patch('onediffx.lora.safetensors_utils.safetensors.torch.load_file') as mock:
        mock.return_value = {}

        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            f.write(b"X" * 1000)
            f.flush()

            # Path object
            load_safetensors_robust(Path(f.name), device="cpu")

            # Note: String paths are converted to Path internally by callers
