# Direct GPU loading utilities for OneDiffX
# Loads LoRA files directly to GPU memory, bypassing diffusers' CPU loading

import os
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import torch
import logging

logger = logging.getLogger(__name__)

from .state_dict_utils import convert_state_dict_to_diffusers
from .safetensors_utils import (
    load_safetensors_robust,
    inspect_safetensors_metadata,
    detect_lora_format_from_keys,
    clear_gpu_memory_cache,
    SafetensorsLoadError,
    SafetensorsValidationError,
    SafetensorsCorruptedError,
    DEFAULT_MAX_FILE_SIZE_GB
)


def load_lora_direct(
    pretrained_model_name_or_path_or_dict: Union[str, Path, Dict[str, torch.Tensor]],
    device: Union[str, torch.device] = "cuda",
    device_map: Optional[Union[str, Dict[str, str]]] = None,
    weight_name: Optional[str] = None,
    subfolder: Optional[str] = None,
    unet_config: Optional[Dict] = None,
    **kwargs
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Load a LoRA directly to the specified device, bypassing diffusers' CPU loading.

    This function loads safetensors files directly to GPU memory, then auto-detects
    the format (Kohya, PEFT, diffusers old/new) and converts as needed.

    **NEW:** Supports device_map for distributed loading and async loading!

    Args:
        pretrained_model_name_or_path_or_dict: Path to the LoRA file or directory
        device: Target device for tensor loading (default: "cuda")
        device_map: Optional device mapping for distributed loading (NEW)
                   - Simple string: "cuda:0"
                   - Dict: {"layer1": "cuda:0", "layer2": "disk"}
        weight_name: Specific weight file name (e.g., "pytorch_lora_weights.safetensors")
        subfolder: Subfolder within the model directory
        unet_config: UNet configuration (passed to conversion if needed)
        **kwargs: Additional arguments (strategy, use_async, etc.)

    Returns:
        Tuple of (state_dict, network_alphas)

    Raises:
        ValueError: If file not found or loading fails
    """
    # If already a dict, just run through conversion
    if isinstance(pretrained_model_name_or_path_or_dict, dict):
        logger.info(f"[OneDiffX Direct] Converting provided state dict")
        state_dict = pretrained_model_name_or_path_or_dict
        
        # Convert format if needed (auto-detects format)
        converted_state_dict = convert_state_dict_to_diffusers(state_dict, **kwargs)
        network_alphas = kwargs.get("_network_alphas", {})
        
        return converted_state_dict, network_alphas
    
    # Convert to Path object
    model_path = Path(pretrained_model_name_or_path_or_dict)
    
    # Handle subfolder
    if subfolder:
        model_path = model_path / subfolder
    
    # Determine the actual file to load
    if weight_name:
        # Use specified weight file
        lora_file = model_path / weight_name if model_path.is_dir() else model_path.parent / weight_name
    else:
        # Auto-detect safetensors file
        if model_path.is_file() and model_path.suffix == ".safetensors":
            lora_file = model_path
        elif model_path.is_dir():
            # Look for safetensors files in the directory
            safetensors_files = list(model_path.glob("*.safetensors"))
            if not safetensors_files:
                raise ValueError(f"No .safetensors files found in {model_path}")
            
            # Prefer files with common LoRA names
            for preferred_name in ["pytorch_lora_weights.safetensors", "lora_weights.safetensors"]:
                for f in safetensors_files:
                    if f.name == preferred_name:
                        lora_file = f
                        break
                else:
                    continue
                break
            else:
                # Fallback: prefer files with "lora" in the name
                lora_files = [f for f in safetensors_files if "lora" in f.name.lower()]
                if lora_files:
                    lora_file = lora_files[0]
                else:
                    lora_file = safetensors_files[0]
            
            if len(safetensors_files) > 1:
                logger.warning(
                    f"Multiple safetensors files found in {model_path}. "
                    f"Using {lora_file.name}. Specify weight_name to use a different file."
                )
        else:
            raise ValueError(f"Invalid path: {model_path} is neither a file nor a directory")
    
    # Check if file exists
    if not lora_file.exists():
        raise ValueError(f"LoRA file not found: {lora_file}")
    
    # Load using robust loader with error handling, validation, and device_map support
    state_dict = load_safetensors_robust(
        lora_file=lora_file,
        device=device,  # No str() conversion - supports Path and torch.device directly
        device_map=device_map,  # NEW: Pass device_map for distributed loading
        strategy=kwargs.get("strategy", "auto"),  # NEW: Strategy selection
        use_async=kwargs.get("use_async", False),  # NEW: Async support
        validate=True,
        max_size_gb=kwargs.get("max_size_gb", DEFAULT_MAX_FILE_SIZE_GB),
        use_streaming=kwargs.get("use_streaming", None)  # DEPRECATED
    )
    
    # Convert format if needed (auto-detects format: Kohya, PEFT, diffusers old/new)
    # Pass unet_config if provided
    if unet_config is not None:
        kwargs["unet_config"] = unet_config
        
    converted_state_dict = convert_state_dict_to_diffusers(state_dict, **kwargs)
    
    # Extract network_alphas if they were set during conversion (e.g., for Kohya format)
    network_alphas = kwargs.get("_network_alphas", {})
    
    return converted_state_dict, network_alphas


def should_use_direct_loader(
    pretrained_model_name_or_path_or_dict: Union[str, Path, Dict[str, torch.Tensor]],
    **kwargs
) -> bool:
    """
    Determine if we should use the direct loader instead of diffusers.
    
    Returns True if:
    - Input is a string or Path (not a dict)
    - Points to a .safetensors file or directory containing .safetensors files
    - User hasn't explicitly disabled it via use_direct_loader=False
    """
    # Check if explicitly disabled
    if kwargs.get("use_direct_loader", True) is False:
        return False
    
    # Only consider file paths, not dicts
    if isinstance(pretrained_model_name_or_path_or_dict, dict):
        return False
    
    # Convert to Path
    try:
        path = Path(pretrained_model_name_or_path_or_dict)
    except:
        # If path conversion fails, don't use direct loader
        return False
    
    # Check if it's a safetensors file
    if path.is_file() and path.suffix == ".safetensors":
        return True
    
    # Check if it's a directory containing safetensors files
    if path.is_dir():
        safetensors_files = list(path.glob("*.safetensors"))
        return len(safetensors_files) > 0
    
    return False