# Kohya format LoRA conversion utilities for OneDiffX
# Adapted from diffusers to preserve GPU tensors throughout conversion

import os
import re
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import torch
import safetensors.torch
from onediff.utils import logger


def convert_kohya_state_dict_to_diffusers(
    state_dict: Dict[str, torch.Tensor],
    unet_name: str = "unet",
    text_encoder_name: str = "text_encoder"
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Converts a Kohya (non-Diffusers) LoRA state dict to Diffusers compatible format.
    Preserves tensor devices throughout the conversion process.
    
    Args:
        state_dict: The Kohya format state dict to convert
        unet_name: Name prefix for UNet modules (default: "unet")
        text_encoder_name: Name prefix for text encoder modules (default: "text_encoder")
        
    Returns:
        Tuple of (converted_state_dict, network_alphas)
    """
    unet_state_dict = {}
    te_state_dict = {}
    te2_state_dict = {}
    network_alphas = {}
    
    # Create a copy to avoid modifying the original
    remaining_keys = dict(state_dict)
    
    # Process all lora_down.weight keys
    for key in list(remaining_keys.keys()):
        if not key.endswith("lora_down.weight"):
            continue
            
        # Extract LoRA name
        lora_name = key.split(".")[0]
        
        # Find corresponding up weight and alpha
        lora_name_up = lora_name + ".lora_up.weight"
        lora_name_alpha = lora_name + ".alpha"
        
        if lora_name_up not in remaining_keys:
            logger.warning(f"Found {key} but missing {lora_name_up}, skipping")
            continue
        
        # Handle U-Net LoRAs
        if lora_name.startswith("lora_unet_"):
            diffusers_name = _convert_unet_lora_key(key)
            
            # Transfer weights without changing device
            unet_state_dict[diffusers_name] = remaining_keys.pop(key)
            unet_state_dict[diffusers_name.replace(".down.", ".up.")] = remaining_keys.pop(lora_name_up)
            
        # Handle text encoder LoRAs
        elif lora_name.startswith(("lora_te_", "lora_te1_", "lora_te2_")):
            diffusers_name = _convert_text_encoder_lora_key(key, lora_name)
            
            # Store in appropriate text encoder dict
            if lora_name.startswith(("lora_te_", "lora_te1_")):
                te_state_dict[diffusers_name] = remaining_keys.pop(key)
                te_state_dict[diffusers_name.replace(".down.", ".up.")] = remaining_keys.pop(lora_name_up)
            else:
                te2_state_dict[diffusers_name] = remaining_keys.pop(key)
                te2_state_dict[diffusers_name.replace(".down.", ".up.")] = remaining_keys.pop(lora_name_up)
        
        # Handle alpha if present (extract scalar without forcing to CPU)
        if lora_name_alpha in remaining_keys:
            alpha_tensor = remaining_keys.pop(lora_name_alpha)
            # Use item() to get scalar but tensor stays on original device
            alpha_value = alpha_tensor.item() if alpha_tensor.numel() == 1 else float(alpha_tensor)
            network_alphas.update(_get_alpha_name(lora_name_alpha, diffusers_name, alpha_value))
    
    # Remove any processed keys that weren't popped
    for key in list(remaining_keys.keys()):
        if key.endswith((".lora_up.weight", ".alpha")) and any(
            key.startswith(prefix) for prefix in ["lora_unet_", "lora_te_", "lora_te1_", "lora_te2_"]
        ):
            remaining_keys.pop(key)
    
    # Check for unprocessed keys
    if remaining_keys:
        unprocessed = [k for k in remaining_keys.keys() if "lora" in k]
        if unprocessed:
            logger.warning(f"Unprocessed LoRA keys: {unprocessed}")
    
    # Add module prefixes
    unet_state_dict = {f"{unet_name}.{k}": v for k, v in unet_state_dict.items()}
    te_state_dict = {f"{text_encoder_name}.{k}": v for k, v in te_state_dict.items()}
    if te2_state_dict:
        te2_state_dict = {f"text_encoder_2.{k}": v for k, v in te2_state_dict.items()}
        te_state_dict.update(te2_state_dict)
    
    # Combine all state dicts
    new_state_dict = {**unet_state_dict, **te_state_dict}
    
    logger.info(f"Converted Kohya LoRA: {len(new_state_dict)} weights, {len(network_alphas)} alphas")
    
    return new_state_dict, network_alphas


def _convert_unet_lora_key(key: str) -> str:
    """Convert UNet LoRA key from Kohya to Diffusers format."""
    # Remove prefix
    diffusers_name = key.replace("lora_unet_", "")
    
    # Based on Kohya scripts source code:
    # input_blocks.{3*i + j + 1}.0 → down_blocks.{i}.resnets.{j}
    # input_blocks.{3*i + j + 1}.1 → down_blocks.{i}.attentions.{j}
    # input_blocks.{3*(i+1)}.0.op → down_blocks.{i}.downsamplers.0.conv
    
    # Check for downsampler pattern first (before splitting)
    # Downsamplers are at input_blocks.3.0.op, input_blocks.6.0.op, input_blocks.9.0.op
    is_downsampler = False
    for i, block_num in enumerate([3, 6, 9]):
        if diffusers_name.startswith(f"input_blocks_{block_num}_0_op"):
            # This is a downsampler
            new_prefix = f"down_blocks.{i}.downsamplers.0.conv"
            # Get the suffix after "op"
            suffix = diffusers_name[len(f"input_blocks_{block_num}_0_op"):]
            diffusers_name = new_prefix + suffix
            is_downsampler = True
            break
    
    # Handle input_blocks_X_Y pattern (if not already handled as downsampler)
    if not is_downsampler and diffusers_name.startswith("input_blocks_"):
        parts = diffusers_name.split("_")
        if len(parts) >= 4 and parts[2].isdigit() and parts[3].isdigit():
            input_block_num = int(parts[2])
            inner_block_type = int(parts[3])
            
            # Special case for input_blocks.0.0 (conv_in)
            if input_block_num == 0 and inner_block_type == 0:
                new_prefix = "conv_in"
            else:
                # Regular blocks: compute block_id and layer_in_block
                # Formula: input_blocks.{3*i + j + 1} where i is block, j is layer
                block_id = (input_block_num - 1) // 3
                layer_in_block = (input_block_num - 1) % 3
                
                if inner_block_type == 0:
                    new_prefix = f"down_blocks.{block_id}.resnets.{layer_in_block}"
                elif inner_block_type == 1:
                    new_prefix = f"down_blocks.{block_id}.attentions.{layer_in_block}"
                else:
                    # Fallback
                    new_prefix = f"down_blocks.{block_id}.{inner_block_type}.{layer_in_block}"
            
            # Replace the prefix and reconstruct
            diffusers_name = new_prefix + "." + "_".join(parts[4:])
    
    # Handle output_blocks_X_Y pattern
    elif diffusers_name.startswith("output_blocks_"):
        parts = diffusers_name.split("_")
        if len(parts) >= 4 and parts[2].isdigit() and parts[3].isdigit():
            output_block_num = int(parts[2])
            inner_block_type = int(parts[3])
            
            # Compute block_id and layer_in_block
            block_id = output_block_num // 3
            layer_in_block = output_block_num % 3
            
            # Upsamplers are at positions 2, 5, 8 (layer_in_block == 2)
            if layer_in_block == 2 and inner_block_type == 2:
                new_prefix = f"up_blocks.{block_id}.upsamplers.0"
            else:
                if inner_block_type == 0:
                    new_prefix = f"up_blocks.{block_id}.resnets.{layer_in_block}"
                elif inner_block_type == 1:
                    new_prefix = f"up_blocks.{block_id}.attentions.{layer_in_block}"
                else:
                    # Fallback
                    new_prefix = f"up_blocks.{block_id}.{inner_block_type}.{layer_in_block}"
            
            # Replace the prefix and reconstruct
            # For upsamplers, handle potential upsampler conversion
            if "upsamplers" in new_prefix:
                diffusers_name = new_prefix + "." + "_".join(parts[4:])
            else:
                diffusers_name = new_prefix + "." + "_".join(parts[4:])
    
    # Handle middle_block_X pattern
    elif diffusers_name.startswith("middle_block_"):
        parts = diffusers_name.split("_")
        if len(parts) >= 3 and parts[2].isdigit():
            middle_id = int(parts[2])
            
            # From Kohya source: middle_block.0 → mid_block.resnets.0
            # middle_block.1 → mid_block.attentions.0
            # middle_block.2 → mid_block.resnets.1
            if middle_id == 0:
                new_prefix = "mid_block.resnets.0"
            elif middle_id == 1:
                new_prefix = "mid_block.attentions.0"
            elif middle_id == 2:
                new_prefix = "mid_block.resnets.1"
            else:
                new_prefix = f"mid_block.{middle_id}"  # fallback
            
            # Replace the prefix and reconstruct
            diffusers_name = new_prefix + "." + "_".join(parts[3:])
    
    # Handle time_embed pattern before converting underscores
    if diffusers_name.startswith("time_embed_"):
        parts = diffusers_name.split("_")
        if len(parts) >= 3 and parts[2].isdigit():
            time_id = int(parts[2])
            # time_embed.0 → time_embedding.linear_1
            # time_embed.2 → time_embedding.linear_2
            linear_id = (time_id // 2) + 1
            new_prefix = f"time_embedding.linear_{linear_id}"
            diffusers_name = new_prefix + "." + "_".join(parts[3:])
    
    # Handle label_emb pattern before converting underscores
    elif diffusers_name.startswith("label_emb_0_"):
        parts = diffusers_name.split("_")
        if len(parts) >= 4 and parts[3].isdigit():
            label_id = int(parts[3])
            # label_emb.0.0 → add_embedding.linear_1
            # label_emb.0.2 → add_embedding.linear_2
            linear_id = (label_id // 2) + 1
            new_prefix = f"add_embedding.linear_{linear_id}"
            diffusers_name = new_prefix + "." + "_".join(parts[4:])
    
    # Handle output conversion layers
    elif diffusers_name.startswith("out_"):
        parts = diffusers_name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            out_id = int(parts[1])
            if out_id == 0:
                new_prefix = "conv_norm_out"
            elif out_id == 2:
                new_prefix = "conv_out"
            else:
                new_prefix = f"out.{out_id}"
            diffusers_name = new_prefix + "." + "_".join(parts[2:])
    
    # Now convert remaining underscores to dots
    diffusers_name = diffusers_name.replace("_", ".")
    
    # Restore module names that must have underscores (not dots)
    diffusers_name = diffusers_name.replace("down.blocks", "down_blocks")
    diffusers_name = diffusers_name.replace("up.blocks", "up_blocks")
    diffusers_name = diffusers_name.replace("mid.block", "mid_block")
    diffusers_name = diffusers_name.replace("time.embedding", "time_embedding")
    diffusers_name = diffusers_name.replace("add.embedding", "add_embedding")
    diffusers_name = diffusers_name.replace("conv.norm.out", "conv_norm_out")
    diffusers_name = diffusers_name.replace("conv.in", "conv_in")
    diffusers_name = diffusers_name.replace("conv.out", "conv_out")
    
    # ResNet layer conversions (based on Kohya source)
    diffusers_name = diffusers_name.replace("in.layers.0.", "norm1.")
    diffusers_name = diffusers_name.replace("in.layers.2.", "conv1.")
    diffusers_name = diffusers_name.replace("out.layers.0.", "norm2.")
    diffusers_name = diffusers_name.replace("out.layers.3.", "conv2.")
    diffusers_name = diffusers_name.replace("emb.layers.1.", "time_emb_proj.")
    diffusers_name = diffusers_name.replace("skip.connection.", "conv_shortcut.")
    
    # Standard conversions
    diffusers_name = diffusers_name.replace("transformer.blocks", "transformer_blocks")
    
    # Attention layer conversions
    diffusers_name = diffusers_name.replace("to.q.lora", "to_q_lora")
    diffusers_name = diffusers_name.replace("to.k.lora", "to_k_lora")
    diffusers_name = diffusers_name.replace("to.v.lora", "to_v_lora")
    diffusers_name = diffusers_name.replace("to.out.0.lora", "to_out_lora")
    
    # Projection conversions
    diffusers_name = diffusers_name.replace("proj.in", "proj_in")
    diffusers_name = diffusers_name.replace("proj.out", "proj_out")
    
    # Downsampler/upsampler op → conv conversion
    if "downsamplers" in diffusers_name or "upsamplers" in diffusers_name:
        diffusers_name = diffusers_name.replace(".op.", ".conv.")
    
    # SDXL specific time embedding handling
    if "emb" in diffusers_name and "time.emb.proj" not in diffusers_name and "time_emb_proj" not in diffusers_name:
        # Remove trailing number from emb layers
        pattern = r"\.\d+(?=\D*$)"
        diffusers_name = re.sub(pattern, "", diffusers_name, count=1)
    
    # LyCORIS specific
    if "time.emb.proj" in diffusers_name:
        diffusers_name = diffusers_name.replace("time.emb.proj", "time_emb_proj")
    if "conv.shortcut" in diffusers_name:
        diffusers_name = diffusers_name.replace("conv.shortcut", "conv_shortcut")
    
    # Transformer block specific
    if "transformer_blocks" in diffusers_name:
        if "attn1" in diffusers_name or "attn2" in diffusers_name:
            diffusers_name = diffusers_name.replace("attn1", "attn1.processor")
            diffusers_name = diffusers_name.replace("attn2", "attn2.processor")
    
    return diffusers_name


def _convert_text_encoder_lora_key(key: str, lora_name: str) -> str:
    """Convert text encoder LoRA key from Kohya to Diffusers format."""
    # Determine which prefix to remove
    if lora_name.startswith(("lora_te_", "lora_te1_")):
        key_to_replace = "lora_te_" if lora_name.startswith("lora_te_") else "lora_te1_"
    else:
        key_to_replace = "lora_te2_"
    
    # Remove prefix and convert underscores
    diffusers_name = key.replace(key_to_replace, "").replace("_", ".")
    
    # Text model conversions
    diffusers_name = diffusers_name.replace("text.model", "text_model")
    diffusers_name = diffusers_name.replace("self.attn", "self_attn")
    
    # Projection conversions
    diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
    diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
    diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
    diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
    diffusers_name = diffusers_name.replace("text.projection", "text_projection")
    
    # MLP layers need special handling
    if "mlp" in diffusers_name and "self_attn" not in diffusers_name and "text_projection" not in diffusers_name:
        diffusers_name = diffusers_name.replace(".lora.", ".lora_linear_layer.")
    
    return diffusers_name


def _get_alpha_name(lora_name_alpha: str, diffusers_name: str, alpha: float) -> Dict[str, float]:
    """Get the correct alpha name for the Diffusers model."""
    # Determine module prefix based on original key
    if lora_name_alpha.startswith("lora_unet_"):
        prefix = "unet."
    elif lora_name_alpha.startswith(("lora_te_", "lora_te1_")):
        prefix = "text_encoder."
    else:
        prefix = "text_encoder_2."
    
    # Extract base name before .lora
    base_name = diffusers_name.split(".lora.")[0] if ".lora." in diffusers_name else diffusers_name.rsplit(".", 2)[0]
    new_name = prefix + base_name + ".alpha"
    
    return {new_name: alpha}


def is_kohya_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    """
    Check if a state dict is in Kohya format.
    
    Kohya format characteristics:
    - Keys start with lora_unet_, lora_te_, lora_te1_, or lora_te2_
    - Keys end with .lora_down.weight, .lora_up.weight, or .alpha
    """
    kohya_prefixes = ("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_")
    kohya_suffixes = (".lora_down.weight", ".lora_up.weight", ".alpha")
    
    for key in state_dict.keys():
        if any(key.startswith(prefix) for prefix in kohya_prefixes):
            if any(key.endswith(suffix) for suffix in kohya_suffixes):
                return True
    
    return False


def load_kohya_lora_direct(
    pretrained_model_name_or_path_or_dict: Union[str, Path, Dict[str, torch.Tensor]],
    device: Union[str, torch.device] = "cuda",
    weight_name: Optional[str] = None,
    subfolder: Optional[str] = None,
    **kwargs
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Load a Kohya format LoRA directly to the specified device, bypassing diffusers.
    
    This function loads safetensors files directly to GPU memory, avoiding the CPU
    memory overhead of diffusers' loading process.
    
    Args:
        pretrained_model_name_or_path_or_dict: Path to the LoRA file or directory
        device: Target device for tensor loading (default: "cuda")
        weight_name: Specific weight file name (e.g., "pytorch_lora_weights.safetensors")
        subfolder: Subfolder within the model directory
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        Tuple of (converted_state_dict, network_alphas)
        
    Raises:
        ValueError: If the file is not in Kohya format or file not found
    """
    # If already a dict, check format and convert
    if isinstance(pretrained_model_name_or_path_or_dict, dict):
        state_dict = pretrained_model_name_or_path_or_dict
        if not is_kohya_state_dict(state_dict):
            raise ValueError("Provided state dict is not in Kohya format")
        logger.info(f"[OneDiffX] Converting Kohya dict to diffusers format")
        return convert_kohya_state_dict_to_diffusers(state_dict)
    
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
            
            # Prefer files with "lora" in the name
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
    
    logger.info(f"[OneDiffX] Loading Kohya LoRA directly from {lora_file} to {device}")
    
    # Load directly to the target device
    # Use safetensors to load directly to device
    state_dict = safetensors.torch.load_file(str(lora_file), device=str(device))
    
    # Verify it's Kohya format
    if not is_kohya_state_dict(state_dict):
        # Check first few keys for debugging
        first_keys = list(state_dict.keys())[:5]
        logger.error(f"Not a Kohya format LoRA. First keys: {first_keys}")
        raise ValueError(
            f"File {lora_file} is not in Kohya format. "
            "This direct loader only supports Kohya format LoRAs."
        )
    
    # Convert to diffusers format (tensors remain on target device)
    converted_state_dict, network_alphas = convert_kohya_state_dict_to_diffusers(state_dict)
    
    return converted_state_dict, network_alphas