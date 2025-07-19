import functools
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import diffusers

import torch
from diffusers.loaders import LoraLoaderMixin
from onediff.utils import logger
from packaging import version

if version.parse(diffusers.__version__) >= version.parse("0.21.0"):
    from diffusers.models.lora import PatchedLoraProjection
else:
    from diffusers.loaders import PatchedLoraProjection

from .text_encoder import load_lora_into_text_encoder
from .unet import load_lora_into_unet
from .utils import (
    _delete_adapter,
    _maybe_map_sgm_blocks_to_diffusers,
    _set_adapter,
    _unfuse_lora,
    is_peft_available,
)
from .kohya_utils import is_kohya_state_dict, convert_kohya_state_dict_to_diffusers
from .direct_loader import load_lora_direct, should_use_direct_loader

if is_peft_available():
    import peft
is_onediffx_lora_available = version.parse(diffusers.__version__) >= version.parse(
    "0.19.3"
)


class OneDiffXWarning(Warning):
    pass


warnings.filterwarnings("always", category=OneDiffXWarning)
warnings.filterwarnings("always", category=DeprecationWarning)


def deprecated():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"function {func.__name__} of onediffx.lora is deprecated",
                category=DeprecationWarning,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


USE_PEFT_BACKEND = False


@deprecated()
def load_and_fuse_lora(
    pipeline: LoraLoaderMixin,
    pretrained_model_name_or_path_or_dict: Union[str, Path, Dict[str, torch.Tensor]],
    adapter_name: Optional[str] = None,
    *,
    lora_scale: float = 1.0,
    offload_device="cuda",
    use_cache=False,
    device: Union[str, torch.device] = None,
    **kwargs,
):
    return load_lora_and_optionally_fuse(
        pipeline,
        pretrained_model_name_or_path_or_dict,
        adapter_name,
        lora_scale=lora_scale,
        offload_device=offload_device,
        use_cache=use_cache,
        fuse=True,
        device=device,
        **kwargs,
    )


@deprecated()
def load_lora_and_optionally_fuse(
    pipeline: LoraLoaderMixin,
    pretrained_model_name_or_path_or_dict: Union[str, Path, Dict[str, torch.Tensor]],
    adapter_name: Optional[str] = None,
    *,
    fuse,
    lora_scale: Optional[float] = None,
    offload_device="cuda",
    use_cache=False,
    device: Union[str, torch.device] = None,
    **kwargs,
) -> None:
    if not is_onediffx_lora_available:
        raise RuntimeError(
            "onediffx.lora only supports diffusers of at least version 0.19.3"
        )

    # Default device to cuda if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"[OneDiffX] No device specified, defaulting to: {device}")

    _init_adapters_info(pipeline)

    if lora_scale is None:
        lora_scale = 1.0
    elif not fuse:
        warnings.warn(
            "When fuse=False, the lora_scale will be ignored and set to 1.0 as default",
            category=OneDiffXWarning,
        )

    if fuse and len(pipeline._active_adapter_names) > 0:
        warnings.warn(
            "The current API is supported for operating with a single LoRA file. "
            "You are trying to load and fuse more than one LoRA "
            "which is not well-supported and may lead to accuracy issues.",
            category=OneDiffXWarning,
        )

    if adapter_name is None:
        adapter_name = create_adapter_names(pipeline)

    if adapter_name in pipeline._adapter_names:
        warnings.warn(
            f"adapter_name {adapter_name} already exists, will be ignored",
            category=OneDiffXWarning,
        )
        return

    pipeline._adapter_names.add(adapter_name)

    if fuse:
        pipeline._active_adapter_names[adapter_name] = lora_scale

    self = pipeline

    # Check if we should use the direct loader
    if should_use_direct_loader(pretrained_model_name_or_path_or_dict, **kwargs):
        try:
            logger.info("[OneDiffX] Using direct GPU loader to avoid CPU memory overhead")
            state_dict, network_alphas = load_lora_direct(
                pretrained_model_name_or_path_or_dict,
                device=device,
                unet_config=self.unet.config,
                **kwargs,
            )
            # Skip format checks below since direct loader already handled conversion
            skip_format_check = True
        except Exception as e:
            # Direct loader failed, fall back to regular loading
            logger.warning(f"[OneDiffX] Direct loader error: {e}, falling back to diffusers")
            skip_format_check = False
            # Continue with regular loading below
    else:
        skip_format_check = False
    
    # Only use regular loading if direct loader wasn't used or failed
    if not skip_format_check or 'state_dict' not in locals():
        if use_cache:
            state_dict, network_alphas = load_state_dict_cached(
                pretrained_model_name_or_path_or_dict,
                device=device,
                unet_config=self.unet.config,
                **kwargs,
            )
        else:
            # Pass device parameter to diffusers if specified
            if device is not None:
                kwargs['device'] = device
                
            # for diffusers <= 0.20
            if hasattr(LoraLoaderMixin, "_map_sgm_blocks_to_diffusers"):
                orig_func = getattr(LoraLoaderMixin, "_map_sgm_blocks_to_diffusers")
                LoraLoaderMixin._map_sgm_blocks_to_diffusers = (
                    _maybe_map_sgm_blocks_to_diffusers
                )
            
            state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(
                pretrained_model_name_or_path_or_dict,
                unet_config=self.unet.config,
                **kwargs,
            )
            
            if hasattr(LoraLoaderMixin, "_map_sgm_blocks_to_diffusers"):
                LoraLoaderMixin._map_sgm_blocks_to_diffusers = orig_func

    # Check if the state dict is in Kohya format and convert if needed
    # Skip this if we already used the direct loader (skip_format_check is True)
    if not skip_format_check and is_kohya_state_dict(state_dict):
        logger.info("[OneDiffX] Detected Kohya format LoRA, converting to diffusers format while preserving GPU tensors")
        state_dict, converted_network_alphas = convert_kohya_state_dict_to_diffusers(state_dict)
        # Merge converted alphas with existing ones
        if converted_network_alphas:
            if network_alphas is None:
                network_alphas = {}
            network_alphas.update(converted_network_alphas)
    
    is_correct_format = all("lora" in key for key in state_dict.keys())
    if not is_correct_format:
        raise ValueError("[OneDiffX load_and_fuse_lora] Invalid LoRA checkpoint.")

    # State dict loaded successfully

    # load lora into unet
    load_lora_into_unet(
        self,
        state_dict,
        network_alphas,
        self.unet,
        adapter_name=adapter_name,
        lora_scale=lora_scale,
        offload_device=offload_device,
        use_cache=use_cache,
        fuse=fuse,
    )

    # load lora weights into text encoder
    text_encoder_state_dict = {
            k: v for k, v in state_dict.items() if "text_encoder." in k
        }
    
    if len(text_encoder_state_dict) > 0:
        load_lora_into_text_encoder(
                self,
                text_encoder_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder,
                prefix="text_encoder",
                lora_scale=lora_scale,
                adapter_name=adapter_name,
                _pipeline=self,
                fuse=fuse,
            )

    text_encoder_2_state_dict = {
            k: v for k, v in state_dict.items() if "text_encoder_2." in k
        }
    
    if len(text_encoder_2_state_dict) > 0 and hasattr(self, "text_encoder_2"):
        load_lora_into_text_encoder(
                self,
                text_encoder_2_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder_2,
                prefix="text_encoder_2",
                lora_scale=lora_scale,
                adapter_name=adapter_name,
                _pipeline=self,
                fuse=fuse,
            )
    
    # Clear the state dict to free any remaining CPU memory
    # Note: This is safe because we've already transferred all needed tensors
    remaining_keys = list(state_dict.keys())
    if remaining_keys:
        logger.debug(f"[OneDiffX] Clearing {len(remaining_keys)} remaining keys from state_dict")
        state_dict.clear()


@deprecated()
def unfuse_lora(pipeline: LoraLoaderMixin):
    def _unfuse_lora_apply(m: torch.nn.Module):
        if isinstance(m, (torch.nn.Linear, PatchedLoraProjection, torch.nn.Conv2d)):
            _unfuse_lora(m)
        elif is_peft_available() and isinstance(
            m,
            (peft.tuners.lora.layer.Linear, peft.tuners.lora.layer.Conv2d),
        ):
            _unfuse_lora(m.base_layer)

    pipeline._active_adapter_names.clear()

    pipeline.unet.apply(_unfuse_lora_apply)
    if hasattr(pipeline, "text_encoder"):
        pipeline.text_encoder.apply(_unfuse_lora_apply)
    if hasattr(pipeline, "text_encoder_2"):
        pipeline.text_encoder_2.apply(_unfuse_lora_apply)


@deprecated()
def set_and_fuse_adapters(
    pipeline: LoraLoaderMixin,
    adapter_names: Optional[Union[List[str], str]] = None,
    adapter_weights: Optional[List[float]] = None,
):
    if not hasattr(pipeline, "_adapter_names"):
        raise RuntimeError("Didn't find any LoRA, please load LoRA first")
    if adapter_names is None:
        adapter_names = pipeline.active_adapter_names

    if isinstance(adapter_names, str):
        adapter_names = [adapter_names]

    if adapter_weights is None:
        adapter_weights = [
            1.0,
        ] * len(adapter_names)
    elif isinstance(adapter_weights, (int, float)):
        adapter_weights = [
            adapter_weights,
        ] * len(adapter_names)

    adapter_names = [x for x in adapter_names if x in pipeline._adapter_names]
    pipeline._active_adapter_names = {
        k: v for k, v in zip(adapter_names, adapter_weights)
    }

    def set_adapters_apply(m):
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d, PatchedLoraProjection)):
            _set_adapter(m, adapter_names, adapter_weights)
        elif is_peft_available() and isinstance(
            m,
            (peft.tuners.lora.layer.Linear, peft.tuners.lora.layer.Conv2d),
        ):
            _set_adapter(m.base_layer, adapter_names, adapter_weights)

    pipeline.unet.apply(set_adapters_apply)
    if hasattr(pipeline, "text_encoder"):
        pipeline.text_encoder.apply(set_adapters_apply)
    if hasattr(pipeline, "text_encoder_2"):
        pipeline.text_encoder_2.apply(set_adapters_apply)


@deprecated()
def delete_adapters(
    self, adapter_names: Union[List[str], str] = None, safe_delete=True
):
    if adapter_names is None:
        adapter_names = list(self._adapter_names)
    elif isinstance(adapter_names, str):
        adapter_names = [adapter_names]

    for adapter_name in adapter_names:
        self._adapter_names.remove(adapter_name)
        self._active_adapter_names.pop(adapter_name, None)

    def delete_adapters_apply(m):
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d, PatchedLoraProjection)):
            _delete_adapter(m, adapter_names, safe_delete=safe_delete)
        elif is_peft_available() and isinstance(
            m,
            (peft.tuners.lora.layer.Linear, peft.tuners.lora.layer.Conv2d),
        ):
            _delete_adapter(m.base_layer, adapter_names, safe_delete=safe_delete)

    self.unet.apply(delete_adapters_apply)
    if hasattr(self, "text_encoder"):
        self.text_encoder.apply(delete_adapters_apply)
    if hasattr(self, "text_encoder_2"):
        self.text_encoder_2.apply(delete_adapters_apply)


@deprecated()
def get_active_adapters(self) -> List[str]:
    if hasattr(self, "_adapter_names"):
        return list(self._active_adapter_names.keys())
    else:
        return []


def _init_adapters_info(self: torch.nn.Module):
    if not hasattr(self, "_adapter_names"):
        setattr(self, "_adapter_names", set())
        setattr(self, "_active_adapter_names", {})


class LRUCacheDict(OrderedDict):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if len(self) >= self.capacity:
            oldest_key = next(iter(self))
            del self[oldest_key]
        super().__setitem__(key, value)


def load_state_dict_cached(
    lora: Union[str, Path, Dict[str, torch.Tensor]],
    device: Union[str, torch.device] = None,
    **kwargs,
) -> Tuple[Dict, Dict]:
    assert isinstance(lora, (str, Path, dict))
    if isinstance(lora, dict):
        # Use direct loader which handles format detection and conversion
        state_dict, network_alphas = load_lora_direct(lora, device=device, **kwargs)
        return state_dict, network_alphas

    global CachedLoRAs
    
    
    weight_name = kwargs.get("weight_name", None)

    lora_name = str(lora) + (f"/{weight_name}" if weight_name else "")
    if lora_name in CachedLoRAs:
        logger.debug(
            f"[OneDiffX Cached LoRA] get cached lora of name: {str(lora_name)}"
        )
        cached_result = CachedLoRAs[lora_name]
        # Check if we need to move tensors to the requested device
        if isinstance(cached_result, tuple) and len(cached_result) >= 2:
            # Check if we need to move tensors to the requested device
            if device is not None:
                state_dict, network_alphas = cached_result[0], cached_result[1]
                target_device = torch.device(device) if isinstance(device, str) else device
                
                # Check if any tensors are on the wrong device
                needs_device_transfer = False
                for key, tensor in state_dict.items():
                    if torch.is_tensor(tensor) and tensor.device != target_device:
                        needs_device_transfer = True
                        break
                
                if needs_device_transfer:
                    logger.info(f"[OneDiffX Cached LoRA] Moving cached tensors from {tensor.device} to {target_device}")
                    # Create new state dict with tensors on correct device
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if torch.is_tensor(value):
                            new_state_dict[key] = value.to(target_device)
                        else:
                            new_state_dict[key] = value
                    
                    # Update cache with tensors on correct device
                    CachedLoRAs[lora_name] = (new_state_dict, network_alphas)
                    return new_state_dict, network_alphas
                
        return cached_result

    # Try to use direct loader for file paths
    if should_use_direct_loader(lora, **kwargs):
        try:
            logger.info("[OneDiffX Cached] Using direct GPU loader for cached LoRA")
            state_dict, network_alphas = load_lora_direct(
                    lora,
                    device=device,
                    **kwargs,
                )
        except Exception as e:
            # Direct loader failed, fall back to regular loading
            logger.warning(f"[OneDiffX Cached] Direct loader error: {e}, falling back to diffusers")
            # Fall through to regular loading below
            state_dict = None
    else:
        state_dict = None
    
    # Fall back to regular loading if direct loader wasn't used or failed
    if state_dict is None:
        # Pass device parameter to diffusers if specified
        if device is not None:
            kwargs['device'] = device
            
        state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(
                lora,
                **kwargs,
            )
        
        # Check if the state dict is in Kohya format and convert if needed
        if is_kohya_state_dict(state_dict):
            logger.info("[OneDiffX Cached] Detected Kohya format LoRA, converting to diffusers format")
            state_dict, converted_network_alphas = convert_kohya_state_dict_to_diffusers(state_dict)
            if converted_network_alphas:
                if network_alphas is None:
                    network_alphas = {}
                network_alphas.update(converted_network_alphas)
    
    CachedLoRAs[lora_name] = (state_dict, network_alphas)
    
    logger.debug(f"[OneDiffX Cached LoRA] create cached lora of name: {str(lora_name)}")
    return state_dict, network_alphas


CachedLoRAs = LRUCacheDict(100)


def clear_lora_cache():
    """Clear the cached LoRA state dicts to free memory."""
    global CachedLoRAs
    CachedLoRAs.clear()
    logger.info("[OneDiffX] Cleared LoRA cache")


def create_adapter_names(pipe):
    for i in range(0, 10000):
        result = f"default_{i}"
        if result not in pipe._adapter_names:
            return result
    raise RuntimeError("Too much LoRA loaded")
