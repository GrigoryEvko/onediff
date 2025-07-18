import functools
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import diffusers
import safetensors
import safetensors.torch
import torch
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import _get_model_file
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


def _gpu_direct_load_state_dict(
    pretrained_model_name_or_path_or_dict: Union[str, Path, Dict[str, torch.Tensor]],
    device: Union[str, torch.device] = None,
    **kwargs,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Load state dict directly to GPU, bypassing CPU loading.

    This function handles loading LoRA weights directly to the specified device,
    avoiding the CPU memory footprint from diffusers' default loading behavior.
    """
    # If already a dict, extract network_alphas from it
    if isinstance(pretrained_model_name_or_path_or_dict, dict):
        # Extract network alphas from state dict keys ending with .alpha
        network_alphas = {}
        state_dict = {}
        for key, value in pretrained_model_name_or_path_or_dict.items():
            if key.endswith(".alpha"):
                # Extract the module name and store alpha value
                module_name = key[:-6]  # Remove ".alpha" suffix
                network_alphas[module_name] = value.item() if hasattr(value, "item") else value
            else:
                state_dict[key] = value
        return state_dict, network_alphas

    # Extract parameters
    weight_name = kwargs.get("weight_name", None)
    cache_dir = kwargs.get("cache_dir", None)
    force_download = kwargs.get("force_download", False)
    proxies = kwargs.get("proxies", None)
    local_files_only = kwargs.get("local_files_only", False)
    token = kwargs.get("token", None)
    revision = kwargs.get("revision", None)
    subfolder = kwargs.get("subfolder", None)

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try to load safetensors first
    model_file = None
    if weight_name is None or weight_name.endswith(".safetensors"):
        try:
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name or "pytorch_lora_weights.safetensors",
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
            )
            # Load directly to GPU
            state_dict = safetensors.torch.load_file(model_file, device=str(device))
            
            # Extract network alphas from metadata
            network_alphas = {}
            try:
                with safetensors.safe_open(model_file, framework="pt") as f:
                    metadata = f.metadata()
                    if metadata:
                        # Look for alpha values in metadata
                        for key, value in metadata.items():
                            if ".alpha" in key:
                                try:
                                    network_alphas[key] = float(value)
                                except (ValueError, TypeError):
                                    pass
            except Exception:
                pass
            
            # Also extract from state dict keys ending with .alpha
            alpha_keys = [k for k in state_dict.keys() if k.endswith(".alpha")]
            for key in alpha_keys:
                module_name = key[:-6]  # Remove ".alpha" suffix
                alpha_value = state_dict.pop(key)
                network_alphas[module_name] = alpha_value.item() if hasattr(alpha_value, "item") else alpha_value
            
            return state_dict, network_alphas
        except Exception:
            pass

    # Fall back to torch format
    if model_file is None:
        model_file = _get_model_file(
            pretrained_model_name_or_path_or_dict,
            weights_name=weight_name or "pytorch_lora_weights.bin",
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
        )
        # Load directly to GPU
        state_dict = torch.load(model_file, map_location=device)
        
        # Extract network alphas from state dict keys ending with .alpha
        network_alphas = {}
        alpha_keys = [k for k in state_dict.keys() if k.endswith(".alpha")]
        for key in alpha_keys:
            module_name = key[:-6]  # Remove ".alpha" suffix
            alpha_value = state_dict.pop(key)
            network_alphas[module_name] = alpha_value.item() if hasattr(alpha_value, "item") else alpha_value
        
        return state_dict, network_alphas


@deprecated()
def load_and_fuse_lora(
    pipeline: LoraLoaderMixin,
    pretrained_model_name_or_path_or_dict: Union[str, Path, Dict[str, torch.Tensor]],
    adapter_name: Optional[str] = None,
    *,
    lora_scale: float = 1.0,
    offload_device="cuda",
    use_cache=False,
    memory_efficient=True,
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
        memory_efficient=memory_efficient,
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
    memory_efficient=True,
    device: Union[str, torch.device] = None,
    **kwargs,
) -> None:
    if not is_onediffx_lora_available:
        raise RuntimeError(
            "onediffx.lora only supports diffusers of at least version 0.19.3"
        )

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

    # Skip caching when memory_efficient is True to reduce memory usage
    if use_cache and not memory_efficient:
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

    is_correct_format = all("lora" in key for key in state_dict.keys())
    if not is_correct_format:
        raise ValueError("[OneDiffX load_and_fuse_lora] Invalid LoRA checkpoint.")

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
        memory_efficient=memory_efficient,
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
            memory_efficient=memory_efficient,
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
            memory_efficient=memory_efficient,
        )


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
        # If already a dict, return as-is (diffusers will handle it)
        return lora, {}

    global CachedLoRAs
    weight_name = kwargs.get("weight_name", None)

    # Include device in cache key to avoid device mismatches
    device_str = str(device) if device is not None else "cpu"
    lora_name = (
        str(lora) + (f"/{weight_name}" if weight_name else "") + f"/{device_str}"
    )

    if lora_name in CachedLoRAs:
        logger.debug(
            f"[OneDiffX Cached LoRA] get cached lora of name: {str(lora_name)}"
        )
        return CachedLoRAs[lora_name]

    # Pass device parameter to diffusers if specified
    if device is not None:
        kwargs['device'] = device
        
    state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(
        lora,
        **kwargs,
    )

    CachedLoRAs[lora_name] = (state_dict, network_alphas)
    logger.debug(f"[OneDiffX Cached LoRA] create cached lora of name: {str(lora_name)}")
    return state_dict, network_alphas


# Reduced cache size from 100 to 10 for memory efficiency
# Set ONEDIFFX_LORA_CACHE_SIZE=0 to disable caching entirely
CACHE_SIZE = int(os.environ.get("ONEDIFFX_LORA_CACHE_SIZE", "10"))
CachedLoRAs = LRUCacheDict(CACHE_SIZE) if CACHE_SIZE > 0 else {}


def create_adapter_names(pipe):
    for i in range(0, 10000):
        result = f"default_{i}"
        if result not in pipe._adapter_names:
            return result
    raise RuntimeError("Too much LoRA loaded")


def clear_lora_cache():
    """Clear the LoRA cache to free memory"""
    global CachedLoRAs
    if hasattr(CachedLoRAs, "clear"):
        CachedLoRAs.clear()
        logger.info("[OneDiffX] Cleared LoRA cache")
    return
