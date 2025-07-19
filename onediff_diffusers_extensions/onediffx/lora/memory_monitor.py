"""
Memory monitoring utilities for tracking CPU/GPU memory usage during LoRA loading.
Used to identify sources of unexpected CPU memory allocations.
"""

import functools
import gc
import sys
import time
import torch
import psutil
from typing import Tuple, Optional, Any, Callable, Dict

# Global start time for consistent timestamps
_start_time = time.time()

def _timestamp():
    """Return milliseconds since monitoring started"""
    return f"[{int((time.time() - _start_time) * 1000):>6}ms]"


def get_memory_stats() -> Tuple[float, float, float]:
    """Get current memory statistics"""
    try:
        process = psutil.Process()
        cpu_mb = process.memory_info().rss / 1024 / 1024
    except Exception:
        cpu_mb = 0
    
    try:
        gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        gpu_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0
    except Exception:
        gpu_mb = 0
        gpu_reserved_mb = 0
    
    return cpu_mb, gpu_mb, gpu_reserved_mb


def print_memory_delta(operation_name: str, before: Tuple[float, float, float], after: Tuple[float, float, float], threshold_mb: float = 1.0):
    """Print memory delta if significant change detected"""
    cpu_delta = after[0] - before[0]
    gpu_delta = after[1] - before[1]
    gpu_reserved_delta = after[2] - before[2]
    
    # Only print if there's a significant change (> threshold)
    if abs(cpu_delta) > threshold_mb or abs(gpu_delta) > threshold_mb:
        print(f"{_timestamp()} [MEMORY] {operation_name}:")
        print(f"{_timestamp()}   CPU: {before[0]:.1f}MB -> {after[0]:.1f}MB (Δ {cpu_delta:+.1f}MB)")
        print(f"{_timestamp()}   GPU: {before[1]:.1f}MB -> {after[1]:.1f}MB (Δ {gpu_delta:+.1f}MB)")
        print(f"{_timestamp()}   GPU Reserved: {before[2]:.1f}MB -> {after[2]:.1f}MB (Δ {gpu_reserved_delta:+.1f}MB)")


def memory_checkpoint(operation_name: str, threshold_mb: float = 1.0):
    """
    Inline memory checkpoint - call this to monitor memory at specific points
    
    Usage:
        before = memory_checkpoint("Operation start")
        # ... do operation ...
        memory_checkpoint("Operation end", before)
    """
    stats = get_memory_stats()
    if hasattr(memory_checkpoint, '_last_stats'):
        print_memory_delta(operation_name, memory_checkpoint._last_stats, stats, threshold_mb)
    memory_checkpoint._last_stats = stats
    return stats


def memory_monitor(operation_name: str, threshold_mb: float = 1.0, force_gc: bool = False):
    """
    Decorator to monitor memory usage of a function
    
    Args:
        operation_name: Name to display in memory monitoring output
        threshold_mb: Only print if memory change exceeds this threshold
        force_gc: Whether to force garbage collection before measurement
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if force_gc:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            before = get_memory_stats()
            result = func(*args, **kwargs)
            after = get_memory_stats()
            
            print_memory_delta(f"{operation_name} ({func.__name__})", before, after, threshold_mb)
            return result
        return wrapper
    return decorator


class MemoryTracker:
    """Context manager for tracking memory usage in a block of code"""
    
    def __init__(self, operation_name: str, threshold_mb: float = 1.0, force_gc: bool = False):
        self.operation_name = operation_name
        self.threshold_mb = threshold_mb
        self.force_gc = force_gc
        self.before_stats = None
    
    def __enter__(self):
        if self.force_gc:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        self.before_stats = get_memory_stats()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            after_stats = get_memory_stats()
            if self.before_stats is not None:
                print_memory_delta(self.operation_name, self.before_stats, after_stats, self.threshold_mb)
        except Exception as e:
            print(f"[MEMORY] {self.operation_name}: Error in memory tracking - {e}")


def track_tensor_memory(tensor_name: str, tensor: torch.Tensor):
    """Track memory usage of a specific tensor"""
    try:
        if hasattr(tensor, 'element_size') and hasattr(tensor, 'numel'):
            size_mb = tensor.element_size() * tensor.numel() / 1024 / 1024
            device = tensor.device if hasattr(tensor, 'device') else 'unknown'
            shape = tuple(tensor.shape) if hasattr(tensor, 'shape') else 'unknown'
            print(f"{_timestamp()} [TENSOR] {tensor_name}: {size_mb:.1f}MB on {device} (shape: {shape})")
        else:
            print(f"{_timestamp()} [TENSOR] {tensor_name}: Not a tensor or missing attributes")
    except Exception as e:
        print(f"{_timestamp()} [TENSOR] {tensor_name}: Error tracking tensor - {e}")


def track_state_dict_memory(state_dict_name: str, state_dict: dict):
    """Track memory usage of a state dict"""
    try:
        total_size_mb = 0
        tensor_count = 0
        
        for key, tensor in state_dict.items():
            try:
                if hasattr(tensor, 'element_size') and hasattr(tensor, 'numel'):
                    size_mb = tensor.element_size() * tensor.numel() / 1024 / 1024
                    total_size_mb += size_mb
                    tensor_count += 1
            except Exception:
                # Skip items that aren't tensors or have issues
                continue
        
        print(f"{_timestamp()} [STATE_DICT] {state_dict_name}: {total_size_mb:.1f}MB total, {tensor_count} tensors")
    except Exception as e:
        print(f"{_timestamp()} [STATE_DICT] {state_dict_name}: Error tracking state dict - {e}")


def get_pytorch_memory_stats() -> Dict[str, int]:
    """Get detailed PyTorch memory allocator statistics"""
    try:
        if torch.cuda.is_available():
            stats = torch.cuda.memory_stats()
            return {
                'allocated_bytes': stats.get('allocated_bytes.all.current', 0),
                'reserved_bytes': stats.get('reserved_bytes.all.current', 0),
                'active_bytes': stats.get('active_bytes.all.current', 0),
                'inactive_split_bytes': stats.get('inactive_split_bytes.all.current', 0),
                'allocation_count': stats.get('allocation.all.current', 0),
                'reserved_count': stats.get('reserved_bytes.all.current', 0),
            }
        return {}
    except Exception as e:
        print(f"[PYTORCH_STATS] Error getting PyTorch memory stats - {e}")
        return {}


def print_pytorch_memory_stats(label: str):
    """Print detailed PyTorch memory statistics"""
    stats = get_pytorch_memory_stats()
    if stats:
        print(f"[PYTORCH_STATS] {label}:")
        print(f"  Allocated: {stats['allocated_bytes'] / 1024 / 1024:.1f}MB")
        print(f"  Reserved: {stats['reserved_bytes'] / 1024 / 1024:.1f}MB")
        print(f"  Active: {stats['active_bytes'] / 1024 / 1024:.1f}MB")
        print(f"  Inactive: {stats['inactive_split_bytes'] / 1024 / 1024:.1f}MB")
        print(f"  Allocations: {stats['allocation_count']}")


def track_tensor_lifecycle(tensor_name: str, tensor: torch.Tensor, operation: str):
    """Track tensor through its lifecycle with reference counting and data pointer"""
    try:
        if not isinstance(tensor, torch.Tensor):
            print(f"{_timestamp()} [TENSOR_LIFECYCLE] {operation} - {tensor_name}: Not a tensor")
            return
            
        ref_count = sys.getrefcount(tensor) - 1  # Subtract 1 for the function argument
        data_ptr = tensor.data_ptr() if hasattr(tensor, 'data_ptr') else 'N/A'
        size_mb = tensor.element_size() * tensor.numel() / 1024 / 1024
        
        print(f"{_timestamp()} [TENSOR_LIFECYCLE] {operation} - {tensor_name}:")
        print(f"{_timestamp()}   refs={ref_count}, ptr={data_ptr}, device={tensor.device}")
        print(f"{_timestamp()}   shape={tuple(tensor.shape)}, dtype={tensor.dtype}, size={size_mb:.1f}MB")
        
        # Check if tensor storage is shared
        if hasattr(tensor, 'storage'):
            storage_ptr = tensor.storage().data_ptr() if tensor.storage() else 'N/A'
            print(f"{_timestamp()}   storage_ptr={storage_ptr}")
    except Exception as e:
        print(f"{_timestamp()} [TENSOR_LIFECYCLE] {operation} - {tensor_name}: Error - {e}")


def track_dict_memory(dict_name: str, dictionary: dict):
    """Track memory usage of dictionaries containing tensors"""
    try:
        total_dict_size = sys.getsizeof(dictionary) / 1024 / 1024
        tensor_count = 0
        tensor_memory = 0
        cpu_tensors = 0
        gpu_tensors = 0
        
        for key, value in dictionary.items():
            if torch.is_tensor(value):
                tensor_count += 1
                size_mb = value.element_size() * value.numel() / 1024 / 1024
                tensor_memory += size_mb
                
                if value.is_cuda:
                    gpu_tensors += 1
                else:
                    cpu_tensors += 1
        
        print(f"{_timestamp()} [DICT_MEMORY] {dict_name}:")
        print(f"{_timestamp()}   dict_overhead={total_dict_size:.1f}MB, tensors={tensor_count}")
        print(f"{_timestamp()}   tensor_memory={tensor_memory:.1f}MB (CPU:{cpu_tensors}, GPU:{gpu_tensors})")
    except Exception as e:
        print(f"{_timestamp()} [DICT_MEMORY] {dict_name}: Error tracking dict - {e}")


def monitored_gc_collect(label: str = ""):
    """Perform garbage collection with monitoring"""
    try:
        before = get_memory_stats()
        
        # Get object count before GC
        import gc
        before_objects = len(gc.get_objects())
        
        # Run garbage collection
        collected_0 = gc.collect(0)  # Collect young generation
        collected_1 = gc.collect(1)  # Collect middle generation  
        collected_2 = gc.collect(2)  # Collect old generation
        total_collected = collected_0 + collected_1 + collected_2
        
        # Get object count after GC
        after_objects = len(gc.get_objects())
        
        after = get_memory_stats()
        
        print(f"{_timestamp()} [GC] {label}:")
        print(f"{_timestamp()}   Collected: {total_collected} objects (gen0:{collected_0}, gen1:{collected_1}, gen2:{collected_2})")
        print(f"{_timestamp()}   Objects: {before_objects} -> {after_objects} (Δ {after_objects - before_objects})")
        print_memory_delta("  Memory", before, after, threshold_mb=0.1)
        
        return total_collected
    except Exception as e:
        print(f"[GC] Error during garbage collection - {e}")
        return 0


def check_tensor_sharing(tensor1_name: str, tensor1: torch.Tensor, 
                        tensor2_name: str, tensor2: torch.Tensor):
    """Check if two tensors share the same storage"""
    try:
        if not (isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor)):
            return
            
        share_storage = tensor1.data_ptr() == tensor2.data_ptr()
        share_memory = False
        
        if hasattr(tensor1, 'storage') and hasattr(tensor2, 'storage'):
            if tensor1.storage() and tensor2.storage():
                share_memory = tensor1.storage().data_ptr() == tensor2.storage().data_ptr()
        
        if share_storage or share_memory:
            print(f"{_timestamp()} [TENSOR_SHARING] WARNING: {tensor1_name} and {tensor2_name} share "
                  f"{'storage' if share_storage else 'memory'}!")
            print(f"{_timestamp()}   {tensor1_name}: device={tensor1.device}, shape={tuple(tensor1.shape)}")
            print(f"{_timestamp()}   {tensor2_name}: device={tensor2.device}, shape={tuple(tensor2.shape)}")
    except Exception as e:
        print(f"{_timestamp()} [TENSOR_SHARING] Error checking tensor sharing - {e}")