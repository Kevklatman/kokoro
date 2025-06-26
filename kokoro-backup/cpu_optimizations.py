"""
CPU Optimization utilities for Kokoro TTS

This module provides functions to apply CPU-specific optimizations like JIT compilation
to improve inference performance on CPU devices.
"""
import torch
from loguru import logger
from typing import Any, Callable, Optional

def apply_jit_to_method(obj: Any, method_name: str) -> bool:
    """
    Apply JIT compilation to a specific method of an object.
    
    Args:
        obj: The object containing the method to optimize
        method_name: The name of the method to optimize
        
    Returns:
        bool: True if JIT compilation was successfully applied, False otherwise
    """
    if not hasattr(obj, method_name):
        logger.warning(f"Object does not have method {method_name}")
        return False
    
    try:
        # Get the original method
        original_method = getattr(obj, method_name)
        
        # Only apply JIT if we're on CPU
        if str(next(obj.parameters()).device) != 'cpu':
            logger.info(f"Skipping JIT compilation for {method_name} as device is not CPU")
            return False
        
        # Create a wrapper function that will be JIT compiled
        def create_jit_wrapper(func):
            # Use torch.jit.trace for methods that process tensor inputs
            # This is safer than script for complex models
            def wrapper(*args, **kwargs):
                with torch.no_grad():
                    return func(*args, **kwargs)
            
            # Return the JIT optimized function
            return torch.jit.script(wrapper)
        
        # Apply JIT compilation
        jit_method = create_jit_wrapper(original_method)
        
        # Store the original method for reference
        setattr(obj, f"_original_{method_name}", original_method)
        
        # Replace the method with the JIT compiled version
        # We use a lambda to maintain the method signature
        setattr(obj, method_name, lambda *args, **kwargs: jit_method(*args, **kwargs))
        
        logger.info(f"Successfully applied JIT compilation to {method_name}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to apply JIT compilation to {method_name}: {str(e)}")
        # If there was an error, make sure we restore the original method
        if hasattr(obj, f"_original_{method_name}"):
            setattr(obj, method_name, getattr(obj, f"_original_{method_name}"))
        return False

def optimize_for_cpu(model: Any) -> bool:
    """
    Apply CPU-specific optimizations to a model.
    
    Args:
        model: The model to optimize
        
    Returns:
        bool: True if optimizations were successfully applied, False otherwise
    """
    success = False
    
    # Only apply optimizations if we're on CPU
    if str(next(model.parameters()).device) != 'cpu':
        logger.info("Skipping CPU optimizations as device is not CPU")
        return False
    
    try:
        # Apply JIT compilation to forward_with_tokens method
        if apply_jit_to_method(model, 'forward_with_tokens'):
            success = True
            logger.info("Successfully applied JIT compilation to forward_with_tokens")
        
        return success
    except Exception as e:
        logger.error(f"Failed to apply CPU optimizations: {str(e)}")
        return False
