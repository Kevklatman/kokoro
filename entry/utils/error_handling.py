"""
Error handling utilities for consistent error management across the application.
"""
from typing import Optional, Callable, Any
from fastapi import HTTPException
from loguru import logger
import traceback
from entry.utils.string_utils import format_operation_details


def safe_execute(
    operation: Callable,
    context: str = "Operation",
    default_return: Any = None,
    log_errors: bool = True,
    reraise: bool = True
) -> Any:
    """
    Safely execute an operation with consistent error handling.
    
    Args:
        operation: Function to execute
        context: Context for error messages
        default_return: Value to return on error if not reraising
        log_errors: Whether to log errors
        reraise: Whether to reraise exceptions
        
    Returns:
        Result of operation or default_return on error
    """
    try:
        return operation()
    except Exception as e:
        if log_errors:
            logger.error(f"❌ {context} error: {str(e)}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
        
        if reraise:
            raise HTTPException(status_code=500, detail=str(e))
        
        return default_return


def handle_http_error(
    error: Exception,
    status_code: int = 500,
    context: str = "Operation",
    log_error: bool = True
) -> HTTPException:
    """
    Create a consistent HTTP exception with logging.
    
    Args:
        error: The original exception
        status_code: HTTP status code
        context: Context for error messages
        log_error: Whether to log the error
        
    Returns:
        HTTPException with consistent formatting
    """
    if log_error:
        logger.error(f"❌ {context} error: {str(error)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
    
    return HTTPException(status_code=status_code, detail=str(error))


def create_not_found_error(item: str, item_type: str = "Item") -> HTTPException:
    """Create a consistent 404 error."""
    return HTTPException(status_code=404, detail=f"{item_type} '{item}' not found")


def create_validation_error(message: str) -> HTTPException:
    """Create a consistent 400 validation error."""
    return HTTPException(status_code=400, detail=message)


def create_server_error(message: str, context: str = "Server operation") -> HTTPException:
    """Create a consistent 500 server error."""
    logger.error(f"❌ {context} error: {message}")
    return HTTPException(status_code=500, detail=message)


def log_operation_start(operation: str, **kwargs):
    """Log the start of an operation with consistent formatting."""
    details = format_operation_details(**kwargs)
    logger.info(f"🔄 Starting {operation}: {details}" if details else f"🔄 Starting {operation}")


def log_operation_success(operation: str, **kwargs):
    """Log the successful completion of an operation."""
    details = format_operation_details(**kwargs)
    logger.info(f"✅ Completed {operation}: {details}" if details else f"✅ Completed {operation}")


def log_operation_failure(operation: str, error: Exception, **kwargs):
    """Log the failure of an operation."""
    details = format_operation_details(**kwargs)
    logger.error(f"❌ Failed {operation}: {str(error)}")
    if details:
        logger.error(f"❌ Operation details: {details}")


def safe_dict_get(dictionary: dict, key: str, default: Any = None) -> Any:
    """
    Safely get a value from a dictionary with None checking.
    
    Args:
        dictionary: Dictionary to get value from
        key: Key to look up
        default: Default value if key not found or dict is None
        
    Returns:
        Value from dictionary or default
    """
    if dictionary is None:
        return default
    return dictionary.get(key, default)


def extract_emotion_params(emotion_preset: Optional[dict]) -> dict:
    """
    Extract emotion parameters from preset with consistent defaults.
    
    Args:
        emotion_preset: Emotion preset dictionary or None
        
    Returns:
        Dictionary with emotion parameters
    """
    if emotion_preset is None:
        return {
            'breathiness': 0.0,
            'tenseness': 0.0,
            'jitter': 0.0,
            'sultry': 0.0
        }
    
    return {
        'breathiness': safe_dict_get(emotion_preset, 'breathiness', 0.0),
        'tenseness': safe_dict_get(emotion_preset, 'tenseness', 0.0),
        'jitter': safe_dict_get(emotion_preset, 'jitter', 0.0),
        'sultry': safe_dict_get(emotion_preset, 'sultry', 0.0)
    }


def batch_list_operations(operations: list[tuple[str, Any]]) -> dict[str, list]:
    """
    Perform batch list operations with consistent error handling.
    
    Args:
        operations: List of (operation_type, value) tuples
        
    Returns:
        Dictionary with operation results
    """
    results = {
        'append': [],
        'extend': [],
        'insert': []
    }
    
    for op_type, value in operations:
        try:
            if op_type == 'append':
                results['append'].append(value)
            elif op_type == 'extend':
                results['extend'].extend(value)
            elif op_type == 'insert':
                results['insert'].append(value)
        except Exception as e:
            logger.warning(f"Failed to perform {op_type} operation: {e}")
    
    return results 