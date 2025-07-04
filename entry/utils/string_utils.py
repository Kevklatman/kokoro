"""
String manipulation utilities for consistent string operations across the application.
"""
from typing import List, Dict, Any, Optional
import os


def format_size_info(format_name: str, size_kb: float, quality: str = None) -> str:
    """
    Format size information with consistent formatting.
    
    Args:
        format_name: Name of the format
        size_kb: Size in KB
        quality: Quality level (optional)
        
    Returns:
        Formatted size string
    """
    format_upper = format_name.upper()
    if quality:
        return f"{format_upper} response size ({quality}): {size_kb:.1f}KB"
    return f"{format_upper} size: {size_kb:.1f}KB"


def format_operation_details(**kwargs) -> str:
    """
    Format operation details for logging.
    
    Args:
        **kwargs: Key-value pairs to format
        
    Returns:
        Formatted details string
    """
    if not kwargs:
        return ""
    return ", ".join([f"{k}={v}" for k, v in kwargs.items()])


def normalize_string_case(text: str, case: str = "lower") -> str:
    """
    Normalize string case consistently.
    
    Args:
        text: Text to normalize
        case: Target case ('lower', 'upper', 'title')
        
    Returns:
        Normalized string
    """
    case_map = {
        "lower": str.lower,
        "upper": str.upper,
        "title": str.title
    }
    
    if case not in case_map:
        raise ValueError(f"Invalid case: {case}. Valid options: {list(case_map.keys())}")
    
    return case_map[case](text)


def parse_comma_separated_string(text: str, strip_whitespace: bool = True) -> List[str]:
    """
    Parse comma-separated string with consistent logic.
    
    Args:
        text: Comma-separated string
        strip_whitespace: Whether to strip whitespace from items
        
    Returns:
        List of parsed items
    """
    if not text or not text.strip():
        return []
    
    if text.strip() == "*":
        return ["*"]
    
    items = text.split(",")
    if strip_whitespace:
        items = [item.strip() for item in items if item.strip()]
    
    return items


def build_path(*path_parts: str) -> str:
    """
    Build file path with consistent joining.
    
    Args:
        *path_parts: Path components to join
        
    Returns:
        Joined path
    """
    return os.path.join(*path_parts)


def get_file_extension(filename: str) -> str:
    """
    Get file extension consistently.
    
    Args:
        filename: File name
        
    Returns:
        File extension (without dot)
    """
    return os.path.splitext(filename)[1].lower().lstrip('.')


def is_valid_file_extension(filename: str, valid_extensions: List[str]) -> bool:
    """
    Check if file has valid extension.
    
    Args:
        filename: File name to check
        valid_extensions: List of valid extensions (without dots)
        
    Returns:
        True if extension is valid
    """
    extension = get_file_extension(filename)
    return extension in valid_extensions


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system use.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import re
    # Remove or replace unsafe characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    return sanitized


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_file_size(bytes_size: int) -> str:
    """
    Format file size in bytes to human-readable string.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}TB"


def extract_key_value_pairs(text: str, separator: str = "=") -> Dict[str, str]:
    """
    Extract key-value pairs from string.
    
    Args:
        text: Text containing key-value pairs
        separator: Separator between key and value
        
    Returns:
        Dictionary of key-value pairs
    """
    result = {}
    for line in text.split('\n'):
        line = line.strip()
        if separator in line:
            key, value = line.split(separator, 1)
            result[key.strip()] = value.strip()
    return result


def format_list_for_display(items: List[Any], max_items: int = 10) -> str:
    """
    Format list for display with truncation.
    
    Args:
        items: List of items to format
        max_items: Maximum number of items to show
        
    Returns:
        Formatted list string
    """
    if not items:
        return "[]"
    
    if len(items) <= max_items:
        return f"[{', '.join(map(str, items))}]"
    
    displayed = items[:max_items]
    return f"[{', '.join(map(str, displayed))}, ... (+{len(items) - max_items} more)]" 