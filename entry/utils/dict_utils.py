"""
Dictionary utility functions for common operations.
"""
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def safe_dict_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely get a value from a dictionary with a default.
    
    Args:
        dictionary: The dictionary to access
        key: The key to look up
        default: Default value if key doesn't exist
        
    Returns:
        The value or default
    """
    return dictionary.get(key, default)


def safe_dict_get_multi(dictionary: Dict[str, Any], keys: List[str], defaults: List[Any]) -> List[Any]:
    """
    Safely get multiple values from a dictionary with defaults.
    
    Args:
        dictionary: The dictionary to access
        keys: List of keys to look up
        defaults: List of default values (must match keys length)
        
    Returns:
        List of values or defaults
    """
    if len(keys) != len(defaults):
        raise ValueError("Keys and defaults lists must have the same length")
    
    return [dictionary.get(key, default) for key, default in zip(keys, defaults)]


def safe_dict_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """
    Safely update a dictionary with another dictionary.
    
    Args:
        target: The dictionary to update
        source: The dictionary to merge from
    """
    if source:
        target.update(source)


def safe_dict_clear(dictionaries: List[Dict[str, Any]]) -> None:
    """
    Safely clear multiple dictionaries.
    
    Args:
        dictionaries: List of dictionaries to clear
    """
    for dictionary in dictionaries:
        if dictionary is not None:
            dictionary.clear()


def extract_dict_values(dictionary: Dict[str, Any], keys: List[str], defaults: List[Any]) -> Dict[str, Any]:
    """
    Extract specific values from a dictionary with defaults.
    
    Args:
        dictionary: The source dictionary
        keys: List of keys to extract
        defaults: List of default values (must match keys length)
        
    Returns:
        Dictionary with extracted values
    """
    if len(keys) != len(defaults):
        raise ValueError("Keys and defaults lists must have the same length")
    
    return {key: dictionary.get(key, default) for key, default in zip(keys, defaults)}


def merge_dicts_safe(*dictionaries: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely merge multiple dictionaries.
    
    Args:
        *dictionaries: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for dictionary in dictionaries:
        if dictionary:
            result.update(dictionary)
    return result


def filter_dict_by_keys(dictionary: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """
    Filter a dictionary to only include specified keys.
    
    Args:
        dictionary: The source dictionary
        keys: List of keys to include
        
    Returns:
        Filtered dictionary
    """
    return {key: dictionary[key] for key in keys if key in dictionary}


def set_dict_value_safe(dictionary: Dict[str, Any], key: str, value: Any) -> None:
    """
    Safely set a value in a dictionary.
    
    Args:
        dictionary: The dictionary to modify
        key: The key to set
        value: The value to set
    """
    if dictionary is not None:
        dictionary[key] = value


def get_nested_dict_value(dictionary: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    """
    Get a value from a nested dictionary using a path.
    
    Args:
        dictionary: The source dictionary
        path: List of keys representing the path
        default: Default value if path doesn't exist
        
    Returns:
        The value at the path or default
    """
    current = dictionary
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def set_nested_dict_value(dictionary: Dict[str, Any], path: List[str], value: Any) -> None:
    """
    Set a value in a nested dictionary using a path.
    
    Args:
        dictionary: The dictionary to modify
        path: List of keys representing the path
        value: The value to set
    """
    current = dictionary
    for key in path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[path[-1]] = value 