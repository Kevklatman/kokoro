"""
List utility functions for common operations.
"""
from typing import List, Any, Optional, Callable, Dict, TypeVar, Generic
from loguru import logger

T = TypeVar('T')


def safe_list_append(target_list: List[Any], item: Any) -> None:
    """
    Safely append an item to a list.
    
    Args:
        target_list: The list to append to
        item: The item to append
    """
    if target_list is not None:
        target_list.append(item)


def safe_list_extend(target_list: List[Any], items: List[Any]) -> None:
    """
    Safely extend a list with items.
    
    Args:
        target_list: The list to extend
        items: The items to add
    """
    if target_list is not None and items:
        target_list.extend(items)


def safe_list_insert(target_list: List[Any], index: int, item: Any) -> None:
    """
    Safely insert an item at a specific index.
    
    Args:
        target_list: The list to insert into
        index: The index to insert at
        item: The item to insert
    """
    if target_list is not None and 0 <= index <= len(target_list):
        target_list.insert(index, item)


def safe_list_remove(target_list: List[Any], item: Any) -> bool:
    """
    Safely remove an item from a list.
    
    Args:
        target_list: The list to remove from
        item: The item to remove
        
    Returns:
        True if item was removed, False otherwise
    """
    if target_list is not None and item in target_list:
        target_list.remove(item)
        return True
    return False


def safe_list_pop(target_list: List[Any], index: int = -1) -> Optional[Any]:
    """
    Safely pop an item from a list.
    
    Args:
        target_list: The list to pop from
        index: The index to pop from (default: -1)
        
    Returns:
        The popped item or None if list is empty or index invalid
    """
    if target_list is not None and 0 <= abs(index) <= len(target_list):
        return target_list.pop(index)
    return None


def extend_list_to_length(target_list: List[Any], target_length: int, default_value: Any) -> None:
    """
    Extend a list to a target length with default values.
    
    Args:
        target_list: The list to extend
        target_length: The target length
        default_value: The default value to use for extension
    """
    if target_list is not None and len(target_list) < target_length:
        needed = target_length - len(target_list)
        target_list.extend([default_value] * needed)


def batch_list_append(target_list: List[Any], items: List[Any], batch_size: int = 1) -> None:
    """
    Append items to a list in batches.
    
    Args:
        target_list: The list to append to
        items: The items to append
        batch_size: Size of each batch (default: 1)
    """
    if target_list is not None and items:
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            target_list.extend(batch)


def filter_list_by_condition(items: List[T], condition: Callable[[T], bool]) -> List[T]:
    """
    Filter a list based on a condition.
    
    Args:
        items: The list to filter
        condition: The condition function
        
    Returns:
        Filtered list
    """
    return [item for item in items if condition(item)]


def map_list_safe(items: List[T], mapper: Callable[[T], Any], default: Any = None) -> List[Any]:
    """
    Safely map a function over a list, using default for failed operations.
    
    Args:
        items: The list to map over
        mapper: The mapping function
        default: Default value for failed operations
        
    Returns:
        Mapped list
    """
    result = []
    for item in items:
        try:
            result.append(mapper(item))
        except Exception as e:
            logger.warning(f"Mapping failed for item {item}: {str(e)}")
            result.append(default)
    return result


def group_list_by_key(items: List[T], key_func: Callable[[T], str]) -> Dict[str, List[T]]:
    """
    Group items in a list by a key function.
    
    Args:
        items: The list to group
        key_func: Function to extract key from each item
        
    Returns:
        Dictionary with keys and grouped items
    """
    grouped = {}
    for item in items:
        key = key_func(item)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(item)
    return grouped


def flatten_list(nested_list: List[List[T]]) -> List[T]:
    """
    Flatten a nested list.
    
    Args:
        nested_list: The nested list to flatten
        
    Returns:
        Flattened list
    """
    return [item for sublist in nested_list for item in sublist]


def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        items: The list to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def remove_duplicates_preserve_order(items: List[T]) -> List[T]:
    """
    Remove duplicates from a list while preserving order.
    
    Args:
        items: The list to deduplicate
        
    Returns:
        List with duplicates removed
    """
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def safe_list_get(items: List[T], index: int, default: T = None) -> Optional[T]:
    """
    Safely get an item from a list by index.
    
    Args:
        items: The list to access
        index: The index to access
        default: Default value if index is out of bounds
        
    Returns:
        The item at index or default
    """
    if items is not None and 0 <= index < len(items):
        return items[index]
    return default


def list_to_dict(items: List[T], key_func: Callable[[T], str]) -> Dict[str, T]:
    """
    Convert a list to a dictionary using a key function.
    
    Args:
        items: The list to convert
        key_func: Function to extract key from each item
        
    Returns:
        Dictionary with keys and items
    """
    return {key_func(item): item for item in items}


def merge_lists_safe(*lists: List[T]) -> List[T]:
    """
    Safely merge multiple lists.
    
    Args:
        *lists: Lists to merge
        
    Returns:
        Merged list
    """
    result = []
    for lst in lists:
        if lst:
            result.extend(lst)
    return result 