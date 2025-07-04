"""
Batch processing utilities for handling multiple items efficiently.
"""
from typing import List, Tuple, Any, Callable, Optional
from loguru import logger


def categorize_items(
    items: List[Any],
    categories: List[str],
    category_func: Callable[[Any], str]
) -> dict[str, List[Tuple[int, Any]]]:
    """
    Categorize items into different groups with their original indices.
    
    Args:
        items: List of items to categorize
        categories: List of valid categories
        category_func: Function that returns category for an item
        
    Returns:
        Dictionary mapping categories to lists of (index, item) tuples
    """
    categorized = {category: [] for category in categories}
    
    for i, item in enumerate(items):
        try:
            category = category_func(item)
            if category in categorized:
                categorized[category].append((i, item))
            else:
                logger.warning(f"Unknown category '{category}' for item {i}")
        except Exception as e:
            logger.warning(f"Error categorizing item {i}: {e}")
    
    return categorized


def process_batch_items(
    items: List[Any],
    processor_func: Callable[[Any], Any],
    batch_name: str = "items",
    error_handling: str = "skip"
) -> List[Optional[Any]]:
    """
    Process a batch of items with consistent error handling.
    
    Args:
        items: List of items to process
        processor_func: Function to process each item
        batch_name: Name for logging
        error_handling: How to handle errors ('skip', 'fail', 'return_none')
        
    Returns:
        List of processed results (None for failed items)
    """
    results = []
    
    logger.info(f"🔄 Processing {len(items)} {batch_name}")
    
    for i, item in enumerate(items):
        try:
            result = processor_func(item)
            results.append(result)
            logger.debug(f"✅ Processed {batch_name} {i+1}/{len(items)}")
        except Exception as e:
            logger.error(f"❌ Failed to process {batch_name} {i+1}: {str(e)}")
            
            if error_handling == "fail":
                raise
            elif error_handling == "return_none":
                results.append(None)
            elif error_handling == "skip":
                results.append(None)
    
    success_count = sum(1 for r in results if r is not None)
    logger.info(f"✅ Completed {batch_name} processing: {success_count}/{len(items)} successful")
    
    return results


def merge_batch_results(
    original_indices: List[int],
    results: List[Any],
    total_count: int,
    default_value: Any = None
) -> List[Any]:
    """
    Merge batch results back into original order.
    
    Args:
        original_indices: Original indices of items
        results: Results from batch processing
        total_count: Total number of items in original list
        default_value: Value to use for missing items
        
    Returns:
        List of results in original order
    """
    merged = [default_value] * total_count
    
    for idx, result in zip(original_indices, results):
        if idx < total_count:
            merged[idx] = result
        else:
            logger.warning(f"Index {idx} out of bounds for total count {total_count}")
    
    return merged


def validate_batch_request(
    texts: List[str],
    required_params: dict,
    optional_params: dict = None
) -> Tuple[bool, str]:
    """
    Validate batch request parameters.
    
    Args:
        texts: List of texts to process
        required_params: Required parameters
        optional_params: Optional parameters with defaults
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not texts:
        return False, "No texts provided"
    
    if len(texts) > 100:  # Reasonable limit
        return False, "Too many texts (max 100)"
    
    for i, text in enumerate(texts):
        if not text or not text.strip():
            return False, f"Empty text at index {i}"
        
        if len(text) > 10000:  # Reasonable limit
            return False, f"Text too long at index {i} (max 10000 characters)"
    
    # Validate required parameters
    for param, value in required_params.items():
        if value is None:
            return False, f"Missing required parameter: {param}"
    
    return True, ""


def create_batch_response(
    results: List[Any],
    format_type: str = "json",
    include_metadata: bool = True
) -> dict:
    """
    Create a standardized batch response.
    
    Args:
        results: List of results
        format_type: Response format type
        include_metadata: Whether to include metadata
        
    Returns:
        Standardized response dictionary
    """
    response = {
        "results": results,
        "count": len(results),
        "success_count": sum(1 for r in results if r is not None),
        "format": format_type
    }
    
    if include_metadata:
        response["metadata"] = {
            "total_processed": len(results),
            "success_rate": sum(1 for r in results if r is not None) / len(results) if results else 0
        }
    
    return response


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def parallel_process_batch(
    items: List[Any],
    processor_func: Callable[[Any], Any],
    max_workers: int = 4,
    chunk_size: int = 10
) -> List[Any]:
    """
    Process items in parallel with chunking.
    
    Args:
        items: List of items to process
        processor_func: Function to process each item
        max_workers: Maximum number of parallel workers
        chunk_size: Size of each chunk
        
    Returns:
        List of processed results
    """
    import concurrent.futures
    
    chunks = chunk_list(items, chunk_size)
    all_results = []
    
    logger.info(f"🔄 Processing {len(items)} items in {len(chunks)} chunks with {max_workers} workers")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(process_batch_items, chunk, processor_func): i 
            for i, chunk in enumerate(chunks)
        }
        
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_index = future_to_chunk[future]
            try:
                chunk_results = future.result()
                all_results.extend(chunk_results)
                logger.info(f"✅ Completed chunk {chunk_index + 1}/{len(chunks)}")
            except Exception as e:
                logger.error(f"❌ Chunk {chunk_index + 1} failed: {str(e)}")
                # Add None results for failed chunk
                all_results.extend([None] * len(chunks[chunk_index]))
    
    return all_results 