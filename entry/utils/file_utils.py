"""
File system utilities for consistent file operations across the application.
"""
import os
from typing import List, Optional, Tuple
from pathlib import Path


def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception:
        return False


def safe_file_exists(file_path: str) -> bool:
    """
    Safely check if file exists.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file exists
    """
    try:
        return os.path.exists(file_path)
    except Exception:
        return False


def safe_directory_exists(directory_path: str) -> bool:
    """
    Safely check if directory exists.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        True if directory exists
    """
    try:
        return os.path.isdir(directory_path)
    except Exception:
        return False


def get_file_info(file_path: str) -> Optional[dict]:
    """
    Get file information safely.
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file info or None if error
    """
    try:
        if not os.path.exists(file_path):
            return None
        
        stat = os.stat(file_path)
        return {
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'created': stat.st_ctime,
            'is_file': os.path.isfile(file_path),
            'is_dir': os.path.isdir(file_path)
        }
    except Exception:
        return None


def list_directory_contents(directory_path: str, file_extensions: List[str] = None) -> List[str]:
    """
    List directory contents with optional filtering.
    
    Args:
        directory_path: Path to directory
        file_extensions: List of file extensions to include (without dots)
        
    Returns:
        List of file/directory names
    """
    try:
        if not os.path.exists(directory_path):
            return []
        
        contents = os.listdir(directory_path)
        
        if file_extensions:
            contents = [
                item for item in contents
                if os.path.isfile(os.path.join(directory_path, item)) and
                any(item.lower().endswith(f".{ext.lower()}") for ext in file_extensions)
            ]
        
        return sorted(contents)
    except Exception:
        return []


def find_files_by_pattern(directory_path: str, pattern: str) -> List[str]:
    """
    Find files matching pattern in directory.
    
    Args:
        directory_path: Path to directory
        pattern: File pattern to match
        
    Returns:
        List of matching file paths
    """
    try:
        import glob
        pattern_path = os.path.join(directory_path, pattern)
        return glob.glob(pattern_path)
    except Exception:
        return []


def get_relative_path(file_path: str, base_path: str) -> str:
    """
    Get relative path from base path.
    
    Args:
        file_path: Full file path
        base_path: Base directory path
        
    Returns:
        Relative path
    """
    try:
        return os.path.relpath(file_path, base_path)
    except Exception:
        return file_path


def get_absolute_path(file_path: str, base_path: str = None) -> str:
    """
    Get absolute path.
    
    Args:
        file_path: File path (relative or absolute)
        base_path: Base directory for relative paths
        
    Returns:
        Absolute path
    """
    try:
        if os.path.isabs(file_path):
            return file_path
        
        if base_path:
            return os.path.abspath(os.path.join(base_path, file_path))
        
        return os.path.abspath(file_path)
    except Exception:
        return file_path


def safe_read_file(file_path: str, encoding: str = 'utf-8') -> Optional[str]:
    """
    Safely read file contents.
    
    Args:
        file_path: Path to file
        encoding: File encoding
        
    Returns:
        File contents or None if error
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception:
        return None


def safe_write_file(file_path: str, content: str, encoding: str = 'utf-8') -> bool:
    """
    Safely write content to file.
    
    Args:
        file_path: Path to file
        content: Content to write
        encoding: File encoding
        
    Returns:
        True if successful
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory:
            ensure_directory_exists(directory)
        
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception:
        return False


def safe_delete_file(file_path: str) -> bool:
    """
    Safely delete file.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if successful
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception:
        return False


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in MB.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    try:
        if os.path.exists(file_path):
            return os.path.getsize(file_path) / (1024 * 1024)
        return 0.0
    except Exception:
        return 0.0


def is_file_older_than(file_path: str, days: int) -> bool:
    """
    Check if file is older than specified days.
    
    Args:
        file_path: Path to file
        days: Number of days
        
    Returns:
        True if file is older
    """
    try:
        import time
        if not os.path.exists(file_path):
            return True
        
        file_time = os.path.getmtime(file_path)
        current_time = time.time()
        days_old = (current_time - file_time) / (24 * 3600)
        
        return days_old > days
    except Exception:
        return True


def create_temp_file(prefix: str = "temp", suffix: str = "", directory: str = None) -> str:
    """
    Create temporary file.
    
    Args:
        prefix: File prefix
        suffix: File suffix
        directory: Directory to create file in
        
    Returns:
        Path to temporary file
    """
    try:
        import tempfile
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=directory)
        os.close(fd)
        return path
    except Exception:
        return None


def copy_file_safe(source_path: str, dest_path: str) -> bool:
    """
    Safely copy file.
    
    Args:
        source_path: Source file path
        dest_path: Destination file path
        
    Returns:
        True if successful
    """
    try:
        import shutil
        # Ensure destination directory exists
        dest_dir = os.path.dirname(dest_path)
        if dest_dir:
            ensure_directory_exists(dest_dir)
        
        shutil.copy2(source_path, dest_path)
        return True
    except Exception:
        return False


def get_directory_size(directory_path: str) -> int:
    """
    Get total size of directory in bytes.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        Total size in bytes
    """
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size
    except Exception:
        return 0 