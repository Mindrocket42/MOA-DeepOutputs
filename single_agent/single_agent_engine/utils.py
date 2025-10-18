"""
Utility functions for single agent workflow.

This module provides helper functions for:
- File operations and prompt reading
- Filename sanitization
- Text processing and markdown utilities
- Logging helpers
"""

import os
import re
from pathlib import Path
from typing import Optional

def read_prompt_from_file(prompt_file: str = "prompt.txt") -> str:
    """
    Read prompt from a file.

    Args:
        prompt_file: Path to the prompt file (default: "prompt.txt")

    Returns:
        The prompt content as a string

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        ValueError: If the prompt file is empty
    """
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file '{prompt_file}' not found")

    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read().strip()

    if not prompt:
        raise ValueError(f"Prompt file '{prompt_file}' is empty")

    return prompt

def sanitize_filename(text: str, max_length: int = 100) -> str:
    """
    Sanitize text to create a safe filename.

    Args:
        text: Text to sanitize
        max_length: Maximum filename length

    Returns:
        Sanitized filename-safe string
    """
    # Remove or replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', text)
    # Replace multiple spaces/underscores with single underscore
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    # Remove leading/trailing underscores and spaces
    sanitized = sanitized.strip('_ ')
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_ ')
    # Ensure it's not empty
    if not sanitized:
        sanitized = "unnamed_prompt"

    return sanitized

def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure

    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def sanitize_for_markdown(text: str) -> str:
    """
    Sanitize text for safe inclusion in markdown.

    Args:
        text: Text to sanitize

    Returns:
        Markdown-safe text
    """
    if not text:
        return ""

    # Escape markdown special characters that might cause formatting issues
    # Be careful not to escape characters that are meant to be markdown
    text = text.replace('\\', '\\\\')  # Escape backslashes first

    # Escape other characters that might interfere
    escape_chars = ['*', '_', '`', '[', ']', '(', ')', '#', '+', '-', '!', '|']
    for char in escape_chars:
        text = text.replace(char, f'\\{char}')

    return text

def format_timestamp_for_filename() -> str:
    """
    Generate a timestamp string suitable for filenames.

    Returns:
        Timestamp string in YYYYMMDD-HHMMSS format
    """
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length with optional suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix

def extract_section(text: str, section_header: str) -> Optional[str]:
    """
    Extract a section from text based on markdown header.

    Args:
        text: Full text to search
        section_header: Header to find (without # markers)

    Returns:
        Section content or None if not found
    """
    # Look for the header pattern
    pattern = rf'^#+\s*{re.escape(section_header)}\s*$'
    match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)

    if not match:
        return None

    # Extract from header to next header or end
    start_pos = match.end()
    next_header_pattern = r'^#+\s*'
    next_match = re.search(next_header_pattern, text[start_pos:], re.MULTILINE)

    if next_match:
        end_pos = start_pos + next_match.start()
    else:
        end_pos = len(text)

    section_content = text[start_pos:end_pos].strip()
    return section_content if section_content else None

def count_tokens_approximate(text: str) -> int:
    """
    Get approximate token count for text.

    This is a rough approximation: ~4 characters per token for English text.

    Args:
        text: Text to count tokens for

    Returns:
        Approximate token count
    """
    return len(text) // 4

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return ".1f"
    elif seconds < 60:
        return ".1f"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return ".0f"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return ".0f"