import logging
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

def read_prompt_from_file(file_path: str = "prompt.txt") -> str:
    """
    Reads the prompt from the specified file.

    Args:
        file_path: Path to the prompt file, defaults to 'prompt.txt' in the root folder.

    Returns:
        The prompt as a string.

    Raises:
        FileNotFoundError: If the prompt file doesn't exist.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        logging.info(f"Successfully read prompt from {file_path}")
        return prompt
    except FileNotFoundError:
        logging.error(f"Prompt file not found: {file_path}")
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    except Exception as e:
        logging.error(f"Error reading prompt file: {str(e)}")
        raise

def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    Sanitizes a string to be suitable for use as a filename or folder name.
    Uses the first few words for relevance.
    """
    # Take the first N characters to get context
    context = text[:max_length*2] # Take more initially to get words
    words = context.split()[:5] # Take first 5 words
    if not words:
        return "untitled"
    base_name = "_".join(words)
    # Remove invalid characters
    sanitized = re.sub(r'[^\w\-_\. ]', '', base_name)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Truncate to max_length
    sanitized = sanitized[:max_length]
    # Ensure it's not empty
    if not sanitized:
        return "untitled"
    return sanitized.lower()

def sanitize_for_markdown(text: str) -> str:
    """
    Ensures that text does not break markdown formatting, especially code blocks.
    - Escapes accidental triple backticks by replacing them with a similar sequence.
    - Ensures consistent line endings.
    """
    if not isinstance(text, str):
        return str(text)
    # Replace triple backticks with a similar but safe sequence
    return text.replace('```', '``\u200b`')

def ensure_directory(directory: str) -> Path:
    """
    Ensures that a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory to ensure exists.
        
    Returns:
        Path object for the directory.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Saves data to a JSON file with proper error handling.
    
    Args:
        data: Data to save (must be JSON serializable).
        filepath: Path where to save the JSON file.
        indent: Number of spaces for indentation in the JSON file.
        
    Raises:
        TypeError: If data is not JSON serializable.
        IOError: If there are issues writing to the file.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except TypeError as e:
        logging.error(f"Data is not JSON serializable: {str(e)}")
        raise
    except IOError as e:
        logging.error(f"Error writing to file {filepath}: {str(e)}")
        raise

def load_json(filepath: str) -> Any:
    """
    Loads data from a JSON file with proper error handling.
    
    Args:
        filepath: Path to the JSON file to load.
        
    Returns:
        The loaded data.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"JSON file not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in file {filepath}: {str(e)}")
        raise

def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncates text to a maximum length while preserving word boundaries.
    
    Args:
        text: Text to truncate.
        max_length: Maximum length of the truncated text.
        suffix: String to append to truncated text.
        
    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    
    # Find the last space before max_length
    last_space = text[:max_length].rfind(' ')
    if last_space == -1:
        return text[:max_length] + suffix
    
    return text[:last_space] + suffix

def format_timestamp(timestamp: Optional[float] = None) -> str:
    """
    Formats a timestamp in a consistent way.
    
    Args:
        timestamp: Unix timestamp (seconds since epoch). If None, uses current time.
        
    Returns:
        Formatted timestamp string.
    """
    from datetime import datetime
    if timestamp is None:
        timestamp = datetime.now().timestamp()
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")