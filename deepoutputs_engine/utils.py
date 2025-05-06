import logging
import re

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