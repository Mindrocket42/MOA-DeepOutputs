import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Logging Configuration (to be moved to main or a logging module if needed) ---
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- Environment Variable Debug Logging (optional, can be toggled) ---
def log_env_vars():
    logger.info("==== Environment Variable Debug ====")
    for key, value in os.environ.items():
        if "MODEL" in key or "AGENT" in key:
            logger.info(f"ENV: {key}={value}")
    logger.info("===================================")

# --- Agent Model Configuration ---
AGENT1_MODEL = os.getenv("AGENT1_MODEL_SYMBOLIC") or os.getenv("AGENT1_MODEL")
AGENT2_MODEL = os.getenv("AGENT2_MODEL_SYMBOLIC") or os.getenv("AGENT2_MODEL")
AGENT3_MODEL = os.getenv("AGENT3_MODEL_SYMBOLIC") or os.getenv("AGENT3_MODEL")
DEEP_RESEARCH_AGENT_MODEL = os.getenv("DEEP_RESEARCH_AGENT_MODEL_SYMBOLIC") or os.getenv("DEEP_RESEARCH_AGENT_MODEL")
SYNTHESIS_AGENT_MODEL = os.getenv("SYNTHESIS_AGENT_MODEL_SYMBOLIC") or os.getenv("SYNTHESIS_AGENT_MODEL")
DEVILS_ADVOCATE_AGENT_MODEL = os.getenv("DEVILS_ADVOCATE_AGENT_MODEL_SYMBOLIC") or os.getenv("DEVILS_ADVOCATE_AGENT_MODEL")
FINAL_AGENT_MODEL = os.getenv("FINAL_AGENT_MODEL_SYMBOLIC") or os.getenv("FINAL_AGENT_MODEL")

# --- API and Concurrency Settings ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_CONCURRENCY = int(os.getenv("OPENROUTER_CONCURRENCY", "5"))  # Max concurrent requests
API_RETRY_ATTEMPTS = int(os.getenv("API_RETRY_ATTEMPTS", "3"))  # Max retries on specific errors
API_INITIAL_BACKOFF = float(os.getenv("API_INITIAL_BACKOFF", "1.0"))  # Initial delay in seconds for retry
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "120.0"))  # Timeout for API calls

# --- HTTP Headers ---
HTTP_REFERER = os.getenv("HTTP_REFERER", "MOA_Demo/1.0")  # Recommended: Your App Name/Version
X_TITLE = os.getenv("X_TITLE", "MOA Demo")  # Recommended: Your App Name

# --- Stage-specific max_tokens settings ---
INITIAL_MAX_TOKENS = None  # Let API default apply
AGGREGATION_MAX_TOKENS = None  # Let API default apply
SYNTHESIS_MAX_TOKENS = None  # Let API default apply
DEVILS_ADVOCATE_MAX_TOKENS = int(os.getenv("DEVILS_ADVOCATE_MAX_TOKENS", "32000"))
FINAL_MAX_TOKENS = int(os.getenv("FINAL_MAX_TOKENS", "6000"))