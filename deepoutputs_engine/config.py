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
def resolve_model_env(var_name):
    value = os.getenv(var_name)
    # If the value is the name of another env var, resolve it recursively
    if value and value in os.environ:
        return os.getenv(value)
    return value

AGENT1_MODEL = resolve_model_env("AGENT1_MODEL") or os.getenv("AGENT1_MODEL_SYMBOLIC")
AGENT2_MODEL = resolve_model_env("AGENT2_MODEL") or os.getenv("AGENT2_MODEL_SYMBOLIC")
AGENT3_MODEL = resolve_model_env("AGENT3_MODEL") or os.getenv("AGENT3_MODEL_SYMBOLIC")
DEEP_RESEARCH_AGENT_MODEL = resolve_model_env("DEEP_RESEARCH_AGENT_MODEL") or os.getenv("DEEP_RESEARCH_AGENT_MODEL_SYMBOLIC")
SYNTHESIS_AGENT_MODEL = resolve_model_env("SYNTHESIS_AGENT_MODEL") or os.getenv("SYNTHESIS_AGENT_MODEL_SYMBOLIC")
DEVILS_ADVOCATE_AGENT_MODEL = resolve_model_env("DEVILS_ADVOCATE_AGENT_MODEL") or os.getenv("DEVILS_ADVOCATE_AGENT_MODEL_SYMBOLIC")
FINAL_AGENT_MODEL = resolve_model_env("FINAL_AGENT_MODEL") or os.getenv("FINAL_AGENT_MODEL_SYMBOLIC")

# --- API and Concurrency Settings ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_CONCURRENCY = int(os.getenv("OPENROUTER_CONCURRENCY", "5"))  # Max concurrent requests
API_RETRY_ATTEMPTS = int(os.getenv("API_RETRY_ATTEMPTS", "3"))  # Max retries on specific errors
API_INITIAL_BACKOFF = float(os.getenv("API_INITIAL_BACKOFF", "1.0"))  # Initial delay in seconds for retry
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "120.0"))  # Timeout for API calls

# --- HTTP Headers ---
HTTP_REFERRER = "linktr.ee/mindrocket"  # Hardcoded site URL
X_TITLE = "MOA-DeepOutputs"  # Hardcoded app name

# --- Stage-specific max_tokens settings ---
INITIAL_MAX_TOKENS = None  # Let API default apply
AGGREGATION_MAX_TOKENS = None  # Let API default apply
SYNTHESIS_MAX_TOKENS = None  # Let API default apply
DEVILS_ADVOCATE_MAX_TOKENS = int(os.getenv("DEVILS_ADVOCATE_MAX_TOKENS", "32000"))
FINAL_MAX_TOKENS = int(os.getenv("FINAL_MAX_TOKENS", "6000"))

# --- MOA Orchestration Settings ---
MOA_NUM_LAYERS = int(os.getenv("MOA_NUM_LAYERS", "2"))
INCLUDE_DEEP_RESEARCH = os.getenv("INCLUDE_DEEP_RESEARCH", "true").lower() in ("1", "true", "yes")

# --- Output Settings ---
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "reports")  # Directory for saving reports and outputs