import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Logging Configuration ---
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- Environment Variable Debug Logging ---
def log_sa_env_vars():
    """Log single agent environment variables for debugging."""
    logger.info("==== Single Agent Environment Variable Debug ====")
    for key, value in os.environ.items():
        if "SA_" in key or "SINGLE_AGENT" in key:
            logger.info(f"ENV: {key}={value}")
    logger.info("==================================================")

# --- Model Resolution Helper ---
def resolve_sa_model_env(var_name):
    """Resolve model environment variables with symbolic indirection."""
    value = os.getenv(var_name)
    # If the value is the name of another env var, resolve it recursively
    if value and value in os.environ:
        return os.getenv(value)
    return value

# --- Single Agent Model Configuration ---
SA_RESPONSE_AGENT_MODEL = resolve_sa_model_env("SA_RESPONSE_AGENT_MODEL") or os.getenv("SA_RESPONSE_AGENT_MODEL_SYMBOLIC")
SA_DEVILS_ADVOCATE_AGENT_MODEL = resolve_sa_model_env("SA_DEVILS_ADVOCATE_AGENT_MODEL") or os.getenv("SA_DEVILS_ADVOCATE_AGENT_MODEL_SYMBOLIC")
SA_SYNTHESIS_AGENT_MODEL = resolve_sa_model_env("SA_SYNTHESIS_AGENT_MODEL") or os.getenv("SA_SYNTHESIS_AGENT_MODEL_SYMBOLIC")
SA_FINAL_AGENT_MODEL = resolve_sa_model_env("SA_FINAL_AGENT_MODEL") or os.getenv("SA_FINAL_AGENT_MODEL_SYMBOLIC")

# --- Workflow Configuration ---
SA_NUM_LAYERS = int(os.getenv("SA_NUM_LAYERS", "2"))
SA_ENABLE_SELF_REVIEW = os.getenv("SA_ENABLE_SELF_REVIEW", "true").lower() in ("1", "true", "yes")

# --- Directory Configuration ---
SA_OUTPUT_DIR = os.getenv("SA_OUTPUT_DIR", "single_agent/reports")
SA_TRACE_DIR = os.getenv("SA_TRACE_DIR", "single_agent/traces")
SA_DRY_RUN_DIR = os.getenv("SA_DRY_RUN_DIR", "single_agent/dry_runs")

# --- Token Limits per Phase ---
SA_RESPONSE_MAX_TOKENS = int(os.getenv("SA_RESPONSE_MAX_TOKENS", "8000"))
SA_REVIEW_MAX_TOKENS = int(os.getenv("SA_REVIEW_MAX_TOKENS", "4000"))
SA_DEVILS_ADVOCATE_MAX_TOKENS = int(os.getenv("SA_DEVILS_ADVOCATE_MAX_TOKENS", "6000"))
SA_SYNTHESIS_MAX_TOKENS = int(os.getenv("SA_SYNTHESIS_MAX_TOKENS", "8000"))
SA_FINAL_MAX_TOKENS = int(os.getenv("SA_FINAL_MAX_TOKENS", "6000"))

# --- API and Performance Settings ---
# Reuse existing MOA settings for consistency
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_CONCURRENCY = int(os.getenv("OPENROUTER_CONCURRENCY", "5"))
API_TIMEOUT = float(os.getenv("SA_API_TIMEOUT", "120.0"))
API_RETRY_ATTEMPTS = int(os.getenv("SA_API_RETRY_ATTEMPTS", "3"))
API_INITIAL_BACKOFF = float(os.getenv("SA_API_INITIAL_BACKOFF", "1.0"))

# --- HTTP Headers ---
# Reuse existing headers for consistency
HTTP_REFERER = os.getenv("HTTP_REFERER", "https://linktr.ee/mindrocket")
X_TITLE = os.getenv("X_TITLE", "MOADeepOutputs")

logger.info("=== Single Agent Configuration Loaded ===")
logger.info(f"Response Agent Model: {SA_RESPONSE_AGENT_MODEL}")
logger.info(f"Devils Advocate Model: {SA_DEVILS_ADVOCATE_AGENT_MODEL}")
logger.info(f"Synthesis Agent Model: {SA_SYNTHESIS_AGENT_MODEL}")
logger.info(f"Final Agent Model: {SA_FINAL_AGENT_MODEL}")
logger.info(f"Number of Layers: {SA_NUM_LAYERS}")
logger.info(f"Self-Review Enabled: {SA_ENABLE_SELF_REVIEW}")
logger.info("===========================================")