import os
from pathlib import Path
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Config:
    """Application configuration loaded from environment variables"""
    
    # API Keys
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    # OpenRouter Configuration
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    
    # Dataset Configuration
    MAX_DATASET_SEARCH_RESULTS = int(os.getenv("MAX_DATASET_SEARCH_RESULTS", "10"))
    AUTO_SELECT_DATASET = os.getenv("AUTO_SELECT_DATASET", "true").lower() == "true"
    
    @classmethod
    def validate(cls):
        """Validate that all required configuration is present"""
        missing = []
        
        if not cls.OPENROUTER_API_KEY:
            missing.append("OPENROUTER_API_KEY")
        if not cls.HUGGINGFACEHUB_API_TOKEN:
            missing.append("HUGGINGFACEHUB_API_TOKEN")
        
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                f"Please check your .env file at {env_path}"
            )
        
        logger.info("✓ Configuration validated successfully")
        return True
    
    @classmethod
    def log_config(cls):
        """Log configuration (without exposing sensitive keys)"""
        logger.info("Configuration:")
        logger.info(f"  OpenRouter API: {'✓ Set' if cls.OPENROUTER_API_KEY else '✗ Missing'}")
        logger.info(f"  Hugging Face API: {'✓ Set' if cls.HUGGINGFACEHUB_API_TOKEN else '✗ Missing'}")
        logger.info(f"  Model: {cls.DEFAULT_MODEL}")
        logger.info(f"  Max Dataset Results: {cls.MAX_DATASET_SEARCH_RESULTS}")
        logger.info(f"  Auto-select Dataset: {cls.AUTO_SELECT_DATASET}")


# Validate on import (will raise error if keys are missing)
try:
    Config.validate()
except ValueError as e:
    logger.warning(f"Configuration validation failed: {e}")
    logger.warning("Some features may not work without proper configuration")
