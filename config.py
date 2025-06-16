"""
Configuration management for EHR Embeddings project
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for EHR embeddings project"""
    
    # API Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
    
    # Processing Configuration
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    
    # Data Paths
    EHR_DATA_PATH = os.getenv("EHR_DATA_PATH", "data/processed/ehr_data.csv")
    EMBEDDINGS_CACHE_DIR = Path(os.getenv("EMBEDDINGS_CACHE_DIR", "cache/"))
    MODEL_OUTPUT_DIR = Path(os.getenv("MODEL_OUTPUT_DIR", "outputs/models/"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Ensure directories exist
    EMBEDDINGS_CACHE_DIR.mkdir(exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required. Please set it in your environment variables.")
        
        return True 