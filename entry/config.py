"""
Configuration settings for the application
"""
import os
from functools import lru_cache
from typing import List
import dotenv

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

class Settings:
    """Application settings"""
    
    def __init__(self):
        # Environment
        self.env = os.getenv("ENV", "development")
        
        # Security
        self.api_key = os.getenv("API_KEY", "dev-secret-key")
        
        # CORS
        # Handle allowed origins as a simple comma-separated string
        allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
        self.allowed_origins = self._parse_allowed_origins(allowed_origins_str)
        
        # Models
        self.models_dir = os.getenv("MODELS_DIR", "models")
        self.cuda_available = os.getenv("CUDA_AVAILABLE", "True").lower() in ("true", "1", "t")
        
        # Hugging Face
        self.hf_token = os.getenv("HF_TOKEN", "")
        self.offline_mode = os.getenv("OFFLINE_MODE", "False").lower() in ("true", "1", "t")
        
        # Set models directory based on availability
        self._setup_models_dir()
    
    def _parse_allowed_origins(self, origins_str):
        """Parse comma-separated string into list of origins"""
        if not origins_str:
            return ["*"]  # Default to allow all origins
        
        # Special case: if origins_str is exactly '*', allow all origins
        if origins_str.strip() == "*":
            return ["*"]
            
        origins = [origin.strip() for origin in origins_str.split(",") if origin.strip()]
        return origins if origins else ["*"]
    
    def _setup_models_dir(self):
        """Set up models directory based on availability"""
        models_dir_path = os.path.join(os.getcwd(), self.models_dir)
        if not os.path.exists(models_dir_path) and os.path.exists('/app/models'):
            self.models_dir = '/app/models'
        else:
            self.models_dir = models_dir_path


@lru_cache()
def get_settings():
    """Get cached settings instance"""
    return Settings()
