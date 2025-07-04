"""
Configuration settings for the application
"""
import os
from functools import lru_cache
from typing import List
import dotenv
from entry.utils.string_utils import parse_comma_separated_string, build_path

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

def parse_bool_env(var_name: str, default: str = "False") -> bool:
    """Parse boolean environment variable with consistent logic"""
    return os.getenv(var_name, default).lower() in ("true", "1", "t")


def get_env_path(base_path: str, sub_path: str = "") -> str:
    """Get environment-based path with fallback to current directory"""
    env_path = os.getenv(base_path)
    if env_path:
        return build_path(env_path, sub_path) if sub_path else env_path
    return build_path(os.getcwd(), sub_path) if sub_path else os.getcwd()


def is_container_environment() -> bool:
    """Check if running in container environment"""
    return (parse_bool_env('CONTAINER_ENV', 'false') or 
            os.getenv('K_SERVICE', '').lower() != '')


def get_models_directory() -> str:
    """Get models directory path with fallback logic"""
    models_dir = os.getenv('MODELS_DIR', 'models')
    path = get_env_path('MODELS_DIR', models_dir)
    
    # Ensure we always return an absolute path
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    
    return path


class Settings:
    """Application settings"""
    
    def __init__(self):
        # Environment
        self.env = os.getenv("ENV", "development")
        
        # Security
        self.api_key = os.getenv("API_KEY", "dev-secret-key")
        
        # CORS
        origins_str = os.getenv("ALLOWED_ORIGINS", "*")
        self.allowed_origins = parse_comma_separated_string(origins_str)
        
        # Models
        self.models_dir = get_models_directory()
        self.cuda_available = parse_bool_env("CUDA_AVAILABLE", "True")
        
        # Hugging Face
        self.hf_token = os.getenv("HF_TOKEN", "")
        self.offline_mode = parse_bool_env("OFFLINE_MODE", "False")
        
        # Container environment detection
        self.is_container = is_container_environment()

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
