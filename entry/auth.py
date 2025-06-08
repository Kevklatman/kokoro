"""
Authentication utilities
"""
from fastapi import HTTPException, Header, Depends
from config import get_settings


async def verify_api_key(x_api_key: str = Header(...)):
    """Verify API key from header"""
    settings = get_settings()
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")


def get_api_key_dependency():
    """Get API key dependency for protected endpoints"""
    return Depends(verify_api_key)