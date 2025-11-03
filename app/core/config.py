from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    STORAGE_BACKEND: str = "none"
    REEL_KEYS_COUNT: int = 1
    GEMINI_API_KEY: str = "YOUR_API_KEY_HERE"

    class Config:
        env_file = ".env"

settings = Settings()
