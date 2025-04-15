from pydantic import BaseSettings

class Settings(BaseSettings):
    ALLOW_ORIGINS = ["*"]

def get_settings():
    return Settings()
