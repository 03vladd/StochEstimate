from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://stochestimate:stochestimate_dev@localhost:5433/stochestimate_app"

    # Auth
    SECRET_KEY: str = "dev_secret_change_in_production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 43200  # 30 days

    # Email
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM: str = "noreply@stochestimate.com"

    # Frontend
    FRONTEND_URL: str = "http://localhost:5173"

    # Anthropic
    ANTHROPIC_API_KEY: str = ""

    # App
    APP_NAME: str = "StochEstimate"
    DEBUG: bool = True


settings = Settings()
