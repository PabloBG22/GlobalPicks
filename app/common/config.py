# app/common/config.py
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field, model_validator


class Settings(BaseSettings):
    footystats_key: str = Field(alias="footystats_key")
    footystats_url: str = Field("https://api.footystats.org", alias="FOOTYSTATS_BASE_URL")
    football_data_api_key: str | None = Field(None, alias="FOOTBALL_DATA_API_KEY")
    football_data_api_url: str = Field("https://api.football-data-api.com", alias="FOOTBALL_DATA_API_URL")
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")
    gp_key: Optional[str] = Field(default=None, alias="GP_KEY")
    telegram_token: Optional[str] = Field(default=None, alias="TELEGRAM_TOKEN")
    telegram_token_file: Optional[str] = Field(default="config/telegram_token.txt", alias="TELEGRAM_TOKEN_FILE")

    @model_validator(mode="after")
    def load_telegram_token_file(self):
        """
        Permite leer el token desde un archivo si no está en el entorno/.env.
        """
        if not self.telegram_token and self.telegram_token_file:
            path = Path(self.telegram_token_file)
            if path.is_file():
                token = path.read_text(encoding="utf-8").strip()
                if token:
                    object.__setattr__(self, "telegram_token", token)
        return self

    class Config:
        env_file = ".env"


# Instancia global para usar en todo el proyecto
settings = Settings()
