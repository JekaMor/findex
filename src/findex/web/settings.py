"""Configuration from the environment (12-factor, rule III), validated by Pydantic.

    INDEX_PATH=web.bin LOG_LEVEL=DEBUG findex serve

A local ``.env`` file is read too, for development; the same image runs in
production with the platform's environment variables instead.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # FilePath: a missing file fails validation when the app starts, not on
    # the first search.
    index_path: FilePath = Field(default=Path("index.bin"), validate_default=True)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    max_page_size: int = Field(default=50, ge=1, le=200)
    site_name: str = "findex"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read once per process; tests override the dependency instead."""
    return Settings()
