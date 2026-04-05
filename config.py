"""Centralized runtime configuration for external services.

This module loads environment variables from `.env` and exposes service
credentials/settings in one place.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    def load_dotenv(*_args, **_kwargs):
        return False

load_dotenv()


# Required for enabling upstream service integrations in the current codebase.
REQUIRED_ENV_VARS: List[str] = [
    "NEWS_API_KEY",
]

NEWS_API_KEY: Optional[str] = os.getenv("NEWS_API_KEY")


def validate_config(required: Optional[List[str]] = None) -> None:
    """Validate required environment variables.

    Args:
        required: Optional list of env var names to validate.
            Defaults to REQUIRED_ENV_VARS.

    Raises:
        ValueError: If one or more required variables are missing.
    """

    required_vars = required or REQUIRED_ENV_VARS
    missing = [name for name in required_vars if not os.getenv(name)]
    if missing:
        raise ValueError(f"Missing environment variables: {missing}")


def get_active_services() -> Dict[str, bool]:
    """Return active/inactive status for integrated external services."""

    return {
        "news_api": bool(NEWS_API_KEY),
    }
