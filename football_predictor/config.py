from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import toml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    database_url: str
    model_dir: Path
    auto_retrain: bool
    min_training_matches: int
    max_score: int
    random_state: int


def get_settings() -> Settings:
    data_dir = Path(os.getenv("VFOOTBALL_DATA_DIR", PROJECT_ROOT / "data"))
    model_dir = Path(os.getenv("VFOOTBALL_MODEL_DIR", PROJECT_ROOT / "models"))
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    default_db = data_dir / "vfootball.db"
    database_url = os.getenv("VFOOTBALL_DATABASE_URL")
    if not database_url:
        secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
        if secrets_path.is_file():
            secrets = toml.load(secrets_path)
            database_url = secrets.get("VFOOTBALL_DATABASE_URL")
    return Settings(
        database_url=str(database_url or f"sqlite:///{default_db}"),
        model_dir=model_dir,
        auto_retrain=os.getenv("VFOOTBALL_AUTO_RETRAIN", "true").lower() in {"1", "true", "yes"},
        min_training_matches=int(os.getenv("VFOOTBALL_MIN_TRAINING_MATCHES", "20")),
        max_score=int(os.getenv("VFOOTBALL_MAX_SCORE", "10")),
        random_state=int(os.getenv("VFOOTBALL_RANDOM_STATE", "42")),
    )


SUPPORTED_LEAGUES = (
    ("ENG-PL", "England", "Premier League"),
    ("ESP-LL", "Spain", "La Liga"),
    ("FRA-L1", "France", "Ligue 1"),
    ("ITA-SA", "Italy", "Serie A"),
    ("GER-BL", "Germany", "Bundesliga"),
)

