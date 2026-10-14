from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from football_predictor.db import create_db_engine, normalize_database_url


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save a verified PostgreSQL URI for persistent local Streamlit use."
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing local secrets file.",
    )
    args = parser.parse_args()
    target = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if target.exists() and not args.replace:
        print(
            f"{target} already exists. Re-run with --replace only if you intend to change it.",
            file=sys.stderr,
        )
        return 2

    uri = getpass.getpass("Paste the complete Supabase Session pooler URI (hidden): ").strip()
    normalized = normalize_database_url(uri)
    if not normalized.startswith("postgresql+psycopg://"):
        print("That is not a PostgreSQL URI.", file=sys.stderr)
        return 2

    engine = create_db_engine(uri)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        print(f"Connection failed: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"VFOOTBALL_DATABASE_URL = {json.dumps(uri)}\n", encoding="utf-8")
    os.chmod(target, 0o600)
    print("PostgreSQL connection: OK")
    print(f"Saved persistent local configuration to {target}")
    print("Future `streamlit run streamlit_app.py` sessions will use this shared database.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
