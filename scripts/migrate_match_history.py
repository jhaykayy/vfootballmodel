from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import Session


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from football_predictor.db import create_db_engine, normalize_database_url
from football_predictor.migration import migrate_match_history, read_match_history
from football_predictor.models import CompletedMatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely copy completed-match history from local SQLite to PostgreSQL."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data" / "vfootball.db",
        help="Path to the source SQLite database.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the copy. Without this flag, only connectivity and counts are checked.",
    )
    return parser.parse_args()


def target_match_count(engine) -> int:
    if not inspect(engine).has_table("completed_matches"):
        return 0
    with Session(engine) as session:
        return int(session.scalar(select(func.count(CompletedMatch.id))) or 0)


def main() -> int:
    args = parse_args()
    source_path = args.source.expanduser().resolve()
    if not source_path.is_file():
        print(f"Source SQLite database not found: {source_path}", file=sys.stderr)
        return 2

    supplied_target = os.getenv("VFOOTBALL_DATABASE_URL", "").strip()
    if not supplied_target:
        print("VFOOTBALL_DATABASE_URL is not set in this terminal.", file=sys.stderr)
        return 2
    normalized_target = normalize_database_url(supplied_target)
    if not normalized_target.startswith("postgresql+psycopg://"):
        print("The target must be a PostgreSQL connection string.", file=sys.stderr)
        return 2

    source_engine = create_db_engine(f"sqlite:///{source_path}")
    target_engine = create_db_engine(normalized_target)
    try:
        with target_engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        source_rows = read_match_history(source_engine)
        before = target_match_count(target_engine)
        print("PostgreSQL connection: OK")
        print(f"Local SQLite matches: {len(source_rows)}")
        print(f"PostgreSQL matches before migration: {before}")

        if not args.apply:
            print("Dry run only. Re-run with --apply to copy the matches.")
            return 0

        summary = migrate_match_history(source_engine, target_engine)
        print(f"Created: {summary.created}")
        print(f"Already identical (skipped): {summary.skipped}")
        print(f"PostgreSQL matches after migration: {summary.target_after}")
        print("Verification: all local match-history records exist in PostgreSQL.")
        return 0
    finally:
        source_engine.dispose()
        target_engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
