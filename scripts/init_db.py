import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from football_predictor.db import initialize_database


if __name__ == "__main__":
    initialize_database()
    print("VFootball database initialized.")

