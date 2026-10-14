# VFootball Intelligence

VFootball Intelligence is a local, manual-data football prediction application for the Premier League, La Liga, Ligue 1, Serie A, and Bundesliga. It stores results in SQLite, builds every historical feature strictly from information available before kickoff, fits league-specific models, preserves prediction snapshots, and evaluates those predictions when final scores arrive.

It does **not** call a football API and does **not** produce a forecast until a league has enough user-supplied results to fit and validate a model.

## What is included

- Manual completed-match entry, CSV/XLSX imports, duplicate prevention, search, filtering, and score corrections.
- Safe single-record and bulk deletion for completed matches, protected by typed confirmation and followed by Elo rebuilding and model invalidation/retraining.
- Screenshot OCR for completed results and upcoming fixtures, with team-name matching, confidence flags, an editable review table, and a mandatory confirmation step before any database write.
- Chronological Elo history plus recent 5/10-match form, goals, venue form, streaks, opponent strength, rest, league scoring, and sample-limited head-to-head features.
- Separate league models combining Poisson goal regression with calibrated XGBoost outcome probabilities when the dataset supports all three outcome classes. Small datasets fall back transparently to Poisson rather than pretending an advanced model is trained.
- Chronological train/validation/test selection and expanding-window backtesting. A new candidate replaces the active version only when its latest holdout log loss is no worse beyond a small tolerance.
- Internally consistent 1X2, score, totals, and BTTS probabilities. The score matrix is adjusted to share the exact same 1X2 marginals as the selected outcome ensemble.
- Immutable prediction rows with feature snapshots and model-version links. Adding a final result creates an evaluation row for every prediction made before that result.
- Overall and per-league accuracy, exact score, BTTS, over/under, multiclass Brier score, log loss, rolling accuracy, model versions, and training history.

## Installation (Apple Silicon/macOS)

Python 3.10–3.12 is recommended. All declared packages ship Apple Silicon wheels, so a native compiler is normally unnecessary.

```bash
cd /Users/macbook/Documents/vfootball
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
brew install tesseract
python scripts/init_db.py
python scripts/create_templates.py
streamlit run streamlit_app.py
```

The app opens locally, usually at `http://localhost:8501`.

## Normal workflow

1. Open **Add Match Results** and upload historical data or enter results manually.
2. Alternatively, open **Screenshot Import**, choose whether the images contain results or fixtures, select the league, and upload PNG/JPG screenshots. Correct every uncertain field in the review table, then confirm the selected rows. OCR output is never saved automatically.
3. After the configured minimum (20 by default), automatic retraining evaluates a candidate. You can also use **Model Management → Retrain model**.
4. Add one or more matches in **Upcoming Fixtures**, then generate the batch.
5. Review model-backed probabilities and data-reliability notes in **Predictions**.
6. When a fixture finishes, add its completed result with the exact league, date, and teams. Its existing predictions are scored in **Model Performance**; the prediction records themselves are not changed.

### Screenshot tips

- Crop the image around the match list and use the original-resolution screenshot where possible.
- Select the correct league so OCR names can be matched against that league's existing teams.
- Use the fallback date only when the screenshot has no readable date heading.
- Low-confidence review notes are warnings, not guesses to accept blindly. Edit the field before confirming.
- A screenshot batch can be confirmed only once. Database duplicate constraints provide a second protection layer.

For Streamlit Community Cloud, the included `packages.txt` installs the native Tesseract engine. The Python wrapper, Pillow, and RapidFuzz are declared in `requirements.txt`.

### Deleting incorrect historical results

Open **Add Match Results → History & edit**, then expand **Delete completed-match records** below the edit form. You can delete only the selected record or every record matching the current league/team filters. The app displays an exact confirmation phrase such as `DELETE 10`; the delete button remains disabled until that phrase is entered.

Deleting a completed result rebuilds chronological Elo and invalidates models trained on the removed data. Historical prediction snapshots are retained, while result evaluations linked to deleted matches are removed. If enough results remain and automatic retraining is enabled, a replacement model is evaluated automatically.

## Import formats

Completed results require:

```text
league_code,season,match_date,home_team,away_team,home_goals,away_goals
```

Upcoming fixtures require the same fields without the two score columns. Dates should use ISO `YYYY-MM-DD`. Supported league codes are:

| Code | League |
|---|---|
| `ENG-PL` | Premier League |
| `ESP-LL` | La Liga |
| `FRA-L1` | Ligue 1 |
| `ITA-SA` | Serie A |
| `GER-BL` | Bundesliga |

Ready-to-edit CSV and Excel files live in `templates/`.

## Retraining and model promotion

The default threshold is intentionally low enough to exercise the workflow, not to imply that 20 matches are sufficient for a reliable football model. With limited data, Poisson models use league/team defaults and the UI labels reliability accordingly. XGBoost activates only when there are enough examples of home wins, draws, and away wins to fit and calibrate it safely.

For each retrain:

- Features are regenerated in match-date order.
- Training, validation, and test partitions remain chronological.
- Poisson, Elo, calibrated boosting, and the chosen ensemble are measured separately where available.
- Expanding-window backtests refit only on prior matches.
- The candidate is saved even if rejected, preserving the audit trail.
- The deployment artifact is refit on all available results only **after** its design has been evaluated.

Configure behavior with environment variables:

```bash
export VFOOTBALL_AUTO_RETRAIN=true
export VFOOTBALL_MIN_TRAINING_MATCHES=20
export VFOOTBALL_DATA_DIR=/path/to/data
export VFOOTBALL_MODEL_DIR=/path/to/models
export VFOOTBALL_DATABASE_URL=sqlite:////absolute/path/vfootball.db
```

### Shared PostgreSQL database (Supabase)

To share records between a local instance and a Streamlit Community Cloud deployment, use the same Supabase PostgreSQL connection string in both environments. Choose **Session pooler** and **URI** in Supabase's **Connect** dialog. The application accepts either `postgresql://` or `postgresql+psycopg://` and automatically uses Psycopg 3.

For local development, keep the connection string out of source control:

```bash
export VFOOTBALL_DATABASE_URL='postgresql://postgres.PROJECT_REF:PASSWORD@POOLER_HOST:5432/postgres?sslmode=require'
streamlit run streamlit_app.py
```

For Streamlit Community Cloud, open **Manage app → Settings → Secrets** and add:

```toml
VFOOTBALL_DATABASE_URL = "postgresql://postgres.PROJECT_REF:PASSWORD@POOLER_HOST:5432/postgres?sslmode=require"
```

Replace the placeholders with the URI copied from Supabase. Never commit the completed URI or `.streamlit/secrets.toml`; the latter is already ignored by this repository. On first connection, the application creates its schema and five supported leagues automatically. An existing SQLite database is separate and is not copied automatically.

To verify the connection and migrate only completed-match history from the default local SQLite database, first install the current requirements and set the PostgreSQL URI in the same terminal:

```bash
.venv/bin/pip install -r requirements.txt
export VFOOTBALL_DATABASE_URL='YOUR_COMPLETE_SUPABASE_SESSION_POOLER_URI'
.venv/bin/python scripts/migrate_match_history.py
.venv/bin/python scripts/migrate_match_history.py --apply
```

The first migration command is read-only and prints both database counts. The `--apply` command copies matches, skips identical records, rejects conflicting duplicates, rebuilds Elo history, and verifies the copied data. It never deletes or modifies the source SQLite file. The app sidebar also states whether it is using shared PostgreSQL or local SQLite and shows that database's completed-match count.

## Tests

```bash
pytest
```

The suite covers chronological feature leakage, database updates and duplicate prevention, import validation, probability identities, training, immutable prediction history, and result reconciliation.

## Important interpretation notes

- Probabilities describe model uncertainty; they are not guarantees or betting advice.
- A reliability label describes data coverage, not the chance that the selected outcome will happen.
- Predictions for dates at or before an active model's training cutoff are blocked because they could leak future results.
- Promoted/relegated or new teams begin from neutral Elo and league scoring defaults. Their reliability remains limited until their own history grows.
- Head-to-head is used only as a small feature and defaults toward league norms when fewer than three recent meetings exist.

