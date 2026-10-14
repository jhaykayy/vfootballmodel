from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import func, select

from football_predictor.analytics import accuracy_by_league, dashboard_summary, performance_rows
from football_predictor.config import PROJECT_ROOT, get_settings
from football_predictor.db import get_engine, initialize_database, session_scope
from football_predictor.exceptions import FootballPredictorError, InsufficientDataError
from football_predictor.importers import parse_fixture_upload, parse_match_upload
from football_predictor.models import (
    CompletedMatch,
    League,
    ModelMetric,
    ModelVersion,
    Prediction,
    Season,
    Team,
    TrainingRun,
    UpcomingFixture,
)
from football_predictor.ocr import (
    fixture_inputs_from_review,
    match_inputs_from_review,
    ocr_is_available,
    parse_screenshot,
)
from football_predictor.prediction import generate_bulk_predictions
from football_predictor.services import (
    FixtureInput,
    MatchInput,
    add_completed_match,
    add_completed_matches,
    add_upcoming_fixture,
    delete_completed_matches,
    find_match_ids,
    get_league,
    list_match_rows,
    update_completed_match,
    update_upcoming_fixture_kickoff,
)
from football_predictor.time_utils import display_kickoff, kickoff_parts, local_now
from football_predictor.training import train_league


st.set_page_config(page_title="VFootball Intelligence", page_icon="⚽", layout="wide")
initialize_database()
settings = get_settings()

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 3rem;}
      [data-testid="stMetric"] {background: #101923; border: 1px solid #243244; padding: 14px; border-radius: 12px;}
      .vf-card {background: linear-gradient(135deg,#101923,#152638); border:1px solid #29415a;
                border-radius:14px; padding:18px; margin:8px 0 14px 0;}
      .vf-muted {color:#91a4b7; font-size:.9rem;}
      .vf-prob {font-size:1.8rem; font-weight:700; color:#5ee6a8;}
    </style>
    """,
    unsafe_allow_html=True,
)


def league_options() -> dict[str, str]:
    with session_scope() as session:
        leagues = session.scalars(select(League).order_by(League.country)).all()
        return {f"{item.country} — {item.name}": item.code for item in leagues}


def team_names(league_code: str) -> list[str]:
    with session_scope() as session:
        league = get_league(session, league_code)
        return list(
            session.scalars(select(Team.name).where(Team.league_id == league.id).order_by(Team.name)).all()
        )


def retrain_code(league_code: str, trigger: str) -> str:
    with session_scope() as session:
        league = get_league(session, league_code)
        version = train_league(session, league.id, trigger=trigger)
        return (
            f"{league.name}: {version.version} promoted"
            if version.promoted
            else f"{league.name}: candidate evaluated; existing active model retained"
        )


def auto_retrain(codes: set[str]) -> list[str]:
    messages = []
    if not settings.auto_retrain:
        return ["Automatic retraining is disabled by configuration."]
    for code in sorted(codes):
        try:
            messages.append(retrain_code(code, "automatic"))
        except InsufficientDataError as exc:
            messages.append(str(exc))
        except FootballPredictorError as exc:
            messages.append(f"Automatic retraining failed for {code}: {exc}")
    return messages


LEAGUES = league_options()
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Add Match Results",
        "Screenshot Import",
        "Upcoming Fixtures",
        "Predictions",
        "Model Performance",
        "Model Management",
    ],
)
st.sidebar.caption("Probabilities are model estimates, not guarantees.")
with session_scope() as session:
    database_match_count = int(session.scalar(select(func.count(CompletedMatch.id))) or 0)
if get_engine().dialect.name == "postgresql":
    st.sidebar.success("Shared PostgreSQL connected")
else:
    st.sidebar.info("Local SQLite database")
st.sidebar.caption(f"{database_match_count:,} completed matches in this database")


if page == "Dashboard":
    st.title("⚽ VFootball Intelligence")
    st.caption("Leak-safe, continuously updated match forecasting for five European leagues")
    with session_scope() as session:
        summary = dashboard_summary(session)
        by_league = accuracy_by_league(session)
    columns = st.columns(4)
    columns[0].metric("Historical matches", f"{summary['historical_matches']:,}")
    columns[1].metric("Predictions stored", f"{summary['predictions']:,}")
    columns[2].metric(
        "Evaluated 1X2 accuracy",
        f"{summary['accuracy']:.1%}" if summary["accuracy"] is not None else "Not available",
    )
    columns[3].metric(
        "Latest training",
        summary["latest_training"].strftime("%d %b %Y") if summary["latest_training"] else "Not trained",
    )
    if summary["historical_matches"] == 0:
        st.info("Start by uploading historical results in **Add Match Results**. No predictions are produced without fitted data.")
    if by_league:
        frame = pd.DataFrame(by_league)
        st.subheader("Accuracy by league")
        st.dataframe(
            frame.style.format(
                {
                    "1x2_accuracy": "{:.1%}",
                    "exact_score_accuracy": "{:.1%}",
                    "btts_accuracy": "{:.1%}",
                    "over_25_accuracy": "{:.1%}",
                    "brier_score": "{:.3f}",
                    "log_loss": "{:.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )


elif page == "Add Match Results":
    st.title("Add match results")
    manual_tab, upload_tab, history_tab = st.tabs(["Manual entry", "CSV / Excel upload", "History & edit"])
    with manual_tab:
        with st.form("manual_result", clear_on_submit=True):
            league_label = st.selectbox("League", list(LEAGUES))
            season = st.text_input("Season", placeholder="2025/26")
            kickoff_left, kickoff_right = st.columns(2)
            match_date = kickoff_left.date_input("Match date", max_value=date.today())
            match_time = kickoff_right.time_input(
                "Kickoff time",
                value=local_now().time().replace(second=0, microsecond=0),
            )
            left, right = st.columns(2)
            home_team = left.text_input("Home team")
            away_team = right.text_input("Away team")
            home_goals = left.number_input("Home goals", min_value=0, max_value=30, step=1)
            away_goals = right.number_input("Away goals", min_value=0, max_value=30, step=1)
            submitted = st.form_submit_button("Save completed match", type="primary")
        if submitted:
            code = LEAGUES[league_label]
            try:
                with session_scope() as session:
                    add_completed_match(
                        session,
                        MatchInput(
                            code,
                            season,
                            match_date,
                            home_team,
                            away_team,
                            int(home_goals),
                            int(away_goals),
                            kickoff_time=match_time,
                        ),
                    )
                st.success("Match saved and league statistics recalculated.")
                for message in auto_retrain({code}):
                    st.caption(message)
            except FootballPredictorError as exc:
                st.error(str(exc))
    with upload_tab:
        st.write(
            "Required columns: `league_code`, `season`, `match_date`, `home_team`, `away_team`, "
            "`home_goals`, `away_goals`. Add `kickoff_time` as HH:MM; omitted times default to 00:00."
        )
        t1, t2 = st.columns(2)
        t1.download_button(
            "Download CSV template",
            data=(PROJECT_ROOT / "templates/match_results_template.csv").read_bytes(),
            file_name="match_results_template.csv",
        )
        xlsx_path = PROJECT_ROOT / "templates/match_results_template.xlsx"
        if xlsx_path.exists():
            t2.download_button("Download Excel template", data=xlsx_path.read_bytes(), file_name=xlsx_path.name)
        uploaded = st.file_uploader("Historical results", type=["csv", "xlsx"], key="match_upload")
        if uploaded and st.button("Validate and import", type="primary"):
            try:
                rows = parse_match_upload(uploaded, uploaded.name)
                with session_scope() as session:
                    add_completed_matches(session, rows)
                st.success(f"Imported {len(rows):,} matches. Elo history and outcomes were recalculated.")
                for message in auto_retrain({row.league_code for row in rows}):
                    st.caption(message)
            except (FootballPredictorError, ValueError) as exc:
                st.error(str(exc))
    with history_tab:
        if "history_delete_flash" in st.session_state:
            st.success(st.session_state.pop("history_delete_flash"))
        filter_col, search_col = st.columns(2)
        filter_label = filter_col.selectbox("Filter league", ["All"] + list(LEAGUES), key="history_league")
        search = search_col.text_input("Search teams")
        code = None if filter_label == "All" else LEAGUES[filter_label]
        with session_scope() as session:
            rows = list_match_rows(session, league_code=code, search=search)
            filtered_match_ids = find_match_ids(session, league_code=code, search=search)
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            labels = {
                f"#{row['id']} · {row['date']} {row['kickoff_time']:%H:%M} · "
                f"{row['home_team']} {row['home_goals']}–{row['away_goals']} {row['away_team']}": row
                for row in rows
            }
            selected_label = st.selectbox("Edit a record", list(labels))
            selected = labels[selected_label]
            with st.form("edit_match"):
                e1, e2, e3, e4, e5 = st.columns(5)
                edited_date = e1.date_input("Date", value=selected["date"])
                edited_time = e2.time_input("Kickoff", value=selected["kickoff_time"])
                edited_season = e3.text_input("Season", value=selected["season"])
                edited_home = e4.number_input("Home goals", 0, 30, selected["home_goals"])
                edited_away = e5.number_input("Away goals", 0, 30, selected["away_goals"])
                save_edit = st.form_submit_button("Save correction")
            if save_edit:
                try:
                    with session_scope() as session:
                        update_completed_match(
                            session,
                            selected["id"],
                            match_date=edited_date,
                            season_name=edited_season,
                            home_goals=int(edited_home),
                            away_goals=int(edited_away),
                            kickoff_time=edited_time,
                        )
                    st.success("Correction saved; chronological Elo and linked prediction outcomes were recalculated.")
                    for message in auto_retrain({selected["league_code"]}):
                        st.caption(message)
                except FootballPredictorError as exc:
                    st.error(str(exc))

            with st.expander("Delete completed-match records", expanded=False):
                st.warning(
                    "Deletion is permanent. Linked prediction snapshots are preserved, but their result evaluations "
                    "are removed. Elo is rebuilt and affected active models are invalidated before automatic retraining."
                )
                delete_scope = st.radio(
                    "Delete scope",
                    [
                        "Selected record only",
                        f"All {len(filtered_match_ids):,} records matching the current filters",
                    ],
                    key="history_delete_scope",
                )
                target_ids = (
                    [selected["id"]]
                    if delete_scope == "Selected record only"
                    else filtered_match_ids
                )
                confirmation_phrase = f"DELETE {len(target_ids)}"
                st.caption(
                    f"Type `{confirmation_phrase}` exactly to enable deletion. "
                    "Use the league and team filters above to narrow a bulk deletion."
                )
                typed_confirmation = st.text_input(
                    "Deletion confirmation",
                    key="history_delete_confirmation",
                    placeholder=confirmation_phrase,
                )
                if st.button(
                    f"Permanently delete {len(target_ids):,} record(s)",
                    disabled=typed_confirmation.strip() != confirmation_phrase or not target_ids,
                    key="history_delete_button",
                ):
                    try:
                        with session_scope() as session:
                            summary = delete_completed_matches(session, target_ids)
                        retraining_messages = auto_retrain(set(summary.league_codes))
                        detail = (
                            f"Deleted {summary.deleted_matches:,} completed-match record(s), removed "
                            f"{summary.removed_evaluations:,} linked evaluation(s), and rebuilt Elo."
                        )
                        if retraining_messages:
                            detail += " " + " ".join(retraining_messages)
                        st.session_state["history_delete_flash"] = detail
                        st.rerun()
                    except FootballPredictorError as exc:
                        st.error(str(exc))
        else:
            st.info("No historical matches match these filters.")


elif page == "Screenshot Import":
    st.title("Import matches from screenshots")
    st.caption("OCR never writes directly to the database. Review and confirm every extracted row first.")

    setup_left, setup_right = st.columns(2)
    mode_label = setup_left.radio(
        "Screenshot contains",
        ["Completed results", "Upcoming fixtures"],
        horizontal=True,
    )
    screenshot_league_label = setup_right.selectbox("League", list(LEAGUES), key="ocr_league")
    screenshot_code = LEAGUES[screenshot_league_label]
    screenshot_teams = team_names(screenshot_code)
    details_left, details_middle, details_time, details_right = st.columns(4)
    screenshot_season = details_left.text_input("Season", placeholder="2025/26", key="ocr_season")
    default_screenshot_date = details_middle.date_input(
        "Fallback date",
        value=date.today(),
        help="Used only when a date cannot be read from the screenshot.",
    )
    default_screenshot_time = details_time.time_input(
        "Fallback kickoff",
        value=local_now().time().replace(second=0, microsecond=0),
        help="Used only when a kickoff time cannot be read from the screenshot.",
    )
    day_first = details_right.checkbox(
        "Dates use day/month/year",
        value=True,
        help="Turn this off for month/day/year screenshots.",
    )
    if not screenshot_teams:
        st.warning(
            "This league has no existing teams yet. OCR can still extract names, but automatic team-name matching will be limited."
        )
    if not ocr_is_available():
        st.warning(
            "Tesseract is not installed on this computer. Run `brew install tesseract` locally. "
            "Streamlit Community Cloud installs it from the included packages.txt file."
        )

    screenshot_files = st.file_uploader(
        "Upload screenshots",
        type=["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"],
        accept_multiple_files=True,
        key="ocr_images",
        help="Crop screenshots around the match list where practical. Each image must be 12 MB or smaller.",
    )
    if screenshot_files:
        preview_columns = st.columns(min(3, len(screenshot_files)))
        for index, uploaded_image in enumerate(screenshot_files):
            with preview_columns[index % len(preview_columns)]:
                st.image(uploaded_image, caption=uploaded_image.name, width="stretch")

    parse_mode = "results" if mode_label == "Completed results" else "fixtures"
    if st.button(
        "Read screenshots",
        type="primary",
        disabled=not screenshot_files or not ocr_is_available(),
    ):
        extracted_frames = []
        raw_outputs = []
        extraction_warnings = []
        try:
            if len(screenshot_files) > 10:
                raise FootballPredictorError("Import at most 10 screenshots in one review batch.")
            for screenshot_file in screenshot_files:
                parsed = parse_screenshot(
                    screenshot_file.getvalue(),
                    mode=parse_mode,
                    known_teams=screenshot_teams,
                    default_date=default_screenshot_date,
                    source_image=screenshot_file.name,
                    day_first=day_first,
                    default_time=default_screenshot_time,
                )
                if not parsed.rows.empty:
                    extracted_frames.append(parsed.rows)
                raw_outputs.append(
                    {
                        "image": screenshot_file.name,
                        "confidence": parsed.mean_ocr_confidence,
                        "text": parsed.raw_text,
                    }
                )
                extraction_warnings.extend(
                    f"{screenshot_file.name}: {warning}" for warning in parsed.warnings
                )
            if extracted_frames:
                review_frame = pd.concat(extracted_frames, ignore_index=True)
            else:
                base_columns = ["include", "match_date", "kickoff_time", "home_team", "away_team"]
                if parse_mode == "results":
                    base_columns += ["home_goals", "away_goals"]
                base_columns += ["confidence", "review", "source_image", "source_text"]
                review_frame = pd.DataFrame(columns=base_columns)
            st.session_state["ocr_review_frame"] = review_frame
            st.session_state.pop("ocr_review_editor", None)
            st.session_state["ocr_raw_outputs"] = raw_outputs
            st.session_state["ocr_warnings"] = extraction_warnings
            st.session_state["ocr_context"] = {
                "mode": parse_mode,
                "league_code": screenshot_code,
            }
            st.session_state["ocr_saved"] = False
        except FootballPredictorError as exc:
            st.error(str(exc))

    if "ocr_review_frame" in st.session_state:
        for warning in st.session_state.get("ocr_warnings", []):
            st.warning(warning)
        context = st.session_state.get("ocr_context", {})
        context_changed = (
            context.get("mode") != parse_mode or context.get("league_code") != screenshot_code
        )
        if context_changed:
            st.error("The mode or league changed after extraction. Read the screenshots again before confirming.")
        if st.session_state.get("ocr_saved", False):
            st.success("This reviewed screenshot batch has already been saved. Clear it to start another import.")
        review_frame = st.session_state["ocr_review_frame"]
        st.subheader("Verify extracted rows")
        st.caption(
            "Edit any incorrect date, kickoff time, team, or score. Confidence and review notes describe the original OCR extraction. "
            "Untick Include for rows you do not want to save."
        )
        edited_frame = st.data_editor(
            review_frame,
            width="stretch",
            hide_index=True,
            num_rows="dynamic",
            disabled=["confidence", "review", "source_image", "source_text"],
            column_config={
                "include": st.column_config.CheckboxColumn("Include", default=True),
                "match_date": st.column_config.TextColumn("Match date", help="Use YYYY-MM-DD"),
                "kickoff_time": st.column_config.TextColumn("Kickoff", help="Use HH:MM"),
                "home_team": st.column_config.TextColumn("Home team", required=True),
                "away_team": st.column_config.TextColumn("Away team", required=True),
                "home_goals": st.column_config.NumberColumn(
                    "Home goals", min_value=0, max_value=30, step=1
                ),
                "away_goals": st.column_config.NumberColumn(
                    "Away goals", min_value=0, max_value=30, step=1
                ),
                "confidence": st.column_config.ProgressColumn(
                    "OCR confidence", min_value=0, max_value=100
                ),
                "review": st.column_config.TextColumn("Review notes", width="large"),
                "source_image": st.column_config.TextColumn("Image"),
                "source_text": st.column_config.TextColumn("Original OCR text", width="large"),
            },
            key="ocr_review_editor",
        )
        st.session_state["ocr_review_frame"] = edited_frame
        confirm_col, clear_col = st.columns(2)
        if confirm_col.button(
            "Confirm and save selected rows",
            type="primary",
            disabled=context_changed or edited_frame.empty or st.session_state.get("ocr_saved", False),
        ):
            try:
                if parse_mode == "results":
                    confirmed_results = match_inputs_from_review(
                        edited_frame, screenshot_code, screenshot_season
                    )
                    with session_scope() as session:
                        add_completed_matches(session, confirmed_results)
                    st.success(
                        f"Saved {len(confirmed_results)} verified results and recalculated league statistics."
                    )
                    for message in auto_retrain({screenshot_code}):
                        st.caption(message)
                else:
                    confirmed_fixtures = fixture_inputs_from_review(
                        edited_frame, screenshot_code, screenshot_season
                    )
                    with session_scope() as session:
                        for fixture_input in confirmed_fixtures:
                            add_upcoming_fixture(session, fixture_input)
                    st.success(f"Saved {len(confirmed_fixtures)} verified upcoming fixtures.")
                st.session_state["ocr_saved"] = True
            except (FootballPredictorError, ValueError) as exc:
                st.error(str(exc))
        if clear_col.button("Clear screenshot review"):
            for key in (
                "ocr_review_frame",
                "ocr_raw_outputs",
                "ocr_warnings",
                "ocr_context",
                "ocr_review_editor",
                "ocr_saved",
            ):
                st.session_state.pop(key, None)
            st.rerun()

        with st.expander("Raw OCR text", expanded=False):
            for output in st.session_state.get("ocr_raw_outputs", []):
                st.markdown(
                    f"**{output['image']} — mean OCR confidence {output['confidence']:.0f}%**"
                )
                st.code(output["text"] or "No text recognized")


elif page == "Upcoming Fixtures":
    st.title("Upcoming fixtures")
    entry_tab, batch_tab = st.tabs(["Add fixture", "Batch prediction"])
    with entry_tab:
        league_label = st.selectbox("League", list(LEAGUES), key="fixture_league")
        code = LEAGUES[league_label]
        known_teams = team_names(code)
        if len(known_teams) < 2:
            st.warning("Add historical results for at least two teams first.")
        with st.form("fixture_form", clear_on_submit=True):
            season = st.text_input("Season", placeholder="2025/26", key="fixture_season")
            kickoff_left, kickoff_right = st.columns(2)
            fixture_date = kickoff_left.date_input(
                "Scheduled date", min_value=local_now().date()
            )
            fixture_time = kickoff_right.time_input(
                "Kickoff time",
                value=local_now().time().replace(second=0, microsecond=0),
            )
            c1, c2 = st.columns(2)
            home = c1.selectbox("Home team", known_teams, disabled=not known_teams)
            away = c2.selectbox("Away team", known_teams, disabled=not known_teams)
            add_fixture = st.form_submit_button("Add fixture", disabled=len(known_teams) < 2)
        if add_fixture:
            try:
                with session_scope() as session:
                    add_upcoming_fixture(
                        session,
                        FixtureInput(
                            code,
                            season or None,
                            fixture_date,
                            home,
                            away,
                            kickoff_time=fixture_time,
                        ),
                    )
                st.success("Fixture added.")
            except FootballPredictorError as exc:
                st.error(str(exc))
        upload = st.file_uploader("Or upload fixture CSV / Excel", type=["csv", "xlsx"], key="fixture_upload")
        if upload and st.button("Import fixtures"):
            try:
                items = parse_fixture_upload(upload, upload.name)
                with session_scope() as session:
                    for item in items:
                        add_upcoming_fixture(session, item)
                st.success(f"Added {len(items)} fixture rows.")
            except FootballPredictorError as exc:
                st.error(str(exc))
    with batch_tab:
        with session_scope() as session:
            fixtures = session.scalars(
                select(UpcomingFixture)
                .where(UpcomingFixture.status == "scheduled")
                .order_by(UpcomingFixture.match_date, UpcomingFixture.id)
            ).all()
            fixture_rows = [
                {
                    "id": item.id,
                    "kickoff": item.match_date,
                    "date": kickoff_parts(item.match_date)[0],
                    "kickoff_time": kickoff_parts(item.match_date)[1],
                    "league": item.league.name,
                    "home": item.home_team.name,
                    "away": item.away_team.name,
                }
                for item in fixtures
            ]
        if fixture_rows:
            st.dataframe(
                pd.DataFrame(fixture_rows).drop(columns=["date", "kickoff_time"]),
                width="stretch",
                hide_index=True,
            )
            fixture_labels = {
                f"#{item['id']} · {display_kickoff(item['kickoff'])} · "
                f"{item['home']} vs {item['away']}": item
                for item in fixture_rows
            }
            selected_fixture_label = st.selectbox(
                "Correct a scheduled kickoff",
                list(fixture_labels),
                key="fixture_schedule_edit",
            )
            selected_fixture = fixture_labels[selected_fixture_label]
            with st.form("edit_fixture_kickoff"):
                edit_left, edit_right = st.columns(2)
                corrected_fixture_date = edit_left.date_input(
                    "Correct date", value=selected_fixture["date"]
                )
                corrected_fixture_time = edit_right.time_input(
                    "Correct time", value=selected_fixture["kickoff_time"]
                )
                save_fixture_time = st.form_submit_button("Save kickoff correction")
            if save_fixture_time:
                try:
                    with session_scope() as session:
                        update_upcoming_fixture_kickoff(
                            session,
                            selected_fixture["id"],
                            match_date=corrected_fixture_date,
                            kickoff_time=corrected_fixture_time,
                        )
                    st.success("Fixture kickoff corrected.")
                    st.rerun()
                except FootballPredictorError as exc:
                    st.error(str(exc))
            if st.button("Generate predictions for all scheduled fixtures", type="primary"):
                successes, failures = [], []
                for item in fixture_rows:
                    try:
                        with session_scope() as session:
                            prediction = generate_bulk_predictions(session, [item["id"]])[0]
                            successes.append(prediction.id)
                    except FootballPredictorError as exc:
                        failures.append(f"#{item['id']}: {exc}")
                if successes:
                    st.success(f"Stored {len(successes)} new immutable predictions.")
                for failure in failures:
                    st.warning(failure)
        else:
            st.info("No scheduled fixtures are waiting for prediction.")


elif page == "Predictions":
    st.title("Predictions")
    with session_scope() as session:
        items = session.scalars(select(Prediction).order_by(Prediction.created_at.desc())).all()
        for prediction in items:
            fixture = prediction.fixture
            with st.expander(
                f"{display_kickoff(fixture.match_date)} · {fixture.home_team.name} vs "
                f"{fixture.away_team.name} · {prediction.most_likely_score}",
                expanded=False,
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Home win", f"{prediction.home_win_probability:.1%}")
                c2.metric("Draw", f"{prediction.draw_probability:.1%}")
                c3.metric("Away win", f"{prediction.away_win_probability:.1%}")
                g1, g2, g3 = st.columns(3)
                g1.metric("Expected goals", f"{prediction.expected_home_goals:.2f} – {prediction.expected_away_goals:.2f}")
                g2.metric("Most likely score", prediction.most_likely_score)
                g3.metric("Data reliability", prediction.reliability.title())
                probability_rows = pd.DataFrame(
                    {
                        "Market": ["Over 1.5", "Over 2.5", "Under 2.5", "Over 3.5", "BTTS Yes", "BTTS No"],
                        "Probability": [
                            prediction.over_15_probability,
                            prediction.over_25_probability,
                            prediction.under_25_probability,
                            prediction.over_35_probability,
                            prediction.btts_yes_probability,
                            prediction.btts_no_probability,
                        ],
                    }
                )
                st.dataframe(probability_rows.style.format({"Probability": "{:.1%}"}), hide_index=True)
                st.write("Top scores", pd.DataFrame(prediction.top_scores_json).style.format({"probability": "{:.1%}"}))
                for factor in prediction.explanation_json.get("factors", []):
                    st.write(f"• {factor}")
                for warning in prediction.explanation_json.get("data_warnings", []):
                    st.warning(warning)
                st.caption(f"Model {prediction.model_version.version} · generated {prediction.created_at:%d %b %Y %H:%M}")
    if not items:
        st.info("No predictions yet. Add fixtures and run batch prediction.")


elif page == "Model Performance":
    st.title("Model performance")
    with session_scope() as session:
        rows = performance_rows(session)
        leagues = accuracy_by_league(session)
    if not rows:
        st.info("Performance appears after predicted fixtures are matched to completed results.")
    else:
        frame = pd.DataFrame(rows)
        frame["rolling_accuracy"] = frame["outcome_correct"].astype(float).rolling(20, min_periods=1).mean()
        trend_window = min(20, len(frame) // 2)
        if trend_window >= 5:
            previous_accuracy = frame["outcome_correct"].iloc[-2 * trend_window : -trend_window].mean()
            recent_accuracy = frame["outcome_correct"].iloc[-trend_window:].mean()
            delta = recent_accuracy - previous_accuracy
            direction = "Improving" if delta > 0.03 else "Declining" if delta < -0.03 else "Stable"
            st.metric(
                f"Recent performance trend ({trend_window}-prediction windows)",
                direction,
                f"{delta:+.1%}",
            )
        else:
            st.caption("At least 10 evaluated predictions are needed to label the performance trend.")
        st.plotly_chart(
            px.line(frame, x="date", y="rolling_accuracy", color="league", title="Rolling 20-prediction 1X2 accuracy"),
            width="stretch",
        )
        calibration_parts = []
        for outcome, probability_column in (
            ("H", "home_probability"),
            ("D", "draw_probability"),
            ("A", "away_probability"),
        ):
            part = pd.DataFrame(
                {
                    "probability": frame[probability_column],
                    "observed": (frame["actual"] == outcome).astype(float),
                }
            )
            calibration_parts.append(part)
        calibration = pd.concat(calibration_parts, ignore_index=True)
        calibration["bin"] = pd.cut(
            calibration["probability"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True
        )
        calibration_plot = (
            calibration.groupby("bin", observed=False)
            .agg(predicted_probability=("probability", "mean"), observed_frequency=("observed", "mean"), samples=("observed", "size"))
            .dropna()
            .reset_index()
        )
        if not calibration_plot.empty:
            figure = px.line(
                calibration_plot,
                x="predicted_probability",
                y="observed_frequency",
                markers=True,
                hover_data=["samples"],
                title="Probability calibration (all 1X2 events)",
                range_x=[0, 1],
                range_y=[0, 1],
            )
            figure.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "gray"})
            st.plotly_chart(figure, width="stretch")
        st.dataframe(pd.DataFrame(leagues), width="stretch", hide_index=True)
        st.subheader("Prediction audit trail")
        st.dataframe(frame, width="stretch", hide_index=True)


elif page == "Model Management":
    st.title("Model management")
    selected_label = st.selectbox("League", list(LEAGUES), key="training_league")
    if st.button("Retrain model", type="primary"):
        try:
            st.success(retrain_code(LEAGUES[selected_label], "manual"))
        except FootballPredictorError as exc:
            st.error(str(exc))
    with session_scope() as session:
        league = get_league(session, LEAGUES[selected_label])
        versions = session.scalars(
            select(ModelVersion)
            .where(ModelVersion.league_id == league.id)
            .order_by(ModelVersion.created_at.desc())
        ).all()
        latest_metrics = []
        if versions:
            metrics = session.scalars(
                select(ModelMetric)
                .where(ModelMetric.model_version_id == versions[0].id)
                .order_by(ModelMetric.split, ModelMetric.model_name, ModelMetric.metric_name)
            ).all()
            latest_metrics = [
                {
                    "split": item.split,
                    "model": item.model_name,
                    "metric": item.metric_name,
                    "value": item.metric_value,
                }
                for item in metrics
            ]
        runs = session.scalars(
            select(TrainingRun)
            .where(TrainingRun.league_id == league.id)
            .order_by(TrainingRun.started_at.desc())
            .limit(30)
        ).all()
        version_rows = [
            {
                "version": item.version,
                "active": item.is_active,
                "promoted": item.promoted,
                "algorithm": item.algorithm,
                "matches": item.training_matches,
                "cutoff": item.training_cutoff,
                "test_log_loss": item.test_log_loss,
                "created": item.created_at,
            }
            for item in versions
        ]
        run_rows = [
            {
                "started": item.started_at,
                "status": item.status,
                "trigger": item.trigger,
                "promoted": item.promoted,
                "message": item.message,
            }
            for item in runs
        ]
    st.subheader("Model versions")
    st.dataframe(pd.DataFrame(version_rows), width="stretch", hide_index=True)
    st.subheader("Training history")
    st.dataframe(pd.DataFrame(run_rows), width="stretch", hide_index=True)
    if latest_metrics:
        st.subheader("Latest candidate evaluation")
        st.dataframe(pd.DataFrame(latest_metrics), width="stretch", hide_index=True)
    st.caption(
        "Candidates are evaluated on chronological holdouts and walk-forward folds. A retrain can be retained for audit without replacing a better active version."
    )

