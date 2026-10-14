from pathlib import Path

from football_predictor import config


def test_local_streamlit_secret_is_used_when_environment_is_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("VFOOTBALL_DATABASE_URL", raising=False)
    monkeypatch.setattr(config, "PROJECT_ROOT", Path(tmp_path))
    secret_dir = tmp_path / ".streamlit"
    secret_dir.mkdir()
    expected = "postgresql://user:password@example.test:5432/postgres"
    (secret_dir / "secrets.toml").write_text(
        f'VFOOTBALL_DATABASE_URL = "{expected}"\n', encoding="utf-8"
    )

    assert config.get_settings().database_url == expected
