from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import joblib
from sqlalchemy import select
from sqlalchemy.orm import Session

from .ml import ModelBundle
from .models import ModelArtifact, ModelVersion


def save_model_bundle(
    session: Session,
    version: ModelVersion,
    bundle: ModelBundle,
    local_path: Path,
) -> None:
    buffer = BytesIO()
    joblib.dump(bundle, buffer)
    payload = buffer.getvalue()
    local_path.write_bytes(payload)
    existing = session.scalar(
        select(ModelArtifact).where(ModelArtifact.model_version_id == version.id)
    )
    if existing is None:
        session.add(
            ModelArtifact(
                model_version_id=version.id,
                payload=payload,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    else:
        existing.payload = payload
        existing.sha256 = hashlib.sha256(payload).hexdigest()


def load_model_bundle(session: Session, version: ModelVersion) -> ModelBundle:
    artifact = session.scalar(
        select(ModelArtifact).where(ModelArtifact.model_version_id == version.id)
    )
    if artifact is not None:
        if hashlib.sha256(artifact.payload).hexdigest() != artifact.sha256:
            raise ValueError("The stored model artifact failed its integrity check.")
        return joblib.load(BytesIO(artifact.payload))
    path = Path(version.artifact_path)
    if path.exists():
        return joblib.load(path)
    raise FileNotFoundError("The active model artifact is missing. Retrain this league.")


def model_bundle_exists(session: Session, version: ModelVersion) -> bool:
    stored = session.scalar(
        select(ModelArtifact.id).where(ModelArtifact.model_version_id == version.id)
    )
    return stored is not None or Path(version.artifact_path).exists()
