from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Index,
    String,
    Text,
    create_engine,
    delete,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id = Column(String(36), primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    symbol = Column(String(32), nullable=False, index=True)
    document_date = Column(String(10), nullable=False, index=True)
    horizon = Column(String(8), nullable=False, index=True)
    forecast_mode = Column(String(16), nullable=False)
    stance = Column(String(16), nullable=False, index=True)
    sentiment_score = Column(Float, nullable=True)
    predicted_close = Column(Float, nullable=True)
    current_close = Column(Float, nullable=True)
    predicted_volatility = Column(Float, nullable=True)
    payload = Column(JSON, nullable=False)
    text_excerpt = Column(Text, nullable=True)

    __table_args__ = (Index("ix_analysis_runs_created_symbol", "created_at", "symbol"),)

    def to_summary(self) -> dict[str, Any]:
        regime = _extract_regime_summary(self.payload)
        return {
            "id": self.id,
            "created_at": _isoformat(self.created_at),
            "symbol": self.symbol,
            "document_date": self.document_date,
            "horizon": self.horizon,
            "forecast_mode": self.forecast_mode,
            "stance": self.stance,
            "sentiment_score": self.sentiment_score,
            "stance_score": _extract_stance_score(self.payload),
            "predicted_close": self.predicted_close,
            "current_close": self.current_close,
            "predicted_volatility": self.predicted_volatility,
            "text_excerpt": self.text_excerpt,
            "argmax_regime": regime["argmax"],
            "argmax_probability": regime["probability"],
            "regime_set_size": regime["set_size"],
        }

    def to_detail(self) -> dict[str, Any]:
        summary = self.to_summary()
        summary["payload"] = self.payload
        return summary


def _extract_stance_score(payload: Any) -> float | None:
    """``P(hawkish) - P(dovish)`` from a persisted /analyze response.

    Inlined here (rather than imported from ``app.services.stance_context``)
    because that module imports :class:`AnalysisRun` from this one. Returns
    ``None`` when the payload lacks the multi-axis block or either class
    probability so pre-#338 / regression-mode rows degrade to ``null``
    rather than fabricating a zero.
    """

    if not isinstance(payload, dict):
        return None
    multi_axis = payload.get("multi_axis")
    if not isinstance(multi_axis, dict):
        return None
    stance = multi_axis.get("stance")
    if not isinstance(stance, dict):
        return None
    distribution = stance.get("distribution")
    if not isinstance(distribution, dict):
        return None
    hawk = distribution.get("hawkish")
    dove = distribution.get("dovish")
    if not isinstance(hawk, int | float) or not isinstance(dove, int | float):
        return None
    return float(hawk) - float(dove)


def _extract_regime_summary(payload: Any) -> dict[str, Any]:
    """Pull regime argmax / probability / set size off a persisted run.

    History rows stash the full /analyze response under ``payload``; the
    regime card lives at ``payload.regime_classification`` and is
    optional (None when the active checkpoint is regression-mode). Each
    field degrades to None on missing / malformed payloads rather than
    raising so history-list rendering is never blocked by a stale row.
    """

    default = {"argmax": None, "probability": None, "set_size": None}
    if not isinstance(payload, dict):
        return default
    regime = payload.get("regime_classification")
    if not isinstance(regime, dict):
        return default
    argmax = regime.get("argmax_class") if isinstance(regime.get("argmax_class"), str) else None
    distribution = regime.get("distribution") or {}
    probability: float | None = None
    if argmax and isinstance(distribution, dict):
        raw = distribution.get(argmax)
        if isinstance(raw, int | float):
            probability = float(raw)
    set_size = regime.get("set_size")
    if not isinstance(set_size, int):
        set_size = None
    return {"argmax": argmax, "probability": probability, "set_size": set_size}


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _resolve_database_url(database_url: str | None) -> tuple[str, Path | None]:
    if database_url:
        return database_url, None
    data_dir = Path(settings.data_dir)
    db_dir = data_dir / "db"
    db_path = db_dir / "fed_pulse.db"
    return f"sqlite:///{db_path}", db_path


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine(database_url: str | None = None) -> Engine:
    global _engine, _SessionLocal
    if _engine is not None and database_url is None:
        return _engine
    url, db_path = _resolve_database_url(database_url)
    if db_path is not None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    if database_url is None:
        _engine = engine
        _SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    return engine


def get_session() -> Iterator[Session]:
    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Manual session helper for code paths outside FastAPI's ``Depends``.

    Use this when persisting from a background thread or a hook (e.g.
    ``_record_history``) so the ``finally`` cleanup always runs even when the
    body raises before the second ``next()`` of the generator.
    """

    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()


def reset_for_testing(database_url: str) -> Engine:
    global _engine, _SessionLocal
    _engine = create_engine(database_url, future=True)
    Base.metadata.drop_all(_engine)
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, autoflush=False)
    return _engine


def persist_analysis_run(
    session: Session,
    *,
    payload: dict[str, Any],
    request: dict[str, Any],
    response: dict[str, Any],
) -> AnalysisRun:
    stance = (response.get("sentiment", {}) or {}).get("label") or "unknown"
    market = response.get("market", {}) or {}
    prediction = response.get("prediction", {}) or {}
    sentiment = response.get("sentiment", {}) or {}
    excerpt = request.get("text", "")
    if isinstance(excerpt, str) and len(excerpt) > 280:
        excerpt = excerpt[:280] + "…"
    record = AnalysisRun(
        id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        symbol=market.get("symbol") or request.get("symbol", "unknown"),
        document_date=request.get("date", ""),
        horizon=request.get("horizon", ""),
        forecast_mode=request.get("forecast_mode", ""),
        stance=str(stance).lower(),
        sentiment_score=_coerce_float(sentiment.get("score")),
        predicted_close=_coerce_float(prediction.get("close")),
        current_close=_coerce_float(market.get("close")),
        predicted_volatility=_coerce_float(prediction.get("volatility")),
        payload=payload,
        text_excerpt=excerpt or None,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def list_runs(
    session: Session,
    *,
    limit: int,
    offset: int,
    symbol: str | None = None,
    horizon: str | None = None,
    stance: str | None = None,
    document_date: str | None = None,
) -> tuple[list[AnalysisRun], int]:
    stmt = select(AnalysisRun)
    if symbol:
        stmt = stmt.where(AnalysisRun.symbol == symbol)
    if horizon:
        stmt = stmt.where(AnalysisRun.horizon == horizon)
    if stance:
        stmt = stmt.where(AnalysisRun.stance == stance.lower())
    if document_date:
        stmt = stmt.where(AnalysisRun.document_date == document_date)

    count_stmt = stmt.with_only_columns(AnalysisRun.id).order_by(None)
    total = session.execute(count_stmt).scalars().all()
    total_count = len(total)

    stmt = stmt.order_by(AnalysisRun.created_at.desc()).limit(limit).offset(offset)
    rows = list(session.execute(stmt).scalars().all())
    return rows, total_count


def get_run(session: Session, run_id: str) -> AnalysisRun | None:
    return session.get(AnalysisRun, run_id)


def delete_run(session: Session, run_id: str) -> bool:
    result = session.execute(delete(AnalysisRun).where(AnalysisRun.id == run_id))
    session.commit()
    return result.rowcount > 0


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None  # rejects NaN


def serialise_payload(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return json.loads(json.dumps(model, default=str))
