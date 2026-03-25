from __future__ import annotations

import copy
import csv
import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

SEQUENCE_LENGTH = 5
FEATURE_SIZE = 5  # [sentiment_score, market_close, market_volatility, close_change_pct, volatility_change]
FORECAST_CONFIDENCE_LEVEL = 0.8
CONFIDENCE_Z_SCORE = 1.2816  # Approximate central 80% interval
DEFAULT_CLOSE_SCALE = 10000.0
DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EARLY_STOPPING_PATIENCE = 8
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.15
DEFAULT_HEAD_HIDDEN_SIZE = 32

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path("/data") if Path("/data").exists() else BACKEND_ROOT.parent / "data"
MODELS_DIR = BACKEND_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "forecaster_best.pt"

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


@dataclass(frozen=True)
class ModelConfig:
    input_size: int = FEATURE_SIZE
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    num_layers: int = DEFAULT_NUM_LAYERS
    dropout: float = DEFAULT_DROPOUT
    head_hidden_size: int = DEFAULT_HEAD_HIDDEN_SIZE

    @classmethod
    def from_model(cls, model: "ForecasterModel") -> "ModelConfig":
        return cls(
            input_size=model.input_size,
            hidden_size=model.hidden_size,
            num_layers=model.num_layers,
            dropout=model.dropout,
            head_hidden_size=model.head_hidden_size,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationMetrics:
    loss: float
    close_rmse: float
    volatility_rmse: float
    combined_rmse: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingRunSummary:
    model_config: ModelConfig
    device: str
    epochs_requested: int
    epochs_completed: int
    batch_size: int
    learning_rate: float
    validation_split: float
    early_stopping_patience: int
    sequence_groups: int
    total_windows: int
    train_windows: int
    validation_windows: int
    checkpoint_path: str | None
    checkpoint_saved: bool
    best_epoch: int | None = None
    metrics: EvaluationMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingResult:
    model: "ForecasterModel"
    summary: TrainingRunSummary


class ForecasterModel(nn.Module):
    def __init__(
        self,
        input_size: int = FEATURE_SIZE,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        head_hidden_size: int = DEFAULT_HEAD_HIDDEN_SIZE,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.head_hidden_size = head_hidden_size
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, head_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]
        raw = self.head(last_step)
        close = raw[:, 0:1]
        # Volatility must stay non-negative, while close remains unconstrained.
        volatility = F.softplus(raw[:, 1:2])
        return torch.cat((close, volatility), dim=1)


@dataclass
class FeatureVector:
    date: str
    sentiment_score: float
    market_close: float
    market_volatility: float
    close_change_pct: float = 0.0
    volatility_change: float = 0.0

    @classmethod
    def from_market_state(
        cls,
        *,
        date: str,
        sentiment_score: float,
        market_close: float,
        market_volatility: float,
        previous_close: float | None = None,
        previous_volatility: float | None = None,
    ) -> "FeatureVector":
        close_change_pct = 0.0
        if previous_close is not None and abs(previous_close) > 1e-12:
            close_change_pct = (float(market_close) - float(previous_close)) / float(previous_close)

        volatility_change = 0.0
        if previous_volatility is not None:
            volatility_change = float(market_volatility) - float(previous_volatility)

        return cls(
            date=date,
            sentiment_score=float(sentiment_score),
            market_close=float(market_close),
            market_volatility=float(market_volatility),
            close_change_pct=float(close_change_pct),
            volatility_change=float(volatility_change),
        )

    def as_list(self, close_scale: float = DEFAULT_CLOSE_SCALE) -> list[float]:
        return [
            float(self.sentiment_score),
            float(self.market_close) / close_scale,
            float(self.market_volatility),
            max(min(float(self.close_change_pct), 1.0), -1.0),
            max(min(float(self.volatility_change), 1.0), -1.0),
        ]


@dataclass
class TrainingDataSourceSummary:
    path: Path
    format: str
    record_groups: int
    records: int
    vectors: int
    usable_sequences: int
    status: str
    message: str


_model: ForecasterModel | None = None
_model_artifact_metadata: dict[str, Any] | None = None
_model_lock = threading.Lock()


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _coerce_model_config(model_config: ModelConfig | dict[str, Any] | None = None) -> ModelConfig:
    if isinstance(model_config, ModelConfig):
        return model_config
    if isinstance(model_config, dict):
        return ModelConfig(
            input_size=int(model_config.get("input_size", FEATURE_SIZE)),
            hidden_size=int(model_config.get("hidden_size", DEFAULT_HIDDEN_SIZE)),
            num_layers=int(model_config.get("num_layers", DEFAULT_NUM_LAYERS)),
            dropout=float(model_config.get("dropout", DEFAULT_DROPOUT)),
            head_hidden_size=int(model_config.get("head_hidden_size", DEFAULT_HEAD_HIDDEN_SIZE)),
        )
    return ModelConfig()


def _read_checkpoint_payload(checkpoint_path: Path, device: torch.device) -> dict[str, Any] | None:
    if not checkpoint_path.exists():
        return None

    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload
    return {"model_state_dict": payload}


def _metrics_from_payload(payload: dict[str, Any] | None) -> EvaluationMetrics | None:
    if not isinstance(payload, dict):
        return None
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        try:
            return EvaluationMetrics(
                loss=float(metrics["loss"]),
                close_rmse=float(metrics["close_rmse"]),
                volatility_rmse=float(metrics["volatility_rmse"]),
                combined_rmse=float(metrics["combined_rmse"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _checkpoint_metadata(
    payload: dict[str, Any] | None,
    checkpoint_path: Path,
    *,
    runtime_mode: str = "fast",
    model: ForecasterModel | None = None,
    adaptation_summary: TrainingRunSummary | None = None,
) -> dict[str, Any]:
    model_config = (
        ModelConfig.from_model(model)
        if model is not None
        else _coerce_model_config(payload.get("model_config") if isinstance(payload, dict) else None)
    )
    payload_metrics = _metrics_from_payload(payload)
    metrics = adaptation_summary.metrics if adaptation_summary and adaptation_summary.metrics else payload_metrics
    metadata: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.exists(),
        "checkpoint_loaded": bool(payload),
        "runtime_mode": runtime_mode,
        "hidden_size": model_config.hidden_size,
        "num_layers": model_config.num_layers,
        "dropout": model_config.dropout,
        "head_hidden_size": model_config.head_hidden_size,
        "close_scale": (
            float(payload.get("close_scale", DEFAULT_CLOSE_SCALE))
            if isinstance(payload, dict)
            else float(DEFAULT_CLOSE_SCALE)
        ),
        "sequence_length": (
            int(payload.get("sequence_length", SEQUENCE_LENGTH))
            if isinstance(payload, dict)
            else int(SEQUENCE_LENGTH)
        ),
        "best_loss": payload_metrics.loss if payload_metrics else payload.get("best_loss") if isinstance(payload, dict) else None,
        "combined_rmse": payload_metrics.combined_rmse if payload_metrics else None,
        "adaptation_epochs_completed": adaptation_summary.epochs_completed if adaptation_summary else None,
        "adaptation_best_epoch": adaptation_summary.best_epoch if adaptation_summary else None,
        "adaptation_loss": metrics.loss if adaptation_summary and metrics else None,
        "adaptation_combined_rmse": metrics.combined_rmse if adaptation_summary and metrics else None,
    }
    return metadata


def _build_model(
    model_config: ModelConfig | dict[str, Any] | None = None,
    *,
    device: torch.device | None = None,
) -> ForecasterModel:
    resolved_config = _coerce_model_config(model_config)
    model = ForecasterModel(**resolved_config.to_dict())
    if device is not None:
        model = model.to(device)
    return model


def _checkpoint_payload(model: ForecasterModel, summary: TrainingRunSummary) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "best_loss": float(summary.metrics.loss) if summary.metrics else None,
        "metrics": summary.metrics.to_dict() if summary.metrics else None,
        "model_config": ModelConfig.from_model(model).to_dict(),
        "training_summary": summary.to_dict(),
        "input_size": FEATURE_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "close_scale": DEFAULT_CLOSE_SCALE,
    }


def _save_model_checkpoint(model: ForecasterModel, checkpoint_path: Path, summary: TrainingRunSummary) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(model, summary), checkpoint_path)


def _load_model_checkpoint(
    model: ForecasterModel,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any] | None:
    payload = _read_checkpoint_payload(checkpoint_path, device)
    if payload is None:
        return None
    state_dict = payload["model_state_dict"]
    model.load_state_dict(state_dict)
    return payload


def _get_model() -> ForecasterModel:
    global _model, _model_artifact_metadata
    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            device = _resolve_device()
            payload = _read_checkpoint_payload(BEST_MODEL_PATH, device)
            model = _build_model(
                payload.get("model_config") if isinstance(payload, dict) else None,
                device=device,
            )
            if payload is not None:
                model.load_state_dict(payload["model_state_dict"])
            model.eval()
            _model = model
            _model_artifact_metadata = _checkpoint_metadata(payload, BEST_MODEL_PATH, model=model)
    return _model


def _parse_horizon_steps(horizon: str) -> int:
    if horizon.endswith("d") and horizon[:-1].isdigit():
        return max(1, int(horizon[:-1]))
    return 3


def parse_horizon_steps(horizon: str) -> int:
    return _parse_horizon_steps(horizon)


def _extract_required_float(record: dict[str, Any], keys: Sequence[str]) -> float:
    for key in keys:
        if key in record and record[key] not in {None, ""}:
            return float(record[key])
    raise ValueError(f"Missing required numeric field from keys: {', '.join(keys)}")


def build_feature_vectors(
    records: Sequence[dict[str, Any]],
    sentiment_score: float | None = None,
) -> list[FeatureVector]:
    vectors: list[FeatureVector] = []
    previous_close: float | None = None
    previous_volatility: float | None = None

    sorted_records = sorted(records, key=lambda item: str(item.get("date", item.get("timestamp", ""))))
    for record in sorted_records:
        date_value = str(record.get("date", record.get("timestamp", "")))
        if not date_value:
            continue

        close_value = _extract_required_float(record, ("close", "market_close"))
        volatility_value = _extract_required_float(
            record,
            ("volatility_5d", "market_volatility", "volatility"),
        )
        row_sentiment = float(record.get("sentiment_score", sentiment_score if sentiment_score is not None else 0.0))
        vectors.append(
            FeatureVector.from_market_state(
                date=date_value,
                sentiment_score=row_sentiment,
                market_close=close_value,
                market_volatility=volatility_value,
                previous_close=previous_close,
                previous_volatility=previous_volatility,
            )
        )
        previous_close = close_value
        previous_volatility = volatility_value

    return vectors


def build_last5_sequence(vectors: Iterable[FeatureVector], length: int = SEQUENCE_LENGTH) -> list[FeatureVector]:
    items = list(vectors)
    if not items:
        items = [FeatureVector(date="", sentiment_score=0.0, market_close=0.0, market_volatility=0.0)]

    if len(items) >= length:
        return items[-length:]

    pad = [items[0] for _ in range(length - len(items))]
    return pad + items


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "rows", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _load_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _is_record_mapping_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _extract_record_groups(payload: Any) -> list[list[dict[str, Any]]]:
    if _is_record_mapping_list(payload):
        if payload and any(any(key in item for key in ("records", "rows", "data", "items")) for item in payload):
            nested_groups: list[list[dict[str, Any]]] = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                nested_groups.extend(_extract_record_groups(item))
            return nested_groups or [payload]
        return [payload]

    if isinstance(payload, dict):
        for key in ("sequences", "series", "groups"):
            nested = payload.get(key)
            if isinstance(nested, list):
                groups: list[list[dict[str, Any]]] = []
                for entry in nested:
                    groups.extend(_extract_record_groups(entry))
                if groups:
                    return groups

        for key in ("records", "rows", "data", "items"):
            nested = payload.get(key)
            if _is_record_mapping_list(nested):
                return [nested]

    return []


def _load_record_groups(path: Path) -> tuple[list[list[dict[str, Any]]], str]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _extract_record_groups(payload), "json"
    if path.suffix == ".jsonl":
        return [_load_jsonl_records(path)], "jsonl"
    if path.suffix == ".csv":
        return [_load_csv_records(path)], "csv"
    return [], path.suffix.lstrip(".") or "unknown"


def inspect_training_data_sources(
    data_dir: str | Path | None = None,
) -> tuple[list[list[FeatureVector]], list[TrainingDataSourceSummary]]:
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    if not root.exists():
        return [], []

    sequences: list[list[FeatureVector]] = []
    summaries: list[TrainingDataSourceSummary] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue

        try:
            groups, format_name = _load_record_groups(path)
            if not groups:
                summaries.append(
                    TrainingDataSourceSummary(
                        path=path,
                        format=format_name,
                        record_groups=0,
                        records=0,
                        vectors=0,
                        usable_sequences=0,
                        status="ignored",
                        message="No trainable market-record groups detected.",
                    )
                )
                continue

            record_count = sum(len(group) for group in groups)
            vectors_for_path = [build_feature_vectors(group) for group in groups]
            usable = [vector_group for vector_group in vectors_for_path if len(vector_group) >= SEQUENCE_LENGTH + 1]
            sequences.extend(usable)
            summaries.append(
                TrainingDataSourceSummary(
                    path=path,
                    format=format_name,
                    record_groups=len(groups),
                    records=record_count,
                    vectors=sum(len(group) for group in vectors_for_path),
                    usable_sequences=len(usable),
                    status="usable" if usable else "insufficient",
                    message=(
                        "Usable training sequences detected."
                        if usable
                        else f"Need at least {SEQUENCE_LENGTH + 1} usable rows per sequence."
                    ),
                )
            )
        except Exception as exc:
            summaries.append(
                TrainingDataSourceSummary(
                    path=path,
                    format=path.suffix.lstrip(".") or "unknown",
                    record_groups=0,
                    records=0,
                    vectors=0,
                    usable_sequences=0,
                    status="error",
                    message=str(exc),
                )
            )
            continue

    return sequences, summaries


def load_training_sequences_from_data(data_dir: str | Path | None = None) -> list[list[FeatureVector]]:
    sequences, _ = inspect_training_data_sources(data_dir)
    return sequences


def _build_training_tensors(
    sequence_groups: Sequence[Sequence[FeatureVector]],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    sequences: list[list[list[float]]] = []
    targets: list[list[float]] = []

    for sequence_group in sequence_groups:
        if len(sequence_group) < SEQUENCE_LENGTH + 1:
            continue
        for idx in range(SEQUENCE_LENGTH, len(sequence_group)):
            window = sequence_group[idx - SEQUENCE_LENGTH : idx]
            target = sequence_group[idx]
            sequences.append([item.as_list() for item in window])
            targets.append(
                [
                    target.market_close / DEFAULT_CLOSE_SCALE,
                    max(target.market_volatility, 0.0),
                ]
            )

    if not sequences:
        return None, None

    x = torch.tensor(sequences, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    return x, y


def _split_train_validation(
    x: torch.Tensor,
    y: torch.Tensor,
    validation_split: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(x) < 2:
        return x, y, x, y

    val_size = max(1, int(len(x) * validation_split))
    train_size = max(1, len(x) - val_size)
    if train_size >= len(x):
        train_size = len(x) - 1
        val_size = 1

    return x[:train_size], y[:train_size], x[train_size:], y[train_size:]


def _evaluate_model(
    model: ForecasterModel,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> EvaluationMetrics:
    model.eval()
    total_loss = 0.0
    total_items = 0
    close_squared_error = 0.0
    volatility_squared_error = 0.0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=device.type == "cuda")
            batch_y = batch_y.to(device, non_blocking=device.type == "cuda")
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            batch_size = batch_x.size(0)
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
            delta = predictions - batch_y
            close_squared_error += float(torch.square(delta[:, 0]).sum().item())
            volatility_squared_error += float(torch.square(delta[:, 1]).sum().item())
    if total_items <= 0:
        return EvaluationMetrics(
            loss=float("inf"),
            close_rmse=float("inf"),
            volatility_rmse=float("inf"),
            combined_rmse=float("inf"),
        )

    combined_squared_error = close_squared_error + volatility_squared_error
    return EvaluationMetrics(
        loss=total_loss / total_items,
        close_rmse=math.sqrt(close_squared_error / total_items),
        volatility_rmse=math.sqrt(volatility_squared_error / total_items),
        combined_rmse=math.sqrt(combined_squared_error / (total_items * 2)),
    )


def train_model(
    *,
    base_model: ForecasterModel | None = None,
    model_config: ModelConfig | dict[str, Any] | None = None,
    vectors: list[FeatureVector] | None = None,
    data_dir: str | Path | None = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    checkpoint_path: str | Path | None = None,
    save_checkpoint: bool = True,
    device: str | torch.device | None = None,
) -> TrainingResult:
    device_obj = _resolve_device(device)
    active_model_config = ModelConfig.from_model(base_model) if base_model is not None else _coerce_model_config(model_config)
    sequence_groups = load_training_sequences_from_data(data_dir)
    if vectors:
        sequence_groups.append(list(vectors))

    x, y = _build_training_tensors(sequence_groups)
    if x is None or y is None:
        model = copy.deepcopy(base_model).to(device_obj) if base_model is not None else _build_model(active_model_config, device=device_obj)
        model.eval()
        return TrainingResult(
            model=model,
            summary=TrainingRunSummary(
                model_config=ModelConfig.from_model(model),
                device=str(device_obj),
                epochs_requested=epochs,
                epochs_completed=0,
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_split=validation_split,
                early_stopping_patience=early_stopping_patience,
                sequence_groups=len(sequence_groups),
                total_windows=0,
                train_windows=0,
                validation_windows=0,
                checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else str(BEST_MODEL_PATH),
                checkpoint_saved=False,
                best_epoch=None,
                metrics=None,
            ),
        )

    train_x, train_y, val_x, val_y = _split_train_validation(x, y, validation_split)
    # The current Torch build emits deprecation warnings from DataLoader pinning internals.
    # For this dataset size, disabling pinning keeps training clean without a meaningful throughput hit.
    pin_memory = False
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=min(batch_size, len(train_x)),
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=min(batch_size, len(val_x)),
        shuffle=False,
        pin_memory=pin_memory,
    )

    work_model = (
        copy.deepcopy(base_model).to(device_obj)
        if base_model is not None
        else _build_model(active_model_config, device=device_obj)
    )
    work_model.train()
    optimizer = torch.optim.AdamW(work_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loss_fn = nn.SmoothL1Loss(beta=0.02)

    best_metrics: EvaluationMetrics | None = None
    best_state = copy.deepcopy(work_model.state_dict())
    best_epoch: int | None = None
    completed_epochs = 0
    stale_epochs = 0
    checkpoint_target = Path(checkpoint_path) if checkpoint_path is not None else BEST_MODEL_PATH

    for epoch_index in range(epochs):
        work_model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device_obj, non_blocking=device_obj.type == "cuda")
            batch_y = batch_y.to(device_obj, non_blocking=device_obj.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            predictions = work_model(batch_x)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(work_model.parameters(), max_norm=1.0)
            optimizer.step()

        completed_epochs = epoch_index + 1
        eval_metrics = _evaluate_model(work_model, val_loader, device_obj, loss_fn)
        scheduler.step(eval_metrics.loss)

        if best_metrics is None or eval_metrics.loss + 1e-6 < best_metrics.loss:
            best_metrics = eval_metrics
            best_state = copy.deepcopy(work_model.state_dict())
            best_epoch = completed_epochs
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                break

    work_model.load_state_dict(best_state)
    work_model.eval()
    if best_metrics is None:
        best_metrics = _evaluate_model(work_model, val_loader, device_obj, loss_fn)

    summary = TrainingRunSummary(
        model_config=ModelConfig.from_model(work_model),
        device=str(device_obj),
        epochs_requested=epochs,
        epochs_completed=completed_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_split=validation_split,
        early_stopping_patience=early_stopping_patience,
        sequence_groups=len(sequence_groups),
        total_windows=len(x),
        train_windows=len(train_x),
        validation_windows=len(val_x),
        checkpoint_path=str(checkpoint_target),
        checkpoint_saved=save_checkpoint,
        best_epoch=best_epoch,
        metrics=best_metrics,
    )

    if save_checkpoint:
        _save_model_checkpoint(work_model, checkpoint_target, summary)

    global _model, _model_artifact_metadata
    with _model_lock:
        if save_checkpoint:
            _model = copy.deepcopy(work_model).to(device_obj)
            _model.eval()
            _model_artifact_metadata = _checkpoint_metadata(
                _read_checkpoint_payload(checkpoint_target, device_obj),
                checkpoint_target,
                model=_model,
            )

    return TrainingResult(model=work_model, summary=summary)


def _predict_next_point(model: ForecasterModel, sequence: list[FeatureVector]) -> tuple[float, float]:
    device = next(model.parameters()).device
    x = torch.tensor([[item.as_list() for item in sequence]], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(x).squeeze(0)
    close_scale = float((_model_artifact_metadata or {}).get("close_scale", DEFAULT_CLOSE_SCALE))
    pred_close = float(out[0].item()) * close_scale
    pred_vol = float(out[1].item())
    return pred_close, pred_vol


def checkpoint_exists(checkpoint_path: str | Path = BEST_MODEL_PATH) -> bool:
    return Path(checkpoint_path).exists()


def bootstrap_checkpoint(
    *,
    vectors: list[FeatureVector],
    epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    validation_split: float = 0.2,
    early_stopping_patience: int = 10,
    checkpoint_path: str | Path = BEST_MODEL_PATH,
) -> TrainingResult:
    return train_model(
        vectors=vectors,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_split=validation_split,
        early_stopping_patience=early_stopping_patience,
        checkpoint_path=checkpoint_path,
        save_checkpoint=True,
    )


def _sample_std(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if len(items) < 2:
        return 0.0
    mean = sum(items) / len(items)
    variance = sum((value - mean) ** 2 for value in items) / (len(items) - 1)
    return math.sqrt(max(variance, 0.0))


def _build_confidence_bands(
    history_close: list[float],
    history_vol: list[float],
    forecast_close: list[float],
    forecast_vol: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    close_returns = [
        (curr - prev) / prev
        for prev, curr in zip(history_close, history_close[1:])
        if abs(prev) > 1e-12
    ]
    vol_changes = [curr - prev for prev, curr in zip(history_vol, history_vol[1:])]

    close_sigma = max(_sample_std(close_returns), 0.0025)
    latest_vol = max(history_vol[-1] if history_vol else 0.0, forecast_vol[0] if forecast_vol else 0.0)
    vol_sigma = max(_sample_std(vol_changes), latest_vol * 0.08, 0.00015)

    forecast_close_lower: list[float] = []
    forecast_close_upper: list[float] = []
    forecast_vol_lower: list[float] = []
    forecast_vol_upper: list[float] = []

    for step_idx, (pred_close, pred_vol) in enumerate(zip(forecast_close, forecast_vol), start=1):
        horizon_scale = math.sqrt(step_idx)
        close_width = max(pred_close, 1.0) * close_sigma * CONFIDENCE_Z_SCORE * horizon_scale
        vol_width = vol_sigma * CONFIDENCE_Z_SCORE * horizon_scale

        forecast_close_lower.append(max(0.0, pred_close - close_width))
        forecast_close_upper.append(pred_close + close_width)
        forecast_vol_lower.append(max(0.0, pred_vol - vol_width))
        forecast_vol_upper.append(pred_vol + vol_width)

    return (
        forecast_close_lower,
        forecast_close_upper,
        forecast_vol_lower,
        forecast_vol_upper,
    )


def get_model_artifact_metadata(
    *,
    runtime_mode: str = "fast",
    model: ForecasterModel | None = None,
    adaptation_summary: TrainingRunSummary | None = None,
) -> dict[str, Any]:
    base_metadata = dict(
        _model_artifact_metadata
        or _checkpoint_metadata(
            None,
            BEST_MODEL_PATH,
            runtime_mode=runtime_mode,
            model=model,
            adaptation_summary=adaptation_summary,
        )
    )
    base_metadata["runtime_mode"] = runtime_mode
    if model is not None:
        config = ModelConfig.from_model(model)
        base_metadata.update(
            {
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
                "dropout": config.dropout,
                "head_hidden_size": config.head_hidden_size,
            }
        )
    if adaptation_summary is not None:
        base_metadata.update(
            {
                "adaptation_epochs_completed": adaptation_summary.epochs_completed,
                "adaptation_best_epoch": adaptation_summary.best_epoch,
                "adaptation_loss": adaptation_summary.metrics.loss if adaptation_summary.metrics else None,
                "adaptation_combined_rmse": (
                    adaptation_summary.metrics.combined_rmse if adaptation_summary.metrics else None
                ),
            }
        )
    return base_metadata


def forecast_quantitative_series(
    vectors: list[FeatureVector],
    forecast_mode: str = "fast",
    horizon: str = "3d",
    forecast_dates: list[str] | None = None,
) -> dict[str, object]:
    if not vectors:
        vectors = [FeatureVector(date="", sentiment_score=0.0, market_close=0.0, market_volatility=0.0)]

    base_model = _get_model()
    training_result = (
        train_model(
            base_model=base_model,
            vectors=vectors,
            epochs=18,
            batch_size=32,
            learning_rate=5e-4,
            validation_split=0.25,
            early_stopping_patience=4,
            save_checkpoint=False,
        )
        if forecast_mode == "quick_train"
        else None
    )
    model = training_result.model if training_result is not None else base_model

    history_vectors = vectors[-30:]
    history_timestamps = [item.date for item in history_vectors]
    history_close = [float(item.market_close) for item in history_vectors]
    history_vol = [float(item.market_volatility) for item in history_vectors]

    steps = _parse_horizon_steps(horizon)
    rolling = history_vectors[-SEQUENCE_LENGTH:]
    forecast_close: list[float] = []
    forecast_vol: list[float] = []
    forecast_timestamps: list[str] = []

    last_date = history_timestamps[-1] if history_timestamps else ""
    for step in range(steps):
        fixed_sequence = build_last5_sequence(rolling)
        next_close, next_vol = _predict_next_point(model, fixed_sequence)
        last_vector = fixed_sequence[-1]
        if forecast_dates and step < len(forecast_dates):
            next_date_label = str(forecast_dates[step])
        else:
            next_date_label = f"{last_date}+{step + 1}" if last_date else f"t+{step + 1}"
        next_vector = FeatureVector.from_market_state(
            date=next_date_label,
            sentiment_score=float(last_vector.sentiment_score),
            market_close=next_close,
            market_volatility=next_vol,
            previous_close=float(last_vector.market_close),
            previous_volatility=float(last_vector.market_volatility),
        )
        rolling = (rolling + [next_vector])[-SEQUENCE_LENGTH:]

        forecast_timestamps.append(next_date_label)
        forecast_close.append(next_close)
        forecast_vol.append(next_vol)

    (
        forecast_close_lower,
        forecast_close_upper,
        forecast_vol_lower,
        forecast_vol_upper,
    ) = _build_confidence_bands(history_close, history_vol, forecast_close, forecast_vol)

    vol_values = [*history_vol, *forecast_vol, *forecast_vol_lower, *forecast_vol_upper]
    if vol_values:
        vol_min = min(vol_values)
        vol_max = max(vol_values)
        spread = max(vol_max - vol_min, 1e-6)
        vol_scale = {
            "suggested_ymin": max(0.0, vol_min - spread * 0.15),
            "suggested_ymax": vol_max + spread * 0.15,
        }
    else:
        vol_scale = {"suggested_ymin": 0.0, "suggested_ymax": 1.0}

    return {
        "prediction": {
            "close": float(forecast_close[-1]),
            "volatility": float(forecast_vol[-1]),
            "horizon": horizon,
        },
        "model": get_model_artifact_metadata(
            runtime_mode=forecast_mode,
            model=model,
            adaptation_summary=training_result.summary if training_result is not None else None,
        ),
        "series": {
            "timestamps": history_timestamps,
            "history_close": history_close,
            "history_volatility": history_vol,
            "forecast_timestamps": forecast_timestamps,
            "forecast_close": forecast_close,
            "forecast_close_lower": forecast_close_lower,
            "forecast_close_upper": forecast_close_upper,
            "forecast_volatility": forecast_vol,
            "forecast_volatility_lower": forecast_vol_lower,
            "forecast_volatility_upper": forecast_vol_upper,
            "forecast_confidence_level": FORECAST_CONFIDENCE_LEVEL,
            "volatility_scale": vol_scale,
        },
    }
