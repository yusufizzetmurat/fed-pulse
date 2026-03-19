from __future__ import annotations

import copy
import csv
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn
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

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path("/data") if Path("/data").exists() else BACKEND_ROOT.parent / "data"
MODELS_DIR = BACKEND_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "forecaster_best.pt"

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


class ForecasterModel(nn.Module):
    def __init__(
        self,
        input_size: int = FEATURE_SIZE,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
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
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]
        return self.head(last_step)


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


_model: ForecasterModel | None = None
_model_lock = threading.Lock()


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_payload(model: ForecasterModel, best_loss: float) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "best_loss": float(best_loss),
        "input_size": FEATURE_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "close_scale": DEFAULT_CLOSE_SCALE,
    }


def _save_model_checkpoint(model: ForecasterModel, checkpoint_path: Path, best_loss: float) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(model, best_loss), checkpoint_path)


def _load_model_checkpoint(model: ForecasterModel, checkpoint_path: Path, device: torch.device) -> bool:
    if not checkpoint_path.exists():
        return False

    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state_dict)
    return True


def _get_model() -> ForecasterModel:
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            device = _resolve_device()
            model = ForecasterModel().to(device)
            _load_model_checkpoint(model, BEST_MODEL_PATH, device)
            model.eval()
            _model = model
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


def load_training_sequences_from_data(data_dir: str | Path | None = None) -> list[list[FeatureVector]]:
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    if not root.exists():
        return []

    sequences: list[list[FeatureVector]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue

        try:
            if path.suffix == ".json":
                records = _load_json_records(path)
            elif path.suffix == ".jsonl":
                records = _load_jsonl_records(path)
            elif path.suffix == ".csv":
                records = _load_csv_records(path)
            else:
                continue
            vectors = build_feature_vectors(records)
        except Exception:
            continue

        if len(vectors) >= SEQUENCE_LENGTH + 1:
            sequences.append(vectors)

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
                    min(max(target.market_close / DEFAULT_CLOSE_SCALE, 0.0), 1.0),
                    min(max(target.market_volatility, 0.0), 1.0),
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


def _evaluate_model(model: ForecasterModel, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=device.type == "cuda")
            batch_y = batch_y.to(device, non_blocking=device.type == "cuda")
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            batch_size = batch_x.size(0)
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
    return total_loss / max(total_items, 1)


def train_model(
    *,
    base_model: ForecasterModel | None = None,
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
) -> ForecasterModel:
    device_obj = _resolve_device(device)
    sequence_groups = load_training_sequences_from_data(data_dir)
    if vectors:
        sequence_groups.append(list(vectors))

    x, y = _build_training_tensors(sequence_groups)
    if x is None or y is None:
        model = copy.deepcopy(base_model or _get_model()).to(device_obj)
        model.eval()
        return model

    train_x, train_y, val_x, val_y = _split_train_validation(x, y, validation_split)
    pin_memory = device_obj.type == "cuda"
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

    work_model = copy.deepcopy(base_model or _get_model()).to(device_obj)
    work_model.train()
    optimizer = torch.optim.AdamW(work_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loss_fn = nn.SmoothL1Loss(beta=0.02)

    best_eval_loss = float("inf")
    best_state = copy.deepcopy(work_model.state_dict())
    stale_epochs = 0
    checkpoint_target = Path(checkpoint_path) if checkpoint_path is not None else BEST_MODEL_PATH

    for _ in range(epochs):
        work_model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device_obj, non_blocking=pin_memory)
            batch_y = batch_y.to(device_obj, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            predictions = work_model(batch_x)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(work_model.parameters(), max_norm=1.0)
            optimizer.step()

        eval_loss = _evaluate_model(work_model, val_loader, device_obj, loss_fn)
        scheduler.step(eval_loss)

        if eval_loss + 1e-6 < best_eval_loss:
            best_eval_loss = eval_loss
            best_state = copy.deepcopy(work_model.state_dict())
            stale_epochs = 0
            if save_checkpoint:
                _save_model_checkpoint(work_model, checkpoint_target, best_eval_loss)
        else:
            stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                break

    work_model.load_state_dict(best_state)
    work_model.eval()

    if save_checkpoint and best_eval_loss == float("inf"):
        best_eval_loss = _evaluate_model(work_model, val_loader, device_obj, loss_fn)
        _save_model_checkpoint(work_model, checkpoint_target, best_eval_loss)

    global _model
    with _model_lock:
        if save_checkpoint:
            _model = copy.deepcopy(work_model).to(device_obj)
            _model.eval()

    return work_model


def _predict_next_point(model: ForecasterModel, sequence: list[FeatureVector]) -> tuple[float, float]:
    device = next(model.parameters()).device
    x = torch.tensor([[item.as_list() for item in sequence]], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(x).squeeze(0)
    pred_close = float(out[0].item()) * DEFAULT_CLOSE_SCALE
    pred_vol = float(out[1].item())
    return pred_close, pred_vol


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


def forecast_quantitative_series(
    vectors: list[FeatureVector],
    forecast_mode: str = "fast",
    horizon: str = "3d",
) -> dict[str, object]:
    if not vectors:
        vectors = [FeatureVector(date="", sentiment_score=0.0, market_close=0.0, market_volatility=0.0)]

    base_model = _get_model()
    model = (
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
        else base_model
    )

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
