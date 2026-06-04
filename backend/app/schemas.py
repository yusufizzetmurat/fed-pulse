from datetime import date
from typing import Any, Literal

# Local alias so models that already have a ``date: str`` field can still
# annotate other fields with the ``datetime.date`` type without shadowing.
_date_t = date

from pydantic import BaseModel, ConfigDict, Field, field_validator


_STRICT_REQUEST_CONFIG = ConfigDict(extra="forbid", strict=True, frozen=True)
# Response models stay open to extras so the OpenAPI snapshot does not churn;
# `frozen` still blocks mutation after construction.
_FORBID_FROZEN_CONFIG = ConfigDict(frozen=True)
# #99 strict response config: enables Pydantic v2 strict mode so the
# numeric fields refuse cross-type coercion at construction time.
# Concretely it rejects:
#   - float -> int field   (a numpy.float64 leak into lookback_days)
#   - str   -> any numeric (string concat artefacts)
#   - bool  -> any numeric (True/False misuse)
# Pydantic v2 strict_float still accepts a bare ``int`` (treated as a
# lossless promotion), so a numpy.int64 leak into close/volatility
# is NOT caught here -- the guard is asymmetric across the numeric
# directions. It also accepts numpy.float64 against a float field
# because numpy.float64 is a subclass of Python float. Decimal and
# Fraction are likewise silently coerced for float fields in practice
# (pydantic/pydantic#11131) despite the docs implying otherwise. The
# value-add is the directional rejections above; the asymmetry is
# documented so the next audit pass knows what gap remains.
# Applied to the two leaf-level numeric response models whose
# service-layer builders have been audited end-to-end
# (MarketDataResponse and PredictionResponse).
#
# Scope caveat (Pydantic v2 semantics): strict=True is model-local.
# When a model with strict=True is populated as a nested field of a
# non-strict outer model (e.g. AnalyzeResponse), the outer model's
# non-strict coercion governs the validation pass and the nested
# strict guard does NOT re-fire on field values coming from the
# outer dict. Strict therefore catches numpy at direct-construction
# sites (services that build the model by name, tests that
# round-trip it, fixture factories) but not at FastAPI's
# response-serialisation boundary when the outer AnalyzeResponse is
# still _FORBID_FROZEN_CONFIG. The follow-up #99 PR that flips
# AnalyzeResponse to strict will close that hole; the leaf-level
# strict here still adds value for the direct-construction path,
# which is where the service builders actually live.
#
# Remaining response models (SentimentResponse, ChunkAttentionDiagnostics,
# ModelDiagnostics, XaiResponse, HistoryEntry, AnalyzeResponse) wait
# on matching audit passes -- this PR is the first half of the #99
# rollout.
_STRICT_RESPONSE_CONFIG = ConfigDict(strict=True, frozen=True)


class AnalyzeRequest(BaseModel):
    model_config = _STRICT_REQUEST_CONFIG

    text: str = Field(..., min_length=1, description="FOMC statement text")
    date: str = Field(..., description="Document date in ISO format: YYYY-MM-DD")
    symbol: str = Field("^GSPC", description="Market ticker, e.g. ^GSPC or DX-Y.NYB")
    horizon: str = Field("3d", description="Forecast horizon label")
    include_realized: bool = Field(
        False,
        description="When true and date is in the past, include realized forward series overlay.",
    )
    include_xai: bool = Field(
        False,
        description="When true, return per-sentence + per-token XAI attribution alongside the forecast.",
    )
    mask_sentence_indices: list[int] = Field(
        default_factory=list,
        description=(
            "Counterfactual: 0-based indices of sentences to drop from the "
            "text before running the pipeline. Sentences are split using the "
            "same tokenizer that produces xai.sentences. Empty list = no mask."
        ),
    )
    as_of_date: _date_t | None = Field(
        default=None,
        description=(
            "Replay-mode anchor. When set, the pipeline runs as if today "
            "were this date: only the walk-forward fold whose train_end "
            "precedes this date is used for the forecaster + trajectory, "
            "and historical-analog retrieval is filtered to "
            "event_date <= as_of_date. None = live mode (default)."
        ),
    )

    @field_validator("as_of_date", mode="before")
    @classmethod
    def _parse_as_of_date(cls, value: Any) -> Any:
        """Accept ISO ``YYYY-MM-DD`` strings under the model's strict
        config so the JSON wire shape stays a string. Pydantic v2 strict
        otherwise refuses str->date coercion."""

        if value is None or isinstance(value, _date_t):
            return value
        if isinstance(value, str):
            try:
                return _date_t.fromisoformat(value[:10])
            except ValueError as exc:
                raise ValueError(f"as_of_date must be ISO YYYY-MM-DD, got {value!r}") from exc
        return value


class SentimentResponse(BaseModel):
    """Sentence-aggregated stance label for the FOMC excerpt plus an
    out-of-distribution chip so callers can tell when the input lies
    outside the corpus the head was trained on."""

    model_config = _FORBID_FROZEN_CONFIG

    label: str = Field(
        ...,
        description="Aggregated stance label across the input text. One of `hawkish`, `dovish`, `neutral`, or `UNKNOWN` when the input was empty / non-Latin / too short to score.",
    )
    score: float = Field(
        ...,
        description="Probability of the predicted `label` after chunk-level aggregation. In `[0, 1]`.",
    )
    raw: list[dict[str, float | str]] = Field(
        ...,
        description="Per-class probabilities from the underlying classifier before label selection. Each entry is `{label: str, score: float}`.",
    )
    ood_energy: float | None = Field(
        default=None,
        description=(
            "OOD score for the input text. Carries the **Mahalanobis distance** "
            "to the nearest class centroid in the encoder's CLS embedding space "
            "when the active detector is the Lee et al. NeurIPS 2018 manifest "
            "(`forecaster_best.ood_mahalanobis.json`). Falls back to the "
            "Liu et al. NeurIPS 2020 free-energy score "
            "`E(x) = -T * logsumexp(logits / T)` when only the legacy manifest "
            "(`forecaster_best.ood.json`) is on disk. `null` when no calibration "
            "manifest is present. Lower means closer to the training distribution."
        ),
    )
    ood_threshold: float | None = Field(
        default=None,
        description=(
            "Calibrated in-distribution ceiling, set at the 95th percentile of "
            "training-corpus scores during calibration. Inputs above this score "
            "flag out-of-distribution. `null` when no manifest is present."
        ),
    )
    is_in_distribution: bool | None = Field(
        default=None,
        description=(
            "`true` when `ood_energy <= ood_threshold`, i.e. the input embedding "
            "lies near the FOMC training distribution and the stance label is "
            "trustworthy. `false` flags off-domain inputs whose stance label "
            "should be ignored. `null` when no manifest is present."
        ),
    )


class MarketDataResponse(BaseModel):
    """Snapshot of the market series at the trading session used as the
    forecast anchor. ``requested_date`` is what the caller asked for and
    ``date_used`` is the actual session resolved after weekend/holiday
    rollback."""

    model_config = _STRICT_RESPONSE_CONFIG

    symbol: str = Field(..., description="Yahoo Finance ticker the snapshot was fetched for, e.g. `^GSPC` or `DX-Y.NYB`.")
    requested_date: str = Field(..., description="ISO `YYYY-MM-DD` date the caller asked for in the original request.")
    date_used: str = Field(..., description="ISO `YYYY-MM-DD` of the trading session actually used after rolling back over weekends/holidays.")
    lookback_days: int = Field(..., description="Calendar-day lookback window pulled from yfinance to populate the history series.")
    close: float = Field(..., description="Adjusted close price at `date_used` in the ticker's native currency.")
    volatility_5d: float = Field(..., description="Trailing 5-session realised volatility of log returns ending at `date_used`.")


class PredictionResponse(BaseModel):
    """Point forecast emitted by the forecaster head for the chosen horizon.
    Confidence bands and the underlying trajectory are exposed separately in
    ``ForecastSeriesResponse``; this object only carries the headline scalars."""

    model_config = _STRICT_RESPONSE_CONFIG

    close: float = Field(..., description="Forecasted adjusted close at the end of the horizon, in the ticker's native currency.")
    volatility: float = Field(..., description="Forecasted realised volatility at the end of the horizon, in the same units as `market.volatility_5d`.")
    horizon: str = Field(..., description="Horizon label this forecast applies to, e.g. `1d`, `3d`, `5d` trading sessions ahead.")


class ChunkAttentionDiagnostics(BaseModel):
    """Per-chunk attention trace from the encoder's long-document aggregator.
    Exposed so the XAI panel can show which chunks of a long FOMC document
    dominated the pooled representation and how the elapsed-time decay
    reshaped their weights."""

    model_config = _FORBID_FROZEN_CONFIG

    chunk_count: int = Field(..., description="Number of fixed-size chunks the input was split into before pooling.")
    weights: list[float] = Field(..., description="Post-softmax attention weight per chunk after time-decay; sums to ~1.0.")
    decay_coeffs: list[float] = Field(..., description="Time-decay multiplier applied to each chunk before softmax; in `(0, 1]`.")
    chunk_previews: list[str] = Field(..., description="Leading text snippet of each chunk for display in the XAI panel.")
    lambda_value: float = Field(..., description="Decay rate used to compute `decay_coeffs`; higher = older chunks down-weighted faster.")


class ModelDiagnosticsResponse(BaseModel):
    """Forecaster runtime + checkpoint metadata surfaced beside the prediction.
    Exposes which file the service loaded, the network shape it was built
    with, and the most recent training/adaptation losses recorded on disk."""

    model_config = _FORBID_FROZEN_CONFIG

    checkpoint_path: str = Field(..., description="Absolute path of the forecaster checkpoint the service loaded.")
    checkpoint_exists: bool = Field(..., description="True when the checkpoint file is present on disk at `checkpoint_path`.")
    checkpoint_loaded: bool = Field(..., description="True when the in-memory forecaster matches the on-disk checkpoint.")
    runtime_mode: str = Field(..., description="Runtime mode the model is served under, e.g. `fast`, `quick_train`, `real_train`.")
    hidden_size: int = Field(..., description="Hidden dimension of the LSTM backbone in units.")
    num_layers: int = Field(..., description="Number of stacked LSTM layers in the forecaster.")
    dropout: float = Field(..., description="Dropout probability applied between LSTM layers in `[0, 1)`.")
    head_hidden_size: int = Field(..., description="Hidden dimension of the regression head MLP in units.")
    close_scale: float = Field(..., description="Scaling factor applied to close-price targets before training.")
    sequence_length: int = Field(..., description="Number of past timesteps fed into the LSTM each forward pass.")
    best_loss: float | None = Field(default=None, description="Best validation loss recorded during the most recent training run; null when no run logged it.")
    combined_rmse: float | None = Field(default=None, description="Combined close+volatility RMSE from the best run; null when unavailable.")
    adaptation_epochs_completed: int | None = Field(default=None, description="Number of adaptation epochs completed in the most recent quick_train/real_train pass.")
    adaptation_best_epoch: int | None = Field(default=None, description="Epoch index of the lowest adaptation loss in the most recent run.")
    adaptation_loss: float | None = Field(default=None, description="Best loss value observed during the most recent adaptation pass.")
    adaptation_combined_rmse: float | None = Field(default=None, description="Combined close+volatility RMSE from the most recent adaptation pass.")
    decay_rate: float | None = Field(default=None, description="Elapsed-time decay rate (lambda) used by the chunk aggregator; null when no decay is configured.")
    chunk_attention: ChunkAttentionDiagnostics | None = Field(default=None, description="Per-chunk attention trace from the encoder aggregator; null when the input was short enough to skip chunking.")
    encoder_key: str | None = Field(
        default=None,
        description="Encoder alias backing the multi-axis classifier (e.g. `finbert_fed_adjacent`); null when no multi-axis checkpoint is loaded.",
    )


class ForecastSeriesResponse(BaseModel):
    """Chart-ready timeseries arrays for the forecaster panel — history,
    forecast point + confidence bands, and the optional realised overlay
    for replay mode. Indices are aligned across all parallel lists."""

    model_config = _FORBID_FROZEN_CONFIG

    timestamps: list[str] = Field(..., description="ISO timestamps for the historical close/vol arrays, oldest-first.")
    history_close: list[float] = Field(..., description="Historical close prices aligned to `timestamps`, in the ticker's native currency.")
    history_volatility: list[float] = Field(..., description="Historical realised volatility aligned to `timestamps`.")
    forecast_timestamps: list[str] = Field(..., description="ISO timestamps for each forecast step, ordered ascending.")
    forecast_close: list[float] = Field(..., description="Forecast point estimate of close at each forecast timestamp.")
    forecast_close_lower: list[float] = Field(..., description="Lower edge of the close confidence band at each forecast timestamp.")
    forecast_close_upper: list[float] = Field(..., description="Upper edge of the close confidence band at each forecast timestamp.")
    forecast_volatility: list[float] = Field(..., description="Forecast point estimate of realised volatility at each forecast timestamp.")
    forecast_volatility_lower: list[float] = Field(..., description="Lower edge of the volatility confidence band at each forecast timestamp.")
    forecast_volatility_upper: list[float] = Field(..., description="Upper edge of the volatility confidence band at each forecast timestamp.")
    forecast_confidence_level: float = Field(..., description="Nominal confidence level of the forecast bands in `(0, 1)`, e.g. 0.8 for 80%.")
    realized_timestamps: list[str] | None = Field(default=None, description="ISO timestamps for the realised overlay; null when replay mode is off.")
    realized_close: list[float] | None = Field(default=None, description="Realised close prices aligned to `realized_timestamps`; null in live mode.")
    realized_volatility: list[float] | None = Field(default=None, description="Realised volatility aligned to `realized_timestamps`; null in live mode.")
    volatility_scale: dict[str, float] = Field(..., description="Scaling constants applied to the volatility series so chart axes can re-scale display units.")
    forecast_band_source: str = Field(
        default="gaussian_z",
        description="Source of the forecast bands: 'gaussian_z' (z-score) or 'conformal'.",
    )
    conformal_coverage: float | None = Field(
        default=None,
        description="Nominal coverage of conformal bands when forecast_band_source='conformal'.",
    )


class XaiTokenAttribution(BaseModel):
    """One token's contribution to a sentence's salience score.

    Emitted under :class:`XaiSentenceAttribution.topTokens` so the
    frontend can highlight the highest-weighted tokens inline.
    """

    model_config = _FORBID_FROZEN_CONFIG

    token: str = Field(..., description="Surface token (whitespace-stripped) drawn from the input text.")
    weight: float = Field(..., description="Salience weight in [0,1] indicating this token's share of the sentence score.")


class XaiSentenceAttribution(BaseModel):
    """One input sentence with its salience score and top contributing tokens.

    Returned under :class:`XaiResponse.sentences` for the keyword-salience
    explainer. Sentences are listed in original document order.
    """

    model_config = _FORBID_FROZEN_CONFIG

    text: str = Field(..., description="Verbatim sentence text as segmented from the input statement.")
    score: float = Field(..., description="Salience score in [0,1] reflecting this sentence's influence on the prediction.")
    topTokens: list[XaiTokenAttribution] = Field(
        default_factory=list,
        description="Highest-weighted tokens within this sentence, descending by weight; empty when no tokens cross the threshold.",
    )


class XaiFeatureFamilyAttribution(BaseModel):
    """One feature-family bar on a panel attribution chart (#297).

    Emitted under :class:`XaiPanelAttribution.families`. ``magnitude`` is
    the L1 attribution sum across all features in the family (always
    non-negative); ``signed`` is the sum-with-sign so the frontend can
    colour the bar by direction.
    """

    model_config = _FORBID_FROZEN_CONFIG

    family: str = Field(..., description="Feature-family identifier (e.g., sentiment, rates, vol, calendar) grouping related input features.")
    magnitude: float = Field(..., description="L1 sum of attribution across features in this family; always non-negative.")
    signed: float = Field(..., description="Signed sum of attribution across features in this family; sign indicates push direction.")


class XaiPanelAttribution(BaseModel):
    """Integrated-gradients attribution for one /analyze panel (#297).

    Returned under :class:`XaiResponse.panels`. One entry per active
    panel (regime, rates_2y/5y/terminal, trajectory). ``unavailable``
    flips to True when the panel cannot be explained (panel not active
    on the checkpoint, kwarg mismatch, runtime error); the frontend
    then renders the "explanation unavailable" badge rather than an
    empty bar chart.
    """

    model_config = _FORBID_FROZEN_CONFIG

    panel: str = Field(..., description="Panel identifier this attribution covers, e.g. `regime`, `rates_2y`, `trajectory`.")
    target: str = Field(..., description="Target output the attribution explains, e.g. predicted class or scalar head name.")
    families: list[XaiFeatureFamilyAttribution] = Field(default_factory=list, description="Per-feature-family attribution bars; empty when the panel could not be explained.")
    n_steps: int = Field(
        default=0,
        description="Integrated-gradients integration step count used to compute this panel's attribution.",
    )
    unavailable: bool = Field(default=False, description="True when the panel could not be explained; the frontend then shows an unavailable badge.")
    reason: str | None = Field(default=None, description="Structured reason string when `unavailable` is true; null otherwise.")


class XaiResponse(BaseModel):
    """XAI envelope on the /analyze response — keyword-salience sentence
    attribution plus optional per-panel integrated-gradients bars when
    the active checkpoint exposes explainable panels."""

    model_config = _FORBID_FROZEN_CONFIG

    method: str = Field(default="keyword_salience_v1", description="XAI method identifier for the sentence-level explainer.")
    sentences: list[XaiSentenceAttribution] = Field(default_factory=list, description="Per-sentence salience scores with top contributing tokens, in document order.")
    panels: list[XaiPanelAttribution] = Field(
        default_factory=list,
        description="Per-panel integrated-gradients attribution; populated when include_xai is true and the checkpoint surfaces at least one explainable panel.",
    )


class CredibilityResponse(BaseModel):
    """Credibility / drift signals beside the headline forecast. Combines
    a stance-drift score against prior statements with realised-vs-stated
    and market-implied gap estimates when those calibration sources are
    available."""

    model_config = _FORBID_FROZEN_CONFIG

    drift_score: float = Field(..., description="Cosine drift of the current statement embedding vs the mean of prior statements; higher = larger shift.")
    realized_vs_stated_gap: float | None = Field(default=None, description="Signed gap between realised path and the statement's stated trajectory; null when no calibration is on disk.")
    market_implied_gap: float | None = Field(default=None, description="Signed gap between market-implied path and the statement's stated trajectory; null when futures data is unavailable.")
    months_since_reversal: int | None = Field(default=None, description="Months since the last hawkish<->dovish reversal in the stance series; null when no reversal is on record.")
    drift_trend: list[float] = Field(
        default_factory=list,
        description="Per-meeting drift sparkline newest-last; each entry is the cosine distance of one prior statement embedding to the mean of the remaining priors.",
    )


class MultiAxisStanceCard(BaseModel):
    """Stance prediction from the multi-task head."""

    model_config = _FORBID_FROZEN_CONFIG

    label: str = Field(..., description="hawkish | dovish | neutral")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probability of the predicted stance label in `[0, 1]` after softmax.")
    distribution: dict[str, float] = Field(
        default_factory=dict,
        description="Per-class softmax probability over hawkish/dovish/neutral.",
    )


class MultiAxisTimeCard(BaseModel):
    """Forward-looking horizon classification from the multi-task head.

    Two classes: ``forward looking`` (the statement references future
    actions or expectations) vs ``not forward looking`` (backward-looking
    or current-state only). Sourced from the gtfintechlab ``time_label``
    column; trained on ~5 992 labelled sentences.
    """

    model_config = _FORBID_FROZEN_CONFIG

    label: str = Field(..., description="forward looking | not forward looking")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probability of the predicted label in `[0, 1]` after softmax.")
    distribution: dict[str, float] = Field(default_factory=dict, description="Per-class softmax probability over the two forward-looking classes.")


class MultiAxisCertaintyCard(BaseModel):
    """Certainty prediction from the multi-task head."""

    model_config = _FORBID_FROZEN_CONFIG

    label: str = Field(..., description="certain | uncertain | neutral")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probability of the predicted certainty label in `[0, 1]` after softmax.")
    distribution: dict[str, float] = Field(default_factory=dict, description="Per-class softmax probability over the three certainty classes.")


class MultiAxisBlock(BaseModel):
    """Multi-task head per-axis predictions surfaced on /analyze (#78).

    Active axes: stance / certainty / time. The factor axis (GSS
    market-derived regression target) was retired — text cannot predict it
    and the training pool had 0% coverage. Topic was retired in ADR 0044.
    """

    model_config = _FORBID_FROZEN_CONFIG

    stance: MultiAxisStanceCard = Field(..., description="Stance head prediction (hawkish/dovish/neutral) with class distribution.")
    certainty: MultiAxisCertaintyCard | None = Field(default=None, description="Certainty head prediction; null when the active checkpoint omits this axis.")
    time: MultiAxisTimeCard | None = Field(default=None, description="Forward-looking time head prediction; null when the active checkpoint omits this axis.")


class RatesReactionCard(BaseModel):
    """Per-rates-head market-reaction card surfaced on /analyze/market (#293).

    The card carries the model's point prediction in basis points for one
    rates head (``2y`` / ``5y`` / ``terminal``), the symmetric conformal
    band derived from the val-fitted residual quantile, the directional
    bucket (``easing`` / ``neutral`` / ``tightening``) the auxiliary
    classifier emits, and the per-class probability distribution so the
    frontend can render a probability bar alongside the headline number.
    """

    model_config = _FORBID_FROZEN_CONFIG

    head: str = Field(..., description="Head short name: 2y | 5y | terminal")
    point_bps: float = Field(..., description="Point prediction in basis points.")
    lower_bps: float | None = Field(
        default=None,
        description="Conformal band lower bound in bps. None when no calibration sidecar is present.",
    )
    upper_bps: float | None = Field(
        default=None,
        description="Conformal band upper bound in bps. None when no calibration sidecar is present.",
    )
    coverage: float | None = Field(
        default=None,
        description="Nominal conformal coverage (1 - alpha). None when bands are absent.",
    )
    # #317 finding #10: when the checkpoint has no aux classifier
    # mounted (regression-only mode or unbuilt cls head) the response
    # must clearly say "not available" rather than emitting a fake
    # argmax over uniform probabilities. Nullable fields let the
    # frontend render a "no model evidence" badge instead.
    directional_bucket: str | None = Field(
        default=None,
        description=(
            "easing | neutral | tightening (argmax of the auxiliary "
            "classifier). None when no aux classifier is mounted."
        ),
    )
    bucket_probabilities: dict[str, float] | None = Field(
        default=None,
        description=(
            "Per-bucket softmax probabilities over "
            "(easing, neutral, tightening). None when no aux "
            "classifier is mounted."
        ),
    )
    # #317 finding #3: calibrated APS prediction set per rates head
    # (subset of {easing, neutral, tightening}). None when no
    # ``rates_softmax_quantiles`` sidecar is present.
    predicted_set: list[str] | None = Field(
        default=None,
        description=(
            "Calibrated APS prediction set per rates head. None when "
            "no conformal sidecar is present."
        ),
    )


class VolRegimeReactionCard(BaseModel):
    """Vol-regime card mirroring the rates card shape on /analyze/market (#293).

    Carries the dual-head log(RV) regression prediction (or ``None`` when
    the active checkpoint mounts only the classifier), the regime
    classification distribution, and the calibrated APS prediction set
    derived from the existing softmax_quantile.
    """

    model_config = _FORBID_FROZEN_CONFIG

    log_rv_point: float | None = Field(
        default=None,
        description=(
            "Standardised log(forward realized vol) prediction from "
            "the dual-head regression branch. None on classification-only "
            "checkpoints."
        ),
    )
    log_rv_lower: float | None = Field(default=None, description="Lower edge of the 80% conformal band on `log_rv_point`; null when no conformal sidecar is present.")
    log_rv_upper: float | None = Field(default=None, description="Upper edge of the 80% conformal band on `log_rv_point`; null when no conformal sidecar is present.")
    regime_label: str = Field(
        ...,
        description="Argmax regime label: calm | normal | high.",
    )
    regime_probabilities: dict[str, float] = Field(
        default_factory=dict,
        description="Per-class softmax probabilities over the regime classes.",
    )
    predicted_set: list[str] = Field(
        default_factory=list,
        description="Calibrated APS prediction set (list of regime labels).",
    )
    coverage: float | None = Field(default=None, description="Nominal conformal coverage (1 - alpha) for the APS set; null when no sidecar is on disk.")


class MarketReactionPanel(BaseModel):
    """Bundle of four reaction cards (#293).

    Returned by ``POST /analyze/market``. Includes one card per mounted
    rates head plus the existing vol-regime card. Heads not mounted on
    the active checkpoint emit an empty ``rates`` list and ``None`` for
    ``vol_regime`` so the frontend can render a graceful empty state.
    """

    model_config = _FORBID_FROZEN_CONFIG

    rates: list[RatesReactionCard] = Field(default_factory=list, description="One reaction card per mounted rates head (2y/5y/terminal); empty when no rates heads are mounted.")
    vol_regime: VolRegimeReactionCard | None = Field(default=None, description="Vol-regime reaction card; null when the active checkpoint has no regime head.")
    encoder_alias: str | None = Field(default=None, description="Registry alias of the encoder backing this market-reaction panel; null on a stateless build.")
    checkpoint_path: str | None = Field(default=None, description="Absolute path of the checkpoint that produced these cards; null when no checkpoint is loaded.")


class RegimeClassificationCard(BaseModel):
    """Calibrated prediction-set output from the vol-regime classifier (#216).

    Surfaces the conformal APS set on the /analyze response. ``predicted_set``
    holds the class labels the calibrated threshold admits at the
    ``coverage`` level; ``set_label`` is the UI-friendly bracketed string;
    ``distribution`` is the raw per-class softmax for the inference row so
    the frontend can render a bar chart alongside the set chips.

    Populated only when the active checkpoint is classification-mode AND
    a sibling ``.conformal.json`` manifest with ``softmax_quantile`` exists.
    Pre-#216 regression-only checkpoints leave the field as ``None`` on
    the response and the legacy uncalibrated max-softmax stays on the
    ``sentiment`` block.
    """

    model_config = _FORBID_FROZEN_CONFIG

    predicted_set: list[str] = Field(..., description="Calibrated APS prediction set of regime labels admitted at the `coverage` level.")
    set_label: str = Field(..., description="UI-friendly bracketed label of the prediction set, e.g. `{calm, normal}`.")
    set_size: int = Field(..., description="Cardinality of `predicted_set`; ranges 1-3 for the 3-class regime head.")
    coverage: float = Field(..., description="Nominal coverage of the conformal APS set in `(0, 1)`.")
    distribution: dict[str, float] = Field(..., description="Per-class softmax probabilities over the regime labels for the inference row.")
    argmax_class: str = Field(..., description="Argmax regime label drawn from `distribution`; one of calm/normal/high.")
    log_rv_point: float | None = Field(
        default=None,
        description=(
            "Regression-head point prediction in standardised log(forward "
            "realized vol) space, or None on a classification-only checkpoint. "
            "See ADR 0015 / #322."
        ),
    )
    log_rv_lower: float | None = Field(
        default=None,
        description=(
            "80% conformal-band lower bound around log_rv_point (matches the "
            "existing close/vol band convention). None when no conformal "
            "manifest is on disk or the regression head is not active."
        ),
    )
    log_rv_upper: float | None = Field(default=None, description="See log_rv_lower; upper bound.")
    bucket_source: Literal["regression", "classification"] = Field(
        default="classification",
        description=(
            "Declares which head produced argmax_class: 'regression' means "
            "the 3-class label was bucketed UI-side from log_rv_point against "
            "the active checkpoint's vol_regime_quantiles cutoffs (see "
            "app.services.regime_bucketing); 'classification' means the "
            "label came from the 3-class softmax head's argmax."
        ),
    )


class RegimeRegressionCard(BaseModel):
    """Regression-head sibling block on the /analyze response (#304).

    The classification card stays the headline; this block carries the
    same dual-head regression output (``log_rv_point`` + symmetric 90%
    conformal band) as a standalone surface so a downstream consumer
    can read the continuous prediction without parsing it out of the
    classification card. Populated only when the active checkpoint
    mounts the regression head (``head_mode`` in ``regression`` /
    ``dual``) AND ``build_regime_classification_card`` returned a card
    whose ``log_rv_point`` is non-null; otherwise the field stays
    ``None`` on the response.

    Units mirror :class:`RegimeClassificationCard`: ``log_rv_point`` is
    in standardised log(forward realized vol) space (the per-fold
    train-slice standardiser the dual-head trainer fits, see
    ``log_rv_scaler`` on the run summary). ``log_rv_lower`` /
    ``log_rv_upper`` are the symmetric conformal interval at
    ``coverage`` nominal coverage; on a checkpoint without a conformal
    sidecar the bounds collapse to ``None`` even when the point
    estimate is populated.
    """

    model_config = _FORBID_FROZEN_CONFIG

    log_rv_point: float = Field(..., description="Regression-head point prediction in standardised log(forward realized vol) space.")
    log_rv_lower: float | None = Field(default=None, description="Lower edge of the symmetric conformal interval at `coverage`; null when no sidecar is on disk.")
    log_rv_upper: float | None = Field(default=None, description="Upper edge of the symmetric conformal interval at `coverage`; null when no sidecar is on disk.")
    coverage: float | None = Field(
        default=None,
        description=(
            "Nominal coverage on the conformal interval (0.9 by default; "
            "matches the manifest's nominal_coverage when the sidecar "
            "carries one). None when the regression head is mounted but "
            "no conformal manifest is on disk."
        ),
    )


class PolicyActionCard(BaseModel):
    """Mechanical policy decision extracted from the statement text (#446).

    Sibling of :class:`RegimeClassificationCard` on the /analyze
    response. Pure extraction surface — no model inference, no
    calibration. The four fields mirror the
    :class:`app.services.policy_action_extractor.PolicyAction`
    dataclass and are all optional so a statement that names no target
    range (press conference Q&A, scraping miss, non-policy text) still
    serialises as a card with every field ``None``.

    Units: ``target_range_low_bp`` / ``target_range_high_bp`` are in
    basis points (3.50% → 350). ``change_magnitude_bp`` is signed
    (positive on a hike, negative on a cut, zero on a hold). The
    frontend renders the colour by ``change_direction``: hike = red,
    cut = green, hold = neutral.
    """

    model_config = _FORBID_FROZEN_CONFIG

    target_range_low_bp: int | None = Field(
        default=None,
        description=(
            "Lower bound of the named target range, in basis points "
            "(e.g. 350 for a 3.50% lower bound). None when no target "
            "range is named in the text."
        ),
    )
    target_range_high_bp: int | None = Field(
        default=None,
        description="Upper bound of the named target range, in basis points.",
    )
    change_direction: Literal["hike", "hold", "cut"] | None = Field(
        default=None,
        description=(
            "Verb-derived direction of the action. None when no policy "
            "verb is named (e.g. press-conference Q&A) and no prior "
            "midpoint was provided to the extractor."
        ),
    )
    change_magnitude_bp: int | None = Field(
        default=None,
        description=(
            "Signed change in basis points relative to the prior "
            "meeting (positive on a hike, negative on a cut, zero on a "
            "hold). Pulled from in-prose magnitude phrases ('by 25 "
            "basis points', 'by 1/4 percentage point') first; falls "
            "back to ``this_mid - prior_mid`` when the caller supplied "
            "a prior midpoint."
        ),
    )
    balance_sheet_state: Literal["expansion", "tapering", "runoff"] | None = Field(
        default=None,
        description=(
            "Balance-sheet posture extracted from the paragraph that "
            "names balance-sheet operations. None when the paragraph "
            "is absent or carries no posture-defining keyword."
        ),
    )


class InferenceStatusSurface(BaseModel):
    """Structured error surface for the per-card inference helpers (#341).

    Sibling of :class:`RegimeClassificationCard` on the /analyze
    response. Populated when the card-build helper degrades through
    one of three structured branches:

    - ``not_classification_mode`` -- the active checkpoint emits no
      regime card by contract. Legitimate; UI renders the card as
      absent.
    - ``inference_kwarg_missing`` -- the serving call site fed (or
      omitted) a kwarg the checkpoint did not declare in its
      inference contract sidecar. Operator-facing bug.
    - ``unexpected_exception`` -- anything else; ``exception_class``
      carries the class name and ``detail`` the message so the
      operator can grep for it without parsing logs.
    """

    model_config = _FORBID_FROZEN_CONFIG

    status: str = Field(
        ...,
        description="Degradation branch token: 'not_classification_mode', 'inference_kwarg_missing', or 'unexpected_exception'.",
    )
    missing_kwarg: str | None = Field(
        default=None,
        description="Name of the inference kwarg the checkpoint expected but did not receive; only set when status is 'inference_kwarg_missing'.",
    )
    exception_class: str | None = Field(
        default=None,
        description="Python class name of the caught exception when status is 'unexpected_exception'; None otherwise.",
    )
    detail: str | None = Field(
        default=None,
        description="Free-text message lifted from the caught exception so operators can grep without parsing logs.",
    )


class RealisedOutcomeHorizon(BaseModel):
    """A single (h, log_return, realised_vol, close) row in the
    "what actually happened" reveal.

    ``realised_volatility_5d_post_event`` is the rolling stdev of the
    bar-to-bar log-returns over the 5 bars ending at t+h, where t is
    the replay ``as_of_date``. This is **post-event** by construction
    -- it measures volatility AFTER the analyzed statement -- and is
    deliberately not the same series as ``MarketDataResponse.
    volatility_5d``, which measures the rolling 5d stdev over the bars
    BEFORE the request date (the value the forecaster consumes as a
    feature). The two are intentionally distinct and never compared.
    """

    model_config = _FORBID_FROZEN_CONFIG

    horizon: int = Field(..., description="Trading-day horizon offset from the replay anchor, e.g. 1, 3, 5.")
    log_return: float | None = Field(default=None, description="Realised log return from anchor close to t+h close; null when forward data is missing.")
    realised_volatility_5d_post_event: float | None = Field(default=None, description="Rolling 5-bar realised volatility ending at t+h, measured strictly post-event.")
    close: float | None = Field(default=None, description="Realised adjusted close at t+h in the ticker's native currency.")
    date: str | None = Field(default=None, description="ISO `YYYY-MM-DD` of the trading session at t+h; null when the forward bar is unavailable.")


class RealisedOutcomeBlock(BaseModel):
    """Replay-mode reveal block — what actually happened forward of
    `as_of_date` for each tracked horizon. Surfaced only when the request
    set `as_of_date` and the forward bars are on disk."""

    model_config = _FORBID_FROZEN_CONFIG

    as_of_date: str = Field(..., description="ISO `YYYY-MM-DD` replay anchor the realised outcomes are measured forward of.")
    symbol: str = Field(..., description="Yahoo Finance ticker the realised series was pulled for.")
    horizons: list[RealisedOutcomeHorizon] = Field(..., description="Per-horizon realised outcomes ordered ascending by `horizon`.")


class ReplayModeBlock(BaseModel):
    """Replay-mode metadata for the /analyze response.

    Populated only when the request set ``as_of_date``. ``fold_id`` is
    the walk-forward fold whose checkpoint served this prediction;
    ``train_end`` is the latest date the model saw at training time --
    everything strictly after that is unseen ground truth from the
    model's point of view. ``classifier_rewind`` flags that the
    text-encoder weights are NOT rewound to ``as_of_date`` (the
    DAPT-pinned encoder is post-X), so any text-conditioned head is
    served with later-than-X weights -- documented as a known caveat
    rather than silently leaked.
    """

    model_config = _FORBID_FROZEN_CONFIG

    as_of_date: str = Field(..., description="ISO `YYYY-MM-DD` replay anchor echoed from the request.")
    fold_id: str | None = Field(default=None, description="Walk-forward fold identifier whose checkpoint served this prediction; null when no fold matched.")
    train_end: str | None = Field(default=None, description="Latest date the fold's training slice contained; everything after is unseen by the model.")
    classifier_rewind: bool = Field(default=False, description="True when the text-encoder weights are also rewound to `as_of_date`; false flags the known DAPT-pinned caveat.")
    forecaster_checkpoint_rewound: bool = Field(
        default=False,
        description="True once the forecaster is wired to load the per-fold checkpoint identified by `fold_id`; false flags the scaffolding-only state.",
    )
    notes: list[str] = Field(default_factory=list, description="Free-text advisories on the replay run, e.g. caveats about partial rewind.")


class AnalyzeResponse(BaseModel):
    """Top-level envelope for `POST /analyze`. Bundles the stance classifier
    output, the forecaster headline + bands, the market snapshot used as
    anchor, runtime diagnostics, and the optional XAI, replay, multi-axis,
    regime, rates and policy sub-blocks."""

    model_config = _FORBID_FROZEN_CONFIG

    sentiment: SentimentResponse = Field(..., description="Stance classifier output for the input statement including OOD chip.")
    prediction: PredictionResponse = Field(..., description="Headline scalar forecast at the chosen horizon.")
    market: MarketDataResponse = Field(..., description="Market snapshot at the trading session used as forecast anchor.")
    model: ModelDiagnosticsResponse = Field(..., description="Forecaster runtime + checkpoint diagnostics for this response.")
    series: ForecastSeriesResponse = Field(..., description="Chart-ready history, forecast and band arrays for the panel.")
    replay: ReplayModeBlock | None = Field(default=None, description="Replay-mode metadata; populated only when the request set `as_of_date`.")
    realised_outcome: RealisedOutcomeBlock | None = Field(default=None, description="What-actually-happened reveal beside the forecast; populated only in replay mode.")
    xai: XaiResponse | None = Field(default=None, description="XAI sentence + panel attribution; populated when `include_xai=true`.")
    credibility: CredibilityResponse | None = Field(default=None, description="Stance-drift and credibility signals; null when no history is available.")
    multi_axis: MultiAxisBlock | None = Field(default=None, description="Multi-task head per-axis predictions; null when no multi-axis checkpoint is loaded.")
    regime_classification: RegimeClassificationCard | None = Field(default=None, description="Calibrated regime prediction set; null when the active checkpoint has no regime head.")
    regime_regression: RegimeRegressionCard | None = Field(
        default=None,
        description="Regression sibling of the regime card carrying `log_rv_point` + conformal interval; null when the dual-head regression is absent.",
    )
    rates_reaction: list[RatesReactionCard] | None = Field(
        default=None,
        description="Per-rates-head reaction cards (2y/5y/terminal); null on legacy single-head runs, empty when heads exist but no rows were produced.",
    )
    policy_action: PolicyActionCard | None = Field(
        default=None,
        description="Mechanical policy decision extracted from the text via regex/keyword; null when the statement names no target range.",
    )
    regime_classification_status: InferenceStatusSurface | None = Field(
        default=None,
        description="Structured degradation surface mutually exclusive with `regime_classification`; carries the reason the regime card was not built.",
    )


class HistoryEntry(BaseModel):
    """One persisted analyze run summarised for the history list. Carries
    the headline scalars and regime summary so the listing page can render
    without re-fetching the full payload."""

    id: str = Field(..., description="UUID identifier of the persisted analyze run.")
    created_at: str = Field(..., description="ISO-8601 UTC timestamp the run was persisted.")
    symbol: str = Field(..., description="Yahoo Finance ticker the run was anchored to.")
    document_date: str = Field(..., description="ISO `YYYY-MM-DD` event date of the analyzed FOMC document.")
    horizon: str = Field(..., description="Forecast horizon label, e.g. `1d`, `3d`, `5d`.")
    forecast_mode: str = Field(..., description="Forecast mode the run was executed under: `fast`, `quick_train`, or `real_train`.")
    stance: str = Field(..., description="Aggregated stance label persisted with the row: hawkish/dovish/neutral/UNKNOWN.")
    sentiment_score: float | None = Field(default=None, description="Unsigned confidence of the persisted stance label in `[0, 1]`; null when missing.")
    stance_score: float | None = Field(
        default=None,
        description="Signed stance value `P(hawkish) - P(dovish)` from the multi-axis distribution; null when the row pre-dates multi-axis.",
    )
    predicted_close: float | None = Field(default=None, description="Persisted point forecast of close at the chosen horizon; null when missing.")
    current_close: float | None = Field(default=None, description="Anchor-session close from the persisted market snapshot; null when missing.")
    predicted_volatility: float | None = Field(default=None, description="Persisted point forecast of realised volatility at the chosen horizon; null when missing.")
    text_excerpt: str | None = Field(default=None, description="Leading excerpt of the analyzed statement text for the listing tile; null when redacted.")
    argmax_regime: str | None = Field(
        default=None,
        description="Argmax regime label from the persisted regime card; null when the row pre-dates the regime head.",
    )
    argmax_probability: float | None = Field(default=None, description="Probability of the `argmax_regime` label in `[0, 1]`; null when no regime card was persisted.")
    regime_set_size: int | None = Field(default=None, description="Cardinality of the persisted conformal regime set; null when no regime card was persisted.")


class HistoryDetail(HistoryEntry):
    payload: dict[str, Any] = Field(..., description="Full persisted analyze payload as a raw JSON object.")


class HistoryList(BaseModel):
    """Paginated listing envelope for the history page."""

    items: list[HistoryEntry] = Field(..., description="History entries on the current page, newest-first by default.")
    total: int = Field(..., description="Total number of history entries available across all pages.")
    limit: int = Field(..., description="Page size echoed from the request.")
    offset: int = Field(..., description="Page offset echoed from the request.")


class StanceContextPoint(BaseModel):
    """One historical (date, score) pair for the rolling z-score baseline."""

    document_date: str = Field(..., description="ISO `YYYY-MM-DD` event date of the historical FOMC row.")
    stance_score: float = Field(..., description="Signed stance score `P(hawkish) - P(dovish)` for the historical row.")


class StanceContextResponse(BaseModel):
    """Trailing stance-score summary for the dashboard tile.

    ``stance_score = P(hawkish) - P(dovish)`` per the validity study;
    the tile renders the current run as a z-score against this trailing
    mean/std rather than as a raw absolute number. ``mean`` and ``std``
    are ``null`` when fewer than two usable historical rows are found,
    in which case the tile falls back to the raw-value rendering.
    """

    n: int = Field(..., description="Number of usable historical rows in `history`.")
    mean: float | None = Field(..., description="Trailing mean of `stance_score`; null when fewer than two rows are usable.")
    std: float | None = Field(..., description="Trailing sample standard deviation of `stance_score`; null when fewer than two rows are usable.")
    history: list[StanceContextPoint] = Field(..., description="Historical (date, score) pairs ordered oldest-first.")


class HistoryRealizedResponse(BaseModel):
    """Realised forward market path for a single persisted history run,
    plus the bucketed realised regime label so the detail page can
    render a predicted-vs-realised badge."""

    run_id: str = Field(..., description="UUID of the history run this realised payload belongs to.")
    symbol: str = Field(..., description="Yahoo Finance ticker the realised series was pulled for.")
    document_date: str = Field(..., description="ISO `YYYY-MM-DD` event date the run anchored to.")
    horizon: str = Field(..., description="Forecast horizon label echoed from the run, e.g. `3d`.")
    timestamps: list[str] = Field(..., description="ISO timestamps of the forward sessions, oldest-first.")
    close: list[float] = Field(..., description="Realised close prices aligned to `timestamps`.")
    volatility: list[float] = Field(..., description="Realised volatility aligned to `timestamps`.")
    realized_regime: str | None = Field(
        default=None,
        description="Realised regime label derived from the 10d-forward vol path; null when the classifier cutoffs are unavailable on this host.",
    )


class HistoryRealizedBatchResponse(BaseModel):
    """Batched realized payloads keyed by run_id. Missing runs (deleted,
    yfinance failure) come back under ``missing`` so the caller can render
    a partial result instead of failing the whole page."""

    items: dict[str, HistoryRealizedResponse] = Field(..., description="Realised payloads keyed by history run UUID.")
    missing: list[str] = Field(..., description="Run UUIDs for which realised data could not be fetched.")


class HistoryEventStudyResponse(BaseModel):
    """Forward 10-trading-day price path anchored on the event date.

    Backs the event-study chart on /history/[id]: the realised close path
    plus the bucketed realised regime label so the headline can read
    "predicted X, realized Y".
    """

    event_date: str = Field(..., description="ISO `YYYY-MM-DD` event date the forward path is anchored to.")
    symbol: str = Field(..., description="Yahoo Finance ticker the forward series was pulled for.")
    forward_dates: list[str] = Field(..., description="ISO dates of the 10 forward trading sessions, oldest-first.")
    forward_close: list[float] = Field(..., description="Realised close prices aligned to `forward_dates`.")
    forward_log_returns: list[float] = Field(..., description="Bar-to-bar log returns aligned to `forward_dates`.")
    realized_vol_10d: float | None = Field(default=None, description="Realised 10-day forward volatility derived from `forward_log_returns`; null when insufficient bars.")
    predicted_regime: str | None = Field(default=None, description="Predicted regime label persisted with the original run; null when no regime card was persisted.")
    realized_regime: str | None = Field(default=None, description="Realised regime label bucketed from `realized_vol_10d`; null when cutoffs are unavailable.")


class EvaluationCoverageResponse(BaseModel):
    """Empirical conformal coverage aggregated across recent history runs.

    ``nominal`` is the conformal target the active model was calibrated
    to (read off the most-recent run that carries
    ``series.forecast_confidence_level``). ``empirical`` is the fraction
    of runs whose realized regime label fell inside the predicted set.
    Both are None when no qualifying runs exist."""

    model_config = _FORBID_FROZEN_CONFIG

    nominal: float | None = Field(default=None, description="Conformal target the active model was calibrated to; null when no recent run carries one.")
    empirical: float | None = Field(default=None, description="Empirical hit rate of the predicted set across qualifying runs; null when no runs qualify.")
    sample_size: int = Field(..., description="Number of qualifying runs the empirical coverage was computed over.")
    runs_total: int = Field(..., description="Total number of history runs considered before filtering.")
    computed_at: str = Field(..., description="ISO-8601 UTC timestamp the aggregate was computed at.")


class ClassificationBreakdownClass(BaseModel):
    """Per-class accuracy metrics for one class in the regime breakdown."""

    model_config = _FORBID_FROZEN_CONFIG

    class_id: int = Field(..., description="Integer class index in canonical regime ordering (0=calm, 1=normal, 2=high).")
    precision: float = Field(..., description="Precision for this class in `[0, 1]`: true positives / predicted positives.")
    recall: float = Field(..., description="Recall for this class in `[0, 1]`: true positives / actual positives.")
    f1: float = Field(..., description="F1 score for this class in `[0, 1]`: harmonic mean of precision and recall.")
    support: int = Field(..., description="Number of ground-truth samples in this class.")
    roc_auc: float | None = Field(default=None, description="One-vs-rest ROC AUC for this class; null when the source artifact omits it.")
    pr_auc: float | None = Field(default=None, description="One-vs-rest PR AUC for this class; null when the source artifact omits it.")


class ClassificationBreakdownSource(BaseModel):
    """Provenance metadata for the artifact backing a classification breakdown."""

    model_config = _FORBID_FROZEN_CONFIG

    relative_path: str = Field(..., description="Path of the source artifact relative to `data/artifacts/`.")
    training_package_id: str | None = Field(default=None, description="Training package ID the artifact was produced under; null when the field is absent on disk.")
    checkpoint_path: str | None = Field(default=None, description="Checkpoint path the breakdown was computed against; null when not recorded.")
    modified_at: str = Field(..., description="ISO-8601 mtime of the source artifact in UTC.")


class ClassificationBreakdownResponse(BaseModel):
    """The richer eval emitted by app/evaluation/classification_breakdown.py.

    Surfaces the freshest ``best_trial.summary.metrics.classification_breakdown``
    block written under ``data/artifacts/regime_*``. ``available`` is
    false when no qualifying artifact exists — the UI then falls back to
    its client-side aggregation across history rows."""

    model_config = _FORBID_FROZEN_CONFIG

    available: bool = Field(..., description="True when a qualifying artifact was found; false flags UI fallback to client-side aggregation.")
    confusion_matrix: list[list[int]] | None = Field(default=None, description="Row-actual, column-predicted confusion matrix in canonical class order; null when unavailable.")
    per_class: list[ClassificationBreakdownClass] | None = Field(default=None, description="Per-class precision/recall/F1/support entries; null when the artifact is missing.")
    macro_f1: float | None = Field(default=None, description="Unweighted mean F1 across classes in `[0, 1]`; null when unavailable.")
    macro_precision: float | None = Field(default=None, description="Unweighted mean precision across classes in `[0, 1]`; null when unavailable.")
    macro_recall: float | None = Field(default=None, description="Unweighted mean recall across classes in `[0, 1]`; null when unavailable.")
    macro_roc_auc: float | None = Field(default=None, description="Unweighted mean one-vs-rest ROC AUC across classes; null when unavailable.")
    macro_pr_auc: float | None = Field(default=None, description="Unweighted mean one-vs-rest PR AUC across classes; null when unavailable.")
    weighted_f1: float | None = Field(default=None, description="Support-weighted mean F1 across classes in `[0, 1]`; null when unavailable.")
    n_classes: int | None = Field(default=None, description="Number of classes in the breakdown; null when unavailable.")
    class_labels: list[str] | None = Field(
        default=None,
        description="Ordered class labels (e.g. calm/normal/high) from the source artifact; null when the artifact omits them.",
    )
    source: ClassificationBreakdownSource | None = Field(default=None, description="Provenance of the backing artifact; null when no artifact was found.")


class SymbolDescriptor(BaseModel):
    """One entry in the supported-symbols registry served to the asset picker."""

    model_config = _FORBID_FROZEN_CONFIG

    symbol: str = Field(..., description="Yahoo Finance ticker, e.g. `^GSPC` or `DX-Y.NYB`.")
    name: str = Field(..., description="Human-readable name of the instrument shown in the picker.")
    category: str = Field(..., description="Category bucket the symbol falls into, e.g. `equity_index`, `currency`, `rates`.")
    default_horizon: str = Field(..., description="Default forecast horizon label paired with this symbol on the workspace.")


class SymbolListResponse(BaseModel):
    """Response envelope for `GET /symbols`."""

    model_config = _FORBID_FROZEN_CONFIG

    symbols: list[SymbolDescriptor] = Field(..., description="Supported symbols ordered for display in the picker.")


class SettingsCheckpoint(BaseModel):
    """One file under ``backend/models/`` surfaced on the settings page.

    Inventory only; nothing here mutates the running singleton. ``role``
    is inferred from the filename (forecaster / multi_axis / lora /
    calibration) and ``is_active`` flags the file the active service is
    currently loaded from. The diagnostic fields (output_mode,
    encoder_alias, conformal_sidecar_present) only populate on the
    active forecaster + active multi-axis entries.

    #342: ``required_kwargs`` mirrors the inference-contract sidecar
    (empty list when no sidecar — pre-#341 legacy artefact).
    ``supplied_at_inference`` maps each declared kwarg to the live
    serving wiring; mismatches drive the red-badge surface on the
    settings page. ``inference_contract_status`` is ``"sidecar_absent"``
    for legacy checkpoints, otherwise ``"present"``.
    """

    model_config = _FORBID_FROZEN_CONFIG

    filename: str = Field(..., description="Bare filename of the checkpoint under `backend/models/`.")
    relative_path: str = Field(..., description="Path of the checkpoint relative to the models directory.")
    role: str = Field(..., description="Inferred role: forecaster / multi_axis / lora / calibration.")
    size_bytes: int = Field(..., description="File size in bytes as reported by the filesystem.")
    modified_at: str = Field(..., description="ISO-8601 UTC mtime of the checkpoint file.")
    is_active: bool = Field(default=False, description="True when the active service is currently loaded from this file.")
    output_mode: str | None = Field(default=None, description="Output mode declared by the active checkpoint (e.g. classification/regression/dual); null on inactive rows.")
    encoder_alias: str | None = Field(default=None, description="Registry alias of the encoder backing this checkpoint; null on inactive rows.")
    conformal_sidecar_present: bool | None = Field(default=None, description="True when a sibling conformal calibration sidecar is on disk; null on inactive rows.")
    required_kwargs: list[str] = Field(default_factory=list, description="Inference kwargs declared in the checkpoint's contract sidecar; empty for legacy artefacts.")
    supplied_at_inference: dict[str, bool] = Field(default_factory=dict, description="Mapping from declared kwarg name to whether the live serving wiring supplies it.")
    inference_contract_status: str | None = Field(default=None, description="`present` when a contract sidecar exists, `sidecar_absent` for legacy checkpoints.")
    source: str = Field(
        default="models_dir",
        description="Origin of the file on disk: `models_dir` for the host-mounted directory or `hf_cache` for HF snapshot cache.",
    )
    repo: str | None = Field(default=None, description="HF Hub `owner/name` slug when `source == hf_cache`; null otherwise.")
    revision: str | None = Field(default=None, description="Pinned HF commit hash when known; empty/null for unpinned artefacts.")
    snapshot_path: str | None = Field(default=None, description="Absolute path inside the HF cache the file resolves to; null when `source == models_dir`.")


class SettingsCheckpointsResponse(BaseModel):
    """Response envelope for the settings-page checkpoint inventory."""

    model_config = _FORBID_FROZEN_CONFIG

    models_dir: str = Field(..., description="Absolute path of the host-mounted `backend/models/` directory.")
    checkpoints: list[SettingsCheckpoint] = Field(..., description="One entry per checkpoint file discovered under the models directory or HF cache.")


class FomcMeetingResponse(BaseModel):
    """One FOMC meeting row in the calendar response."""

    meeting_date: str = Field(..., description="ISO `YYYY-MM-DD` date of the FOMC meeting.")
    meeting_type: str = Field(..., description="Meeting type label, e.g. `scheduled` or `unscheduled`.")
    statement_release_date: str | None = Field(default=None, description="ISO date the statement was released; null when not yet scheduled or recorded.")
    minutes_release_date: str | None = Field(default=None, description="ISO date the minutes were released; null when not yet released.")
    notes: str | None = Field(default=None, description="Free-text annotation on the meeting, e.g. inter-meeting context; null when absent.")
    statement_available: bool = Field(default=False, description="True when the statement body is available in the `/documents` cache.")
    minutes_available: bool = Field(default=False, description="True when the minutes body is available in the `/documents` cache.")
    press_conference_available: bool = Field(default=False, description="True when a press-conference transcript is available in the cache.")


class FomcCalendarResponse(BaseModel):
    """Past and upcoming FOMC meetings for the calendar panel."""

    past: list[FomcMeetingResponse] = Field(..., description="Past FOMC meetings ordered newest-first.")
    upcoming: list[FomcMeetingResponse] = Field(..., description="Scheduled future FOMC meetings ordered soonest-first.")


class DocumentParseUrlRequest(BaseModel):
    """Request body for `POST /documents/parse-url` — fetch and parse a remote PDF/HTML page."""

    url: str = Field(..., description="Remote URL of the FOMC document (PDF or HTML page) to fetch and parse.")


class DocumentParseResponse(BaseModel):
    """Parsed plain-text payload extracted from a fetched FOMC document."""

    text: str = Field(..., description="Extracted plain-text body after parsing the source document.")
    char_count: int = Field(..., description="Length of `text` in characters.")
    source_kind: str = Field(..., description="Detected source kind, e.g. `pdf`, `html`, `text`.")
    source_metadata: dict[str, str] = Field(..., description="Extracted metadata key/value pairs from the source, e.g. title or author.")


class DocumentDetailResponse(BaseModel):
    """Single FOMC document body served to the path-based
    ``/documents/{type}/{date}`` viewer. The text payload is the
    hygiene-cleaned body — boilerplate (Implementation Note, voting
    roster, navigation chrome) is stripped before it lands here so the
    frontend can render it as prose without further pre-processing."""

    type: str = Field(..., description="One of statement / minutes / press_conference.")
    date: str = Field(..., description="Event ISO date the document indexes against.")
    title: str = Field(..., description="Source title; empty when the JSON row omits it.")
    cleaned_text: str = Field(..., description="Body after text_hygiene.clean_fomc_text().")
    source_url: str | None = Field(
        default=None,
        description="Federal Reserve permalink when the row carries one.",
    )
    scraped_at: str | None = Field(
        default=None,
        description="ISO-8601 UTC timestamp the source row was scraped at.",
    )


# ---------------------------------------------------------------------------
# Research / training / decisions endpoints (Phase 8 multi-page expansion)
# ---------------------------------------------------------------------------


class ArtifactFile(BaseModel):
    """One file under ``data/artifacts/<section>/``."""

    model_config = _FORBID_FROZEN_CONFIG

    relative_path: str = Field(..., description="Path relative to data/artifacts/")
    size_bytes: int = Field(..., description="File size in bytes as reported by the filesystem.")
    modified_at: str = Field(..., description="ISO-8601 mtime, UTC.")
    suffix: str = Field(..., description="File extension including the dot, e.g. .json")


class EncoderBakeoffRow(BaseModel):
    """One encoder's seed-aggregated macro-F1 row in the bake-off table."""

    model_config = _FORBID_FROZEN_CONFIG

    encoder_key: str = Field(..., description="Registry alias of the encoder being compared.")
    checkpoint: str = Field(..., description="Checkpoint path or HF revision the bake-off was run against.")
    seeds: list[int] = Field(..., description="Seed set the bake-off was averaged across.")
    macro_f1_values: list[float] = Field(..., description="Per-seed macro-F1 values aligned to `seeds`.")
    macro_f1_mean: float = Field(..., description="Mean macro-F1 across `seeds` in `[0, 1]`.")
    macro_f1_ci_low: float | None = Field(default=None, description="Lower edge of the macro-F1 95% bootstrap CI; null when CI was not computed.")
    macro_f1_ci_high: float | None = Field(default=None, description="Upper edge of the macro-F1 95% bootstrap CI; null when CI was not computed.")
    weighted_f1_mean: float | None = Field(default=None, description="Support-weighted mean F1 across seeds; null when not recorded.")
    accuracy_mean: float | None = Field(default=None, description="Mean accuracy across seeds in `[0, 1]`; null when not recorded.")
    cohen_kappa: float | None = Field(default=None, description="Mean Cohen kappa across seeds; null when not recorded.")


class EncoderBakeoffSection(BaseModel):
    """Encoder bake-off table for the research artefacts response."""

    model_config = _FORBID_FROZEN_CONFIG

    available: bool = Field(..., description="True when at least one bake-off artifact was found on disk.")
    coverage: float | None = Field(default=None, description="Fraction of the official seed set covered by the rows in `[0, 1]`; null when unavailable.")
    rows: list[EncoderBakeoffRow] = Field(default_factory=list, description="Per-encoder bake-off rows ordered by descending macro-F1 mean.")
    source_files: list[str] = Field(default_factory=list, description="Relative paths of artefact files contributing to this section.")


class TransferMatrixCell(BaseModel):
    """One source -> target cell in the cross-bank transfer matrix."""

    model_config = _FORBID_FROZEN_CONFIG

    source: str = Field(..., description="Source bank short code the model was fine-tuned on.")
    target: str = Field(..., description="Target bank short code the model was evaluated on.")
    metric: float = Field(..., description="Transfer metric value (e.g. macro-F1) on `target` after training on `source`.")


class CrossBankTransferSection(BaseModel):
    """Cross-bank transfer matrix surfaced on the research page."""

    model_config = _FORBID_FROZEN_CONFIG

    available: bool = Field(..., description="True when at least one source/target cell was found on disk.")
    metric_name: str = Field(default="macro_f1", description="Metric carried in each cell, e.g. `macro_f1` or `accuracy`.")
    sources: list[str] = Field(default_factory=list, description="Distinct source banks present in `cells`, ordered for display.")
    targets: list[str] = Field(default_factory=list, description="Distinct target banks present in `cells`, ordered for display.")
    cells: list[TransferMatrixCell] = Field(default_factory=list, description="Flattened (source, target, metric) cells of the transfer matrix.")
    source_files: list[str] = Field(default_factory=list, description="Relative paths of artefact files contributing to this section.")


class EncoderAxisStanceRow(BaseModel):
    """One encoder's validity-study row on the stance axis."""

    model_config = _FORBID_FROZEN_CONFIG

    encoder_alias: str = Field(..., description="Registry alias of the encoder being compared.")
    encoder_display: str = Field(..., description="Display name of the encoder for UI rendering.")
    held_out_f1: float = Field(..., description="Held-out macro-F1 of the stance head fine-tuned on this encoder.")
    spearman_rho: float = Field(..., description="Spearman correlation between predicted stance scores and the labelled axis.")
    auc_hike_vs_cut: float = Field(..., description="AUC of stance scores separating hike vs cut meetings.")
    is_validity_winner: bool = Field(default=False, description="True when this encoder won the validity-study tie-break.")
    is_held_out_winner: bool = Field(default=False, description="True when this encoder won on the held-out macro-F1 metric.")


class EncoderAxisStanceSection(BaseModel):
    """Validity-study table for the stance axis on the research page."""

    model_config = _FORBID_FROZEN_CONFIG

    available: bool = Field(..., description="True when at least one validity-study row was found on disk.")
    rows: list[EncoderAxisStanceRow] = Field(default_factory=list, description="Per-encoder validity rows ordered by display priority.")
    source_doc: str = Field(default="", description="Citation of the source document the rows were lifted from.")


class ResearchArtifactsResponse(BaseModel):
    """Aggregate research-page envelope: artifact file index plus the
    bake-off / validity / cross-bank-transfer sections."""

    model_config = _FORBID_FROZEN_CONFIG

    artifacts_root: str = Field(..., description="Absolute path of the `data/artifacts/` root the listing was taken from.")
    sections: dict[str, list[ArtifactFile]] = Field(..., description="Mapping from section name to its artifact file index.")
    encoder_bakeoff: EncoderBakeoffSection = Field(..., description="Encoder bake-off table on the canonical seed set.")
    encoder_axis_stance: EncoderAxisStanceSection = Field(
        default_factory=lambda: EncoderAxisStanceSection(available=False, rows=[]),
        description="Validity-study table for the stance axis.",
    )
    cross_bank_transfer: CrossBankTransferSection = Field(..., description="Cross-bank transfer matrix across the six covered central banks.")


class NextFomcMeetingPrediction(BaseModel):
    """One next-FOMC ordinal prediction row keyed by model name."""

    model_config = _FORBID_FROZEN_CONFIG

    target_event_date: str = Field(..., description="ISO `YYYY-MM-DD` event date the prediction targets.")
    target_as_of_ts: str = Field(..., description="ISO-8601 UTC timestamp the features were anchored at.")
    target_class: str | None = Field(
        default=None,
        description="Realised class, when known. None for the next-scheduled meeting.",
    )
    n_train_rows: int = Field(..., description="Number of training rows that fed the model for this target.")
    probabilities: dict[str, dict[str, float]] = Field(..., description="Per-model dict of class -> probability for the target meeting.")
    predicted_class: dict[str, str] = Field(..., description="Per-model argmax class label drawn from `probabilities`.")


class NextFomcModelMetrics(BaseModel):
    """Aggregate accuracy metrics for one next-FOMC model on a window."""

    model_config = _FORBID_FROZEN_CONFIG

    n: int = Field(..., description="Number of resolved predictions in the window.")
    brier: float | None = Field(default=None, description="Multiclass Brier score over the window; null when unavailable.")
    log_loss: float | None = Field(default=None, description="Mean cross-entropy loss over the window; null when unavailable.")
    top1_accuracy: float | None = Field(default=None, description="Top-1 argmax accuracy in `[0, 1]`; null when unavailable.")
    macro_f1: float | None = Field(default=None, description="Unweighted mean F1 across classes in `[0, 1]`; null when unavailable.")
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict, description="Nested actual -> predicted counts over the window.")


class NextFomcAttributionRow(BaseModel):
    """One feature-family attribution row in the next-FOMC ablation table."""

    model_config = _FORBID_FROZEN_CONFIG

    subset: str = Field(..., description="Subset identifier the row was scored on (e.g. `full`, `ex_pandemic`).")
    families: list[str] = Field(..., description="Feature families included in this ablation row.")
    n_features: int | None = Field(default=None, description="Number of features used after the family subset was applied; null when unavailable.")
    n: int | None = Field(default=None, description="Number of resolved predictions used for the metrics; null when unavailable.")
    brier: float | None = Field(default=None, description="Multiclass Brier score for this row; null when unavailable.")
    log_loss: float | None = Field(default=None, description="Mean cross-entropy loss for this row; null when unavailable.")
    top1_accuracy: float | None = Field(default=None, description="Top-1 argmax accuracy in `[0, 1]`; null when unavailable.")
    macro_f1: float | None = Field(default=None, description="Unweighted mean F1 across classes in `[0, 1]`; null when unavailable.")


class NextFomcUpcomingMeeting(BaseModel):
    """Lightweight descriptor of the upcoming meeting the headline targets."""

    model_config = _FORBID_FROZEN_CONFIG

    meeting_date: str = Field(..., description="ISO `YYYY-MM-DD` date of the next FOMC meeting.")
    meeting_type: str = Field(..., description="Meeting type label, e.g. `scheduled`.")
    statement_release_date: str | None = Field(default=None, description="ISO date the statement is expected; null when not yet scheduled.")
    days_until: int | None = Field(default=None, description="Calendar days from today to `meeting_date`; null when unavailable.")


class NextFomcForecastResponse(BaseModel):
    """Response envelope for `GET /forecast/next-fomc`."""

    model_config = _FORBID_FROZEN_CONFIG

    available: bool = Field(..., description="True when a next-FOMC artifact was found on disk.")
    artifacts_dir: str = Field(..., description="Absolute path of the artifacts directory backing this response.")
    ordinal_classes: list[str] = Field(..., description="Ordered class labels the next-FOMC heads emit, e.g. cut/hold/hike.")
    model_names: list[str] = Field(default_factory=list, description="Model identifiers compared in this response.")
    upcoming_meeting: NextFomcUpcomingMeeting | None = Field(default=None, description="Descriptor of the upcoming meeting the headline targets; null when none is scheduled.")
    headline: NextFomcMeetingPrediction | None = Field(default=None, description="Headline next-meeting prediction row; null when no prediction is available.")
    history: list[NextFomcMeetingPrediction] = Field(default_factory=list, description="Historical next-meeting predictions ordered oldest-first.")
    metrics_full_window: dict[str, NextFomcModelMetrics] = Field(default_factory=dict, description="Per-model aggregate metrics over the full window keyed by model name.")
    metrics_ex_pandemic: dict[str, NextFomcModelMetrics] = Field(default_factory=dict, description="Per-model aggregate metrics excluding the COVID pandemic window.")
    feature_attribution: list[NextFomcAttributionRow] = Field(default_factory=list, description="Feature-family ablation rows for the model attribution table.")
    summary: dict[str, int] = Field(default_factory=dict, description="Free-form integer summary counters surfaced beside the response.")


# ---------------------------------------------------------------------------
# Historical analog retrieval (#294 — /analyze/analogs)
# ---------------------------------------------------------------------------


class AnalogsRequest(BaseModel):
    """Query a fine-tuned retrieval encoder for past FOMC statements that
    sound like ``text``. Returns top-``k`` analogs ordered by cosine
    similarity together with a coarse post-event volatility regime label
    so the caller can show "what regime followed each analog" without
    exposing the underlying supervised target value."""

    model_config = _STRICT_REQUEST_CONFIG

    text: str = Field(
        ..., min_length=1, description="Statement text to match against past FOMC statements."
    )
    k: int = Field(default=5, ge=1, le=20, description="Number of analogs to return (1-20).")
    as_of_date: date | None = Field(
        default=None,
        description=(
            "Strict-backward walk-forward boundary: only analogs with "
            "``event_date < as_of_date`` are eligible. Default is no "
            "boundary; for ML feature use always pass a date so the "
            "retrieval set does not leak future statements into the "
            "training fold."
        ),
    )

    @field_validator("as_of_date", mode="before")
    @classmethod
    def _coerce_as_of_date(cls, value: Any) -> Any:
        """Accept ISO ``YYYY-MM-DD`` strings under strict mode.

        Pydantic strict mode rejects string -> date coercion by default,
        but the JSON wire format only carries strings; this validator
        parses the ISO form so callers do not need to pre-coerce the
        value before posting.
        """

        if value is None or isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"as_of_date must be ISO YYYY-MM-DD, got {value!r}") from exc
        return value


class AnalogCard(BaseModel):
    """One historical analog returned by /analyze/analogs.

    The card deliberately does NOT carry the raw
    ``forward_realized_vol_10d`` target the project trains on — that
    value is the supervised label and surfacing it as an API field
    would tempt downstream consumers to feed it back into a model.
    Instead the card exposes ``subsequent_vol_regime``, a coarse
    ``calm`` / ``normal`` / ``high`` bucket pinned to the
    ``VOL_REGIME_BUCKET_EDGES`` constant in ``app.retrieval.index``
    (held-out 2000-2015 reference distribution). The bucket label is a
    UI-only marker; treat it as informative for humans, not as a
    feature for a downstream model.
    """

    model_config = _FORBID_FROZEN_CONFIG

    event_date: str = Field(..., description="ISO date of the historical FOMC statement.")
    similarity: float = Field(
        ..., description="Cosine similarity in [-1, 1] vs. the query embedding."
    )
    axis_stance: str | None = Field(
        default=None,
        description="Stored stance label for the analog statement (hawkish / dovish / neutral). None when absent.",
    )
    subsequent_vol_regime: Literal["calm", "normal", "high"] | None = Field(
        default=None,
        description=(
            "Coarse post-event 10-day realised vol bucket — UI-only label, "
            "NOT a model feature. Do not feed back into a downstream model."
        ),
    )
    # #299 — realized S&P forward returns measured from the event-day
    # close (Bloomberg / FactSet convention). The denominator is the
    # close on ``event_date`` (or the nearest prior trading day) and
    # the numerator is the close ``N`` trading days forward of that
    # anchor. These are MARKET DATA OVERLAYS (yfinance ^GSPC), not
    # training labels — safe to surface even when the supervised
    # ``subsequent_vol_regime`` bucket is intentionally suppressed.
    # None when the historical market data is unavailable.
    subsequent_close_pct_5d: float | None = Field(
        default=None,
        description="S&P 500 close-to-close % return over the 5 trading days following the event-day close.",
    )
    subsequent_close_pct_20d: float | None = Field(
        default=None,
        description="S&P 500 close-to-close % return over the 20 trading days following the event-day close.",
    )
    excerpt: str = Field(..., description="First ~280 characters of the analog statement.")


class AnalogsResponse(BaseModel):
    """Result envelope for /analyze/analogs."""

    model_config = _FORBID_FROZEN_CONFIG

    analogs: list[AnalogCard] = Field(default_factory=list)
    index_size: int = Field(
        ..., description="Total number of past statements in the loaded retrieval index."
    )
    encoder_alias: str = Field(
        ..., description="Registry alias of the encoder used to embed the query."
    )


# ---------------------------------------------------------------------------
# Hawkish/dovish trajectory model (#296 — /analyze/trajectory)
# ---------------------------------------------------------------------------


class TrajectoryRequest(BaseModel):
    """Project the FOMC stance trajectory as of ``as_of_date``.

    Strict-backward by construction: the history slice only considers
    meetings whose ``event_date <= as_of_date``. ``history_length``
    caps the number of recent meetings rendered in the panel chart.
    Defaults to 12 (~1.5 years of FOMC meetings) per the §3 Panel 4
    spec.
    """

    model_config = _STRICT_REQUEST_CONFIG

    as_of_date: date = Field(
        ..., description="As-of date for the trajectory projection (YYYY-MM-DD)."
    )
    history_length: int = Field(
        default=12,
        ge=1,
        le=60,
        description="Number of past meetings to surface in the panel (1-60).",
    )

    @field_validator("as_of_date", mode="before")
    @classmethod
    def _coerce_as_of_date(cls, value: Any) -> Any:
        if value is None or isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"as_of_date must be ISO YYYY-MM-DD, got {value!r}") from exc
        return value


class TrajectoryMarker(BaseModel):
    """One past FOMC meeting rendered as a semantic marker on the panel chart."""

    model_config = _FORBID_FROZEN_CONFIG

    event_date: str = Field(..., description="ISO date of the historical FOMC meeting.")
    axis_stance: str | None = Field(
        default=None,
        description=(
            "Stored stance label (hawkish / dovish / neutral) for the meeting. "
            "None when the panel was trained against a corpus without labels."
        ),
    )
    embedding_2d: tuple[float, float] = Field(
        ...,
        description=(
            "PCA / UMAP projection of the meeting's encoder embedding to "
            "2-D space, used as the marker's (x, y) coordinates."
        ),
    )


class TrajectoryProjection(BaseModel):
    """Next-meeting projection — predicted class + calibrated confidence band."""

    model_config = _FORBID_FROZEN_CONFIG

    predicted_stance: str = Field(
        ..., description="Argmax over the next-meeting stance distribution."
    )
    class_probs: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-class probability over hawkish / dovish / neutral. Sums "
            "to 1.0 after defensive renormalisation."
        ),
    )
    confidence_band: list[str] | None = Field(
        default=None,
        description=(
            "APS-calibrated stance set (Romano et al. 2020). Empty when no "
            "conformal sidecar shipped with the bundle; the UI then "
            "renders the argmax marker without the confidence ring."
        ),
    )
    conformal_alpha: float | None = Field(
        default=None,
        description="Conformal mis-coverage level applied to confidence_band.",
    )


class TrajectoryResponse(BaseModel):
    """Result envelope for /analyze/trajectory."""

    model_config = _FORBID_FROZEN_CONFIG

    available: bool = Field(
        ..., description="False when no trajectory bundle is loaded on this host."
    )
    history: list[TrajectoryMarker] = Field(default_factory=list)
    projected_next: TrajectoryProjection | None = None
    architecture: str | None = Field(
        default=None,
        description="lstm | transformer — which arm produced the projection.",
    )
    encoder_alias: str = Field(
        default="",
        description="Registry alias of the encoder backing the meeting embeddings.",
    )
    history_length: int = Field(
        default=0,
        description="How many past meetings were considered (after truncation).",
    )
    train_end: str | None = Field(
        default=None,
        description="Walk-forward boundary from the bundle manifest.",
    )
    as_of_date: str = Field(
        default="",
        description="Echo of the request as_of_date (ISO YYYY-MM-DD).",
    )
    warning: str | None = Field(
        default=None,
        description=(
            "Optional non-fatal advisory — set when ``as_of_date`` is "
            "beyond the bundle's ``train_end`` so the caller can flag "
            "that the projection extrapolates beyond the fold."
        ),
    )
    # Lift-vs-baseline badge fields (#332). Surface the verdict on
    # whether the Transformer arm beats the strongest naive baseline
    # (previous_stance / rolling_majority / small-LSTM) by >= 5pp
    # directional accuracy on the canonical fold protocol. All three
    # fields default to None / False so a bundle trained before #332
    # remains back-compatible.
    lift_vs_baseline: bool = Field(
        default=False,
        description=(
            "True iff the Transformer arm beats the strongest naive "
            "baseline (previous_stance / rolling_majority(3) / "
            "small-LSTM) by >= 5pp directional accuracy on the "
            "canonical fold protocol. False when the holdout slice is "
            "empty, when the bundle predates #332, or when the lift "
            "did not clear the threshold."
        ),
    )
    delta_dir_acc: float | None = Field(
        default=None,
        description=(
            "Transformer directional accuracy minus the strongest "
            "naive baseline's directional accuracy. None when no "
            "baseline comparison is available."
        ),
    )
    baseline_used: str | None = Field(
        default=None,
        description=(
            "Name of the strongest naive baseline the lift verdict "
            "compared against. None when no baseline comparison is "
            "available."
        ),
    )


class ResearchRegistryBaseline(BaseModel):
    """No-text baseline reference row for the encoder registry."""

    model_config = _FORBID_FROZEN_CONFIG

    label: str = Field(..., description="Label of the no-text baseline, e.g. `market_only`.")
    dual_f1: float | None = Field(default=None, description="Baseline macro-F1 on the dual-head surface; null when not recorded.")
    cls_f1: float | None = Field(default=None, description="Baseline macro-F1 on the classification-only surface; null when not recorded.")
    regression_f1: float | None = Field(default=None, description="Baseline macro-F1 on the regression-only surface; null when not recorded.")


class ResearchRegistryRow(BaseModel):
    """One encoder's row in the registry table comparing it to the no-text baseline."""

    model_config = _FORBID_FROZEN_CONFIG

    encoder_alias: str = Field(..., description="Registry alias of the encoder being compared.")
    encoder_display: str = Field(..., description="Display name of the encoder for UI rendering.")
    dual_f1: float | None = Field(default=None, description="Encoder macro-F1 on the dual-head surface; null when not recorded.")
    cls_f1: float | None = Field(default=None, description="Encoder macro-F1 on the classification-only surface; null when not recorded.")
    regression_f1: float | None = Field(default=None, description="Encoder macro-F1 on the regression-only surface; null when not recorded.")
    delta_dual: float | None = Field(
        default=None,
        description="Δ macro-F1 on the dual-head surface vs no-text baseline.",
    )
    delta_cls: float | None = Field(
        default=None,
        description="Δ macro-F1 on the classification-only surface vs no-text baseline.",
    )
    is_winner: bool = Field(
        default=False,
        description="True iff the active surfaces Δ is >= 0 (non-negative lift).",
    )
    checkpoint_relpath: str | None = Field(default=None, description="Path of the encoder checkpoint relative to the models directory; null when not published.")
    cache_uri: str | None = Field(
        default=None,
        description="hf:// URI of the shareable embedding cache parquet, if published.",
    )
    notes: str = Field(default="", description="Free-text notes for the row, e.g. caveats; empty when none.")


class ResearchRegistryResponse(BaseModel):
    """Quant-facing encoder registry response (§6.41 manifest).

    Filtered by default to non-negative Δ on the requested surface so
    the dashboard does not surface negative-lift encoders. Use
    ?include_rejected=true to see the full table including nulls and
    negatives.
    """

    model_config = _FORBID_FROZEN_CONFIG

    available: bool = Field(..., description="True when a registry manifest was found on disk.")
    surface: Literal["dual", "cls"] = Field(..., description="Which head surface the lift is computed on: `dual` or `cls`.")
    baseline: ResearchRegistryBaseline | None = Field(default=None, description="Reference no-text baseline row; null when not recorded.")
    rows: list[ResearchRegistryRow] = Field(default_factory=list, description="Per-encoder registry rows after filtering on lift sign.")
    rejected_count: int = Field(default=0, description="Number of rows filtered out for negative or null lift.")
    training_package_id: str = Field(default="", description="Training package ID the registry was assembled from.")
    head: str = Field(default="", description="Head identifier within the package the rows were computed on.")
    seeds: list[int] = Field(default_factory=list, description="Seed set the registry rows were averaged across.")
    source_wiki_section: str = Field(default="", description="Wiki section identifier the registry sources its baseline from.")


# #299 PR-B — stance-directional backtest engine


class BacktestPositionEntry(BaseModel):
    """One {date, position} signal in the backtest request."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    date: str = Field(..., description="ISO date YYYY-MM-DD of the signal.")
    position: int = Field(
        ..., description="Position in {-1, 0, 1}. Hawkish=-1, neutral=0, dovish=+1."
    )


class BacktestRequest(BaseModel):
    """Request body for POST /research/backtest."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    positions: list[BacktestPositionEntry] = Field(
        ..., min_length=1, description="At least one signal entry."
    )
    symbol: str = Field("^GSPC", description="Market ticker for the strategy backtest.")
    horizon_days: int = Field(
        5,
        ge=1,
        le=60,
        description="Forward holding period in trading days.",
    )


class BacktestTradeRow(BaseModel):
    """One realised trade row from the stance-directional backtest."""

    model_config = _FORBID_FROZEN_CONFIG

    date: str = Field(..., description="ISO `YYYY-MM-DD` date of the signal entry.")
    position: int = Field(..., description="Position taken on `date`: -1 short, 0 flat, +1 long.")
    forward_return_pct: float | None = Field(default=None, description="Forward holding-period close-to-close % return from `date`; null when forward bars are missing.")
    strategy_return_pct: float | None = Field(default=None, description="Strategy % return for the trade = `position * forward_return_pct`; null when the trade is unresolved.")


class BacktestResponse(BaseModel):
    """Aggregate backtest metrics for the quant terminal."""

    model_config = _FORBID_FROZEN_CONFIG

    trades: list[BacktestTradeRow] = Field(default_factory=list, description="Per-signal trade rows produced by the backtest.")
    n_trades: int = Field(..., description="Number of trades executed in the backtest.")
    sharpe: float | None = Field(default=None, description="Annualised Sharpe ratio of the strategy returns; null when fewer than two resolved trades.")
    hit_rate: float | None = Field(default=None, description="Fraction of resolved trades with positive strategy return in `[0, 1]`; null when none resolved.")
    max_dd_pct: float | None = Field(default=None, description="Maximum % drawdown of the cumulative strategy curve; null when unavailable.")
    cum_return_pct: float | None = Field(default=None, description="Cumulative % return of the strategy across all resolved trades; null when none resolved.")
    benchmark_cum_pct: float | None = Field(default=None, description="Cumulative % return of buy-and-hold on `symbol` over the same window; null when unavailable.")
    alpha_cum_pct: float | None = Field(default=None, description="Strategy cumulative return minus benchmark cumulative return; null when either side is unavailable.")
    horizon_days: int = Field(..., description="Forward holding period in trading days echoed from the request.")
    symbol: str = Field(..., description="Yahoo Finance ticker the backtest was run against.")


class RealizedVolHorizonForecast(BaseModel):
    """Banded RV forecast for one horizon (1, 5, or 22 trading days).

    ``point`` and the four ``band_*`` numbers are RV (variance) units, not
    log-RV. ``qlike_model`` / ``qlike_har`` are the pooled walk-forward
    QLIKE losses (lower is better); the card surfaces the gain as a
    beat-HAR badge. ``coverage_empirical_90`` is the prospective empirical
    coverage of the 90% conformal band, for the calibration chip.
    """

    model_config = _FORBID_FROZEN_CONFIG

    h: int = Field(..., description="Forecast horizon in trading days, e.g. 1, 5, or 22.")
    point: float = Field(..., description="Point forecast of realized variance at horizon h.")
    band_lo_80: float = Field(..., description="Lower edge of the 80% conformal band on `point` in RV units.")
    band_hi_80: float = Field(..., description="Upper edge of the 80% conformal band on `point` in RV units.")
    band_lo_90: float = Field(..., description="Lower edge of the 90% conformal band on `point` in RV units.")
    band_hi_90: float = Field(..., description="Upper edge of the 90% conformal band on `point` in RV units.")
    qlike_model: float | None = Field(default=None, description="Pooled walk-forward QLIKE loss for the model at this horizon (lower is better); null when unavailable.")
    qlike_har: float | None = Field(default=None, description="Pooled walk-forward QLIKE loss for HAR-OLS at this horizon (baseline); null when unavailable.")
    coverage_empirical_90: float | None = Field(default=None, description="Prospective empirical coverage of the 90% band in `[0, 1]`; null when unavailable.")


class RealizedVolHistoricalBand(BaseModel):
    """Single walk-forward h=1 conformal band aligned to a realized day.

    Renders behind the realized sparkline so the card shows the band
    actually covered each day's outcome.
    """

    model_config = _FORBID_FROZEN_CONFIG

    date: str = Field(..., description="ISO `YYYY-MM-DD` of the trading session this band targeted.")
    band_lo_80: float = Field(..., description="Lower edge of the 80% conformal band at this session in RV units.")
    band_hi_80: float = Field(..., description="Upper edge of the 80% conformal band at this session in RV units.")
    realized_rv: float | None = Field(default=None, description="Realised RV on this session; null when the bar is unavailable.")


class RealizedVolForecastResponse(BaseModel):
    """Multi-horizon QLIKE-DLq forecast plus last-60d realized history."""

    model_config = _FORBID_FROZEN_CONFIG

    symbol: str = Field(..., description="Yahoo Finance ticker the RV forecast was produced for.")
    horizons: list[RealizedVolHorizonForecast] = Field(..., description="Per-horizon point + conformal-band forecasts ordered ascending by `h`.")
    history: list[float] = Field(default_factory=list, description="Last-60d realised RV history aligned to `history_dates`.")
    history_dates: list[str] = Field(default_factory=list, description="ISO dates aligned to `history` entries, oldest-first.")
    model_revision: str = Field(..., description="Model revision identifier the forecast was produced under.")
    historical_bands: list[RealizedVolHistoricalBand] | None = Field(default=None, description="Per-day walk-forward h=1 conformal bands for the sparkline; null when unavailable.")
    realized_features_source: str = Field(
        default="training_means",
        description="`live` when intraday-derived measures filled the QLIKE head; `training_means` when the head fell back to feat_mean.",
    )
    realized_features_date: str | None = Field(
        default=None,
        description="ISO date of the most-recent intraday session the live measures were reduced from; null when the head fell back to training_means.",
    )


class HarTercileHorizon(BaseModel):
    """HAR-tercile baseline classification for one forecast horizon.

    ``predicted_rv`` is HAR's OLS point in realized-variance units.
    ``tercile`` is the argmax bucket against the q33 / q67 cutoffs the
    response also returns; ``tercile_probs`` is the Gaussian-CDF mass
    triple in log-RV space (sums to 1.0). ``macro_f1`` is wired through
    from wiki section 20 — the published HAR-tercile pooled macro-F1 on
    the canonical 5-fold expanding walk-forward (0.687 / 0.685 / 0.654
    at h=1 / 5 / 22). The serving path does not recompute this number;
    it surfaces the offline-measured one so the frontend can render an
    honest chip alongside the live point forecast.
    """

    model_config = _FORBID_FROZEN_CONFIG

    h: int = Field(..., description="Forecast horizon in trading days, e.g. 1, 5, or 22.")
    predicted_rv: float = Field(..., description="HAR-OLS point forecast of realized variance at horizon h.")
    tercile: Literal["low", "medium", "high"] = Field(..., description="Argmax tercile bucket for `predicted_rv` against the q33/q67 cutoffs.")
    tercile_probs: dict[str, float] = Field(
        default_factory=dict,
        description="Per-class probability over (low, medium, high). Sums to 1.0.",
    )
    macro_f1: float | None = Field(
        default=None,
        description=(
            "Pooled macro-F1 for the HAR-tercile baseline at this horizon, "
            "read off wiki section 20 (Gated_Fusion_InfoNCE_Comprehensive_Null, "
            "Result 2) for ^GSPC. Null for non-canonical symbols (^NDX / ^DJI) "
            "where the baseline macro-F1 is not pinned and the response carries "
            "a per-call OLS HAR fit only."
        ),
    )
    macro_f1_source: str = Field(
        ...,
        description="Citation for the macro_f1 number (wiki section + result block).",
    )
    qlike_model: float | None = Field(
        default=None,
        description=(
            "Pooled walk-forward QLIKE loss for the QLIKE-DLq ensemble at "
            "this horizon (lower is better). None when the eval sidecar is "
            "missing."
        ),
    )
    qlike_har: float | None = Field(
        default=None,
        description="Pooled walk-forward QLIKE loss for HAR-OLS at this horizon.",
    )


class HarTercileBaselineResponse(BaseModel):
    """Multi-horizon HAR-tercile regime baseline for the headline regime card.

    Per wiki section 20, HAR-tercile is the strongest forward-vol-regime
    classifier on the canonical fold protocol (beats market-only and the
    text+market fused arm at h=1 and h=22; null at h=5). The frontend
    renders this response as the headline regime card and demotes the
    text+market fused card to a "second opinion" with explicit
    weaker-baseline disclosure.
    """

    model_config = _FORBID_FROZEN_CONFIG

    symbol: str = Field(..., description="Yahoo Finance ticker the HAR-tercile baseline was produced for.")
    horizons: list[HarTercileHorizon] = Field(..., description="Per-horizon HAR-tercile rows ordered ascending by `h`.")
    cutoffs_q33: float = Field(
        ...,
        description=(
            "Lower tercile cutoff on realized variance used to bucket the "
            "HAR point forecast. Derived from the supplied RV history when "
            "the artifact does not pin a per-horizon train-slice cutoff."
        ),
    )
    cutoffs_q67: float = Field(
        ...,
        description="Upper tercile cutoff on realized variance.",
    )
    model_revision: str = Field(..., description="Model revision identifier the baseline was produced under.")
    generated_at: str = Field(..., description="ISO-8601 UTC timestamp the response was generated at.")


# Workspace-spine bundle: shared response models for the four
# feature steps that build on top of this foundation. The feature
# steps fill the service-layer builders that populate these
# numbers; this module only owns the wire shape so the frontend
# types and the OpenAPI snapshot can settle ahead of the wiring.
#
# SPINE separation: ExpectedVolumeHorizonForecast /
# ExpectedVolumeForecastResponse are the only forecast surface in
# this bundle (market data only). MonetaryPolicySurpriseResponse,
# FuturesConsensusResponse and SemanticDiffResponse are descriptive
# panels (text- or realized-derived) and never feed forecasts.
class ExpectedVolumeHorizonForecast(BaseModel):
    """HAR-based forecast of expected log-residual trading volume for
    one horizon. ``point_log_residual`` is the model's point estimate
    in log-volume residual space (after calendar adjustment when the
    flag is set); ``point_pct_vs_baseline`` is the same number
    expressed as a % deviation from the rolling calendar-adjusted
    baseline so the card can render a human-readable headline.
    ``r2_har`` is the offline pooled walk-forward R^2 of the HAR
    volume head at this horizon, surfaced for the calibration chip.
    """

    model_config = _FORBID_FROZEN_CONFIG

    h: int = Field(..., description="Forecast horizon in trading days, e.g. 1, 5, or 22.")
    point_log_residual: float = Field(..., description="Point estimate in log-volume residual space after optional calendar adjustment.")
    point_pct_vs_baseline: float = Field(..., description="Point estimate as a % deviation from the rolling calendar-adjusted baseline.")
    band_lo_80: float = Field(..., description="Lower edge of the 80% conformal band on `point_log_residual`.")
    band_hi_80: float = Field(..., description="Upper edge of the 80% conformal band on `point_log_residual`.")
    band_lo_90: float = Field(..., description="Lower edge of the 90% conformal band on `point_log_residual`.")
    band_hi_90: float = Field(..., description="Upper edge of the 90% conformal band on `point_log_residual`.")
    r2_har: float | None = Field(default=None, description="Pooled walk-forward R^2 of the HAR volume head at this horizon; null when unavailable.")
    calendar_adjusted: bool = Field(..., description="True when the baseline subtraction used a calendar-adjusted rolling mean.")


class ExpectedVolumeForecastResponse(BaseModel):
    """Multi-horizon HAR-volume forecast for the Expected Volume card.
    Market-data-only forecast; never wired to text features.
    """

    model_config = _FORBID_FROZEN_CONFIG

    symbol: str = Field(..., description="Yahoo Finance ticker the volume forecast was produced for.")
    horizons: list[ExpectedVolumeHorizonForecast] = Field(..., description="Per-horizon volume forecast rows ordered ascending by `h`.")
    model_revision: str = Field(..., description="Model revision identifier the forecast was produced under.")
    generated_at: str = Field(..., description="ISO-8601 UTC timestamp the response was generated at.")


class MonetaryPolicySurpriseResponse(BaseModel):
    """Monetary-policy surprise (descriptive panel, not a forecast input).

    ``mp_surprise_level_bps`` is the realized rate-path surprise in
    basis points relative to the pre-meeting fed-funds futures
    consensus. ``direction`` is the discrete sign bucket the panel
    renders; ``no_surprise`` covers the inside-the-band cases.
    ``is_intermeeting`` flags off-cycle actions where the consensus
    baseline is constructed differently.
    """

    model_config = _FORBID_FROZEN_CONFIG

    event_date: str = Field(..., description="ISO `YYYY-MM-DD` of the FOMC event the surprise was measured at.")
    mp_surprise_level_bps: float = Field(..., description="Realised rate-path surprise vs the pre-meeting fed-funds futures consensus, in basis points (signed).")
    direction: Literal["hawkish", "dovish", "no_surprise"] = Field(..., description="Discrete sign bucket of the surprise the panel renders.")
    magnitude_bps: float = Field(..., description="Absolute magnitude of the surprise in basis points (non-negative).")
    is_intermeeting: bool = Field(..., description="True when the action was off-cycle and the consensus baseline was constructed differently.")
    ff_target_prior_bps: float | None = Field(default=None, description="Prior fed-funds target midpoint in basis points; null when unavailable.")


class FuturesConsensusHorizon(BaseModel):
    """One horizon of the fed-funds futures implied-path consensus.

    Probabilities are derived from the implied-rate distribution and
    bucketed against the current target band; they sum to 1.0 across
    hike / cut / pause.
    """

    model_config = _FORBID_FROZEN_CONFIG

    horizon_label: str = Field(..., description="Horizon identifier for the futures contract, e.g. `next` or `+3m`.")
    implied_rate_bps: float = Field(..., description="Futures-implied fed-funds rate at this horizon in basis points.")
    change_vs_current_bps: float = Field(..., description="Signed change in basis points vs the current target midpoint.")
    probability_hike: float = Field(..., description="Implied probability of a rate hike at this horizon in `[0, 1]`.")
    probability_cut: float = Field(..., description="Implied probability of a rate cut at this horizon in `[0, 1]`.")
    probability_pause: float = Field(..., description="Implied probability of a hold at this horizon in `[0, 1]`.")


class FuturesConsensusResponse(BaseModel):
    """FRED / CME-derived futures consensus panel.

    Descriptive only — the rate-path expectations chart reads off
    realized futures prices and never feeds the forecast cards.
    """

    model_config = _FORBID_FROZEN_CONFIG

    meeting_date: str = Field(..., description="ISO `YYYY-MM-DD` of the FOMC meeting the consensus targets.")
    generated_at: str = Field(..., description="ISO-8601 UTC timestamp the response was generated at.")
    current_target_lo_bps: float = Field(..., description="Lower bound of the current fed-funds target range in basis points.")
    current_target_hi_bps: float = Field(..., description="Upper bound of the current fed-funds target range in basis points.")
    horizons: list[FuturesConsensusHorizon] = Field(..., description="Per-horizon implied-rate + bucket-probability rows ordered by horizon.")
    methodology: str = Field(..., description="Short description of how the implied probabilities were derived.")
    data_source: str = Field(..., description="Upstream data source identifier, e.g. `FRED` or `CME`.")


class SemanticDiffSpan(BaseModel):
    """One token-aligned span of the current-vs-prior statement diff.

    ``kind`` is the alignment bucket; ``paired_text`` carries the
    matched span on the opposite side for ``substituted`` (and
    optionally for ``added`` / ``removed`` if the aligner emitted a
    near-match neighbour).
    """

    model_config = _FORBID_FROZEN_CONFIG

    kind: Literal["unchanged", "added", "removed", "substituted"] = Field(..., description="Alignment bucket of this span vs the prior statement.")
    text: str = Field(..., description="Surface text of the span on the current-statement side.")
    paired_text: str | None = Field(default=None, description="Matched span on the prior-statement side for `substituted` kinds; null when no near-match neighbour was emitted.")


class SemanticDiffTopic(BaseModel):
    """Topic-level emphasis delta across the two statements.

    ``prior_emphasis`` and ``current_emphasis`` are the topic-share
    masses in [0, 1]; ``delta`` = current - prior. ``sample_phrases``
    are the highest-loading n-grams the panel surfaces alongside the
    bar.
    """

    model_config = _FORBID_FROZEN_CONFIG

    topic: str = Field(..., description="Topic identifier from the topic model used for the diff.")
    prior_emphasis: float = Field(..., description="Topic-share mass in the prior statement in `[0, 1]`.")
    current_emphasis: float = Field(..., description="Topic-share mass in the current statement in `[0, 1]`.")
    delta: float = Field(..., description="Signed delta = current_emphasis - prior_emphasis.")
    sample_phrases: list[str] = Field(default_factory=list, description="Highest-loading n-grams the panel surfaces alongside the topic bar.")


class SemanticDiffRequest(BaseModel):
    """Inbound body for ``POST /fomc/semantic-diff``.

    ``current_date`` selects the strict-prior FOMC statement off disk;
    ``current_text`` is the pasted body the panel diffs against that
    prior. Both fields are required — the cold-start case is still a
    valid call, the service just returns an empty span list when no
    strict-prior is on file for the supplied date.
    """

    model_config = _STRICT_REQUEST_CONFIG

    current_date: str = Field(..., description="Document date in ISO format: YYYY-MM-DD")
    # No ``min_length`` guard: empty / whitespace-only / non-Latin bodies
    # must still reach :func:`app.services.semantic_diff.build_response`,
    # which returns a parseable degraded response with a ``status`` field
    # instead of raising. See SemanticDiffResponse.status.
    current_text: str = Field(..., description="FOMC statement text to diff")


class SemanticDiffResponse(BaseModel):
    """Semantic diff between the current statement and its prior.

    Descriptive panel — the spans and topic deltas are post-hoc
    explanations of the realized text change and never feed the
    forecast surface.

    ``status`` carries a parseable signal for the panel to surface
    an informational banner when the service could not produce a
    meaningful diff (empty / near-empty / non-Latin input, or no
    strict-prior on file). The field is optional and defaults to
    ``None`` for backward compatibility — older clients that ignore
    it still see the same empty-list cold-start shape they used to.
    """

    model_config = _FORBID_FROZEN_CONFIG

    current_date: str = Field(..., description="ISO `YYYY-MM-DD` event date of the current statement.")
    prior_date: str = Field(..., description="ISO `YYYY-MM-DD` event date of the strict-prior statement on file; empty when none.")
    token_spans: list[SemanticDiffSpan] = Field(..., description="Token-aligned span sequence between the two statements in current-statement order.")
    topic_deltas: list[SemanticDiffTopic] = Field(..., description="Topic-emphasis delta rows ordered by absolute delta magnitude.")
    summary: str = Field(..., description="Short human-readable summary of the diff, e.g. headline phrase changes.")
    status: Literal["ok", "no_input", "no_prior", "non_english"] | None = Field(
        default=None,
        description="Parseable status flag for the panel banner; null on legacy responses that pre-date the field.",
    )


class HarTercileBacktestRow(BaseModel):
    """One resolved (or pending) row in the HAR-tercile backtest table.

    A row carries the HAR-tercile prediction computed on the rolling
    RV history available at the FOMC event date plus, when the forward
    window has elapsed, the realized tercile bucketed off the same
    cutoffs that produced the prediction. ``correct`` is None for rows
    whose forward window has not yet closed.
    """

    model_config = _FORBID_FROZEN_CONFIG

    event_date: str = Field(..., description="ISO `YYYY-MM-DD` event date the row was anchored to.")
    predicted_tercile: str | None = Field(default=None, description="HAR-tercile predicted bucket (low/medium/high) at the event; null when the forecast was outside the warmup window.")
    predicted_prob: float | None = Field(default=None, description="Probability of the predicted tercile in `[0, 1]`; null when the forecast is unavailable.")
    realized_tercile: str | None = Field(default=None, description="Realised forward-vol tercile bucketed off the same cutoffs; null when the forward window has not yet closed.")
    realized_rv: float | None = Field(default=None, description="Realised RV measured at the forward bar; null when the forward window has not yet closed.")
    correct: bool | None = Field(default=None, description="True when `predicted_tercile == realized_tercile`; null when the row is unresolved.")


class HarAccuracyMetrics(BaseModel):
    """Aggregate accuracy KPIs across the HAR-tercile backtest rows.

    ``total_runs`` is the number of rows in the window, regardless of
    whether their forward window has resolved. ``resolved_runs`` counts
    rows whose realized tercile could be derived. ``accuracy_overall``
    is the hit rate across resolved rows only; ``per_tercile_hit_rate``
    keys are the predicted-tercile labels and values are per-label hit
    rates (denominator = resolved rows whose prediction was that label).
    """

    model_config = _FORBID_FROZEN_CONFIG

    total_runs: int = Field(..., description="Total rows in the window regardless of forward-window resolution.")
    resolved_runs: int = Field(..., description="Number of rows whose realised tercile could be derived.")
    accuracy_overall: float | None = Field(default=None, description="Hit rate across resolved rows in `[0, 1]`; null when none resolved.")
    per_tercile_hit_rate: dict[str, float] = Field(default_factory=dict, description="Per-predicted-tercile hit rate keyed by predicted label.")


class HarTercileBacktestResponse(BaseModel):
    """Response wire shape for ``GET /forecast/har-tercile-backtest``.

    Surfaces the last N FOMC meetings with their on-demand HAR-tercile
    prediction and the realized tercile derived from forward market
    history. Drives the HarAccuracyPanel card.
    """

    model_config = _FORBID_FROZEN_CONFIG

    symbol: str = Field(..., description="Yahoo Finance ticker the backtest was run against.")
    horizon: int = Field(..., description="Forward horizon in trading days the realised tercile was measured at.")
    rows: list[HarTercileBacktestRow] = Field(..., description="Per-event backtest rows ordered by `event_date`.")
    metrics: HarAccuracyMetrics = Field(..., description="Aggregate accuracy metrics across the rows.")
    generated_at: str = Field(..., description="ISO-8601 UTC timestamp the response was generated at.")


class RvBacktestRow(BaseModel):
    """One resolved (or pending) row in the QLIKE-RV backtest table.

    Carries the h=1 point forecast + 80% / 90% conformal bands for the
    persisted FOMC event date, the realized RV on the predicted bar, and
    per-band hit flags. The forecast columns are None on pending rows
    whose event sits inside HAR's monthly-lag warmup window or beyond
    the right edge of the available RV history; ``in_band_*`` are None
    whenever the realized RV is unresolved.
    """

    model_config = _FORBID_FROZEN_CONFIG

    event_date: str = Field(..., description="ISO `YYYY-MM-DD` event date the row was anchored to.")
    point_forecast_rv: float | None = Field(default=None, description="h=1 point forecast of realised variance; null on pending rows inside the HAR warmup or beyond RV history.")
    band_lo_80: float | None = Field(default=None, description="Lower edge of the 80% conformal band; null on pending rows.")
    band_hi_80: float | None = Field(default=None, description="Upper edge of the 80% conformal band; null on pending rows.")
    band_lo_90: float | None = Field(default=None, description="Lower edge of the 90% conformal band; null on pending rows.")
    band_hi_90: float | None = Field(default=None, description="Upper edge of the 90% conformal band; null on pending rows.")
    realized_rv: float | None = Field(default=None, description="Realised RV on the predicted bar; null when the bar is unresolved.")
    in_band_80: bool | None = Field(default=None, description="True when `realized_rv` lies inside the 80% band; null when realised RV is unresolved.")
    in_band_90: bool | None = Field(default=None, description="True when `realized_rv` lies inside the 90% band; null when realised RV is unresolved.")


class RvBacktestCoverage(BaseModel):
    """Aggregate empirical band coverage across the backtest rows.

    ``empirical_coverage_80`` / ``empirical_coverage_90`` are the fraction
    of resolved rows whose realized RV landed inside the corresponding
    conformal band. ``pending_runs`` reports rows we could not score
    (event date in the HAR warmup window or outside the available RV
    history); keeping it separate from ``resolved_runs`` keeps the
    coverage denominator honest. ``nominal_coverage_*`` are pinned at the
    calibration targets (0.80 / 0.90) so the frontend can render a
    nominal-vs-empirical gap chip without re-deriving the constants.
    """

    model_config = _FORBID_FROZEN_CONFIG

    total_runs: int = Field(..., description="Total rows in the backtest window regardless of resolution.")
    resolved_runs: int = Field(..., description="Number of rows with realised RV on the predicted bar.")
    pending_runs: int = Field(default=0, description="Rows that could not be scored because the event sits in the HAR warmup or beyond available RV history.")
    empirical_coverage_80: float | None = Field(default=None, description="Fraction of resolved rows whose realised RV landed inside the 80% band; null when none resolved.")
    empirical_coverage_90: float | None = Field(default=None, description="Fraction of resolved rows whose realised RV landed inside the 90% band; null when none resolved.")
    nominal_coverage_80: float = Field(default=0.80, description="Calibration target for the 80% band, pinned to 0.80.")
    nominal_coverage_90: float = Field(default=0.90, description="Calibration target for the 90% band, pinned to 0.90.")


class RvBacktestResponse(BaseModel):
    """Response wire shape for ``GET /forecast/rv-backtest``.

    Walks the last N persisted ^GSPC analyze runs and reports the
    QLIKE-RV h=1 point forecast + 80% / 90% bands against the realized
    RV on the same bar. Drives the RvAccuracyPanel card alongside the
    HAR-tercile accuracy surface.
    """

    model_config = _FORBID_FROZEN_CONFIG

    symbol: str = Field(..., description="Yahoo Finance ticker the RV backtest was run against.")
    horizon: int = Field(..., description="Forward horizon in trading days the bands were calibrated for (h=1).")
    rows: list[RvBacktestRow] = Field(..., description="Per-event RV backtest rows ordered by `event_date`.")
    coverage: RvBacktestCoverage = Field(..., description="Aggregate empirical-vs-nominal coverage across the rows.")
    generated_at: str = Field(..., description="ISO-8601 UTC timestamp the response was generated at.")


class CrossBankCard(BaseModel):
    """One central-bank card on the cross-bank dashboard panel.

    Mirrors the dict shape emitted by
    :func:`app.services.cross_bank_snapshot.build_bank_card`. Heads
    that could not be filled (corpus missing for a bank, classifier
    checkpoint unavailable, market-data lookup failed) leave the
    matching field as ``None`` and surface the reason in ``status`` /
    ``vol_regime_status`` so the frontend renders an explicit
    "Coming soon" placeholder rather than blank space.
    """

    model_config = _FORBID_FROZEN_CONFIG

    bank: str = Field(..., description="Full bank name identifier, e.g. `federal_reserve`, `european_central_bank`.")
    short_code: str = Field(..., description="Short code used in artifacts/keys, e.g. `fed`, `ecb`, `boj`.")
    display_name: str = Field(..., description="Human-readable bank name shown on the card.")
    flag: str = Field(..., description="Flag emoji or ISO code for the bank's jurisdiction.")
    symbol: str = Field(..., description="Yahoo Finance ticker the vol-regime card was anchored to.")
    latest_statement_date: str | None = Field(..., description="ISO `YYYY-MM-DD` event date of the latest statement scored; null when no statement was found.")
    stance: dict[str, float] | None = Field(..., description="Per-class stance distribution over hawkish/dovish/neutral; null when the head is unavailable.")
    stance_label: str | None = Field(..., description="Argmax stance label; null when the head is unavailable.")
    stance_confidence: float | None = Field(..., description="Probability of `stance_label` in `[0, 1]`; null when the head is unavailable.")
    certainty_label: str | None = Field(..., description="Argmax certainty label; null when the head is unavailable.")
    certainty_confidence: float | None = Field(..., description="Probability of `certainty_label` in `[0, 1]`; null when the head is unavailable.")
    time_axis: str | None = Field(..., description="Forward-looking time axis label; null when the head is unavailable.")
    vol_regime_label: str | None = Field(..., description="Argmax vol-regime label (calm/normal/high); null when the head is unavailable.")
    vol_regime_confidence: float | None = Field(..., description="Probability of `vol_regime_label` in `[0, 1]`; null when the head is unavailable.")
    vol_regime_as_of: str | None = Field(..., description="ISO `YYYY-MM-DD` of the trading session the vol-regime card was anchored at; null when unavailable.")
    vol_regime_status: str | None = Field(..., description="Status string for the vol-regime card, e.g. `ok` or a degradation reason.")
    sample_size: int = Field(..., description="Number of historical statements that fed the card's distribution.")
    status: str = Field(..., description="Overall status string for the card, e.g. `ok` or the reason the card is partial.")


class CrossBankSnapshotResponse(BaseModel):
    """Response wire shape for ``GET /cross-bank/snapshot``.

    Six-card side-by-side stance + vol-regime read across Fed, ECB,
    BoE, BoC, BoJ, RBA. Cached in-process for an hour (statements do
    not change minute-to-minute and the classifier cold-start is
    expensive).
    """

    model_config = _FORBID_FROZEN_CONFIG

    banks: list[CrossBankCard] = Field(..., description="Six-bank stance+vol-regime card list ordered for display.")
    generated_at: str = Field(..., description="ISO-8601 UTC timestamp the snapshot was generated at.")
    cache_ttl_seconds: int = Field(..., description="Seconds the in-process cache holds this snapshot before re-deriving.")
