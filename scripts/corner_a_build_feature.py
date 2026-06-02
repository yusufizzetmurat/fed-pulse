"""Corner A — build the leak-safe daily text-uncertainty feature.

Pre-registered in docs/research/corner-a-text-uncertainty-rv-preregistration.md.
Scores every FOMC statement's certainty axis with the served multi-axis
classifier, forms u = P(uncertain) − P(certain), and aligns it to the SPX RV
calendar as-of (backward fill, leak-safe). Writes a daily parquet
[date, u, post_fomc] consumed by scripts/corner_a_text_uncertainty.py.
Run inside the GPU container (needs torch + the classifier checkpoint).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from app.models.config import MULTI_TASK_CERTAINTY_LABELS as CL  # noqa: E402
from app.services import multi_axis_classifier as mac  # noqa: E402

EVENTS = "data/processed/tp_v3_full_rebuild_2026_05_30/events.parquet"
RV_PATH = "data/external/alphavantage_bars/spx_5min_daily_rv.parquet"
OUT = "data/artifacts/corner_a_text_uncertainty/text_uncertainty_daily.parquet"


def main() -> None:
    ev = pd.read_parquet(EVENTS)
    st = ev[ev["event_kind"] == "statement"].copy()
    st["event_date"] = pd.to_datetime(st["event_date"])
    # one row per meeting date — keep the fullest statement text
    st = (
        st.sort_values(["event_date", "token_count"])
        .drop_duplicates("event_date", keep="last")
        .sort_values("event_date")
        .reset_index(drop=True)
    )

    state = mac.get_classifier()
    if state is None:
        raise RuntimeError("multi-axis classifier failed to load")
    ci_unc, ci_cer = CL.index("uncertain"), CL.index("certain")
    us = []
    for text in st["text"].tolist():
        enc = state.tokenizer(str(text), return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = state.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        p = torch.softmax(logits["certainty"], dim=-1)[0]
        us.append(float(p[ci_unc] - p[ci_cer]))
    st["u"] = us

    rv = pd.read_parquet(RV_PATH)
    rv["date"] = pd.to_datetime(rv["date"])
    rv = rv.sort_values("date").reset_index(drop=True)
    merged = pd.merge_asof(
        rv[["date"]],
        st[["event_date", "u"]].rename(columns={"event_date": "date"}),
        on="date",
        direction="backward",
    )
    merged["u"] = merged["u"].fillna(0.0)

    statement_dates = set(st["event_date"])
    rv_dates = rv["date"].tolist()
    post = np.zeros(len(rv_dates), dtype=bool)
    for i, d in enumerate(rv_dates):
        for k in range(1, 6):  # post-FOMC window [d+1, d+5]
            if (d - pd.Timedelta(days=k)) in statement_dates:
                post[i] = True
                break
    merged["post_fomc"] = post

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT)
    u = np.asarray(us)
    print(
        f"statements={len(st)} u_mean={u.mean():.3f} u_std={u.std():.3f} "
        f"post_fomc_days={int(post.sum())} -> {OUT}"
    )


if __name__ == "__main__":
    main()
