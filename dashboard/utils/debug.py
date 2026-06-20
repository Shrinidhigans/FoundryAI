"""Lightweight debug logging for dashboard renderers."""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def log_df_info(name: str, df: pd.DataFrame) -> None:
    log.info("[DEBUG] %s shape=%s", name, df.shape)
    if len(df) == 0:
        return
    log.info("[DEBUG] %s dtypes sample: %s", name, dict(list(df.dtypes.head(15).items())))
    log.info("[DEBUG] %s columns: %s", name, list(df.columns)[:30])
