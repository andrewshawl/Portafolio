#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MT5 Portfolio Lab — CSV-only (SIN resampling) + UX "Patrón-friendly" + Reporte PDF
=================================================================================
Objetivo: que tu jefe lo use sin pensar (flujo guiado y respuestas directas).

Flujo:
1) Subir CSVs
2) 📥 Procesar CSVs (carga/validación/merge por símbolo)
3) Configurar rango y ▶️ Iniciar análisis
4) Ver "Resumen Ejecutivo" (insights + semáforos + portafolio recomendado)
5) Exportar CSVs y 📄 Reporte PDF (2 páginas)

Notas:
- No resamplea. Se asume que los CSV ya vienen en la MISMA temporalidad.
- MT5 suele traer Tick Volume (<TICKVOL>), no volumen real: para comparaciones entre activos
  se usa volumen relativo (Vol/MA) o z-score.
"""

from __future__ import annotations

import io
import re
import hashlib
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# clustering (sin matplotlib)
try:
    from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
    from scipy.spatial.distance import squareform
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# PDF
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False


# ---------------- UI ----------------
st.set_page_config(page_title="MT5 Portfolio Lab", page_icon="📈", layout="wide")
st.title("📈 MT5 Portfolio Lab — modo patròn (CSV MT5)")
st.caption("Flujo guiado. Sin resampling. El análisis NO corre solo: tú lo disparas con un botón.")

TZ_CDMX = pytz.timezone("America/Mexico_City")


def rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


# ============================================================
# Loader MT5 (tipo analyzer)
# ============================================================
USECOLS = ["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<TICKVOL>"]
RENAME = {
    "<DATE>": "Date",
    "<TIME>": "Time",
    "<OPEN>": "Open",
    "<HIGH>": "High",
    "<LOW>": "Low",
    "<CLOSE>": "Close",
    "<TICKVOL>": "Volume",
}


def detect_encoding(b: bytes) -> str:
    if b.startswith(b"\xff\xfe") or b.startswith(b"\xfe\xff"):
        return "utf-16"
    if b.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if b"\x00" in b[:2000]:
        return "utf-16"
    return "utf-8-sig"


def first_line_text(b: bytes, enc: str) -> str:
    i = b.find(b"\n")
    head = b if i == -1 else b[:i]
    try:
        return head.decode(enc, errors="ignore").lstrip()
    except Exception:
        return head.decode("utf-8", errors="ignore").lstrip()


def infer_symbol_from_filename(name: str) -> str:
    base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    base = base.rsplit(".", 1)[0].strip()
    sym = base.split("_")[0].upper()
    sym = re.split(r"[,\s;()\-]+", sym)[0]
    return sym.upper()


def _safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_and_prepare_bytes(file_bytes: bytes) -> Tuple[pd.DataFrame, Dict]:
    """
    Devuelve df con columnas:
      datetime_cdmx (tz-naive pero en hora CDMX),
      Open, High, Low, Close, Volume, range_pts
    """
    enc = detect_encoding(file_bytes)
    head = first_line_text(file_bytes, enc)
    is_csv = head.startswith("<DATE>")

    info = {"encoding": enc, "hdr": "<DATE>" if is_csv else "other", "sep": None, "note": ""}

    bio = io.BytesIO(file_bytes)

    if is_csv:
        df = None
        # intento 1: TAB
        try:
            bio.seek(0)
            df = pd.read_csv(bio, sep="\t", usecols=USECOLS, encoding=enc).rename(columns=RENAME)
            info["sep"] = "\\t"
        except Exception:
            df = None

        # intento 2: whitespace
        if df is None or df.empty:
            bio.seek(0)
            df = pd.read_csv(bio, sep=r"\s+", engine="python", usecols=USECOLS, encoding=enc).rename(columns=RENAME)
            info["sep"] = "\\s+"

        dt_utc = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            format="%Y.%m.%d %H:%M:%S",
            utc=True,
            errors="coerce",
        )
        df = df.assign(datetime_utc=dt_utc).dropna(subset=["datetime_utc"])
    else:
        # formato alterno (compatibilidad)
        cols = ["Symbol", "Date", "Time", "Open", "High", "Low", "Close", "Volume"]
        bio.seek(0)
        df = pd.read_csv(bio, names=cols, header=None, delim_whitespace=True, encoding=enc)
        dt_utc = pd.to_datetime(
            df["Date"].astype(str) + df["Time"].astype(str),
            format="%Y%m%d%H%M%S",
            utc=True,
            errors="coerce",
        )
        df = df.assign(datetime_utc=dt_utc).dropna(subset=["datetime_utc"])

    # numéricos
    df = _safe_numeric(df, ["Open", "High", "Low", "Close", "Volume"])
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # ✅ convertir a CDMX y volver tz-naive
    dt_cdmx_naive = df["datetime_utc"].dt.tz_convert(TZ_CDMX).dt.tz_localize(None)
    df = df.assign(datetime_cdmx=dt_cdmx_naive).sort_values("datetime_cdmx")

    df["range_pts"] = df["High"] - df["Low"]
    out = df[["datetime_cdmx", "Open", "High", "Low", "Close", "Volume", "range_pts"]].copy()
    out = out.drop_duplicates(subset=["datetime_cdmx"], keep="last")
    return out, info


def to_indexed_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    x = raw.copy()
    x = x.set_index("datetime_cdmx").sort_index()
    x.index.name = "datetime"
    x = x[~x.index.duplicated(keep="last")]
    return x


# ============================================================
# Temporalidad / anualización
# ============================================================
def infer_dt(index: pd.DatetimeIndex) -> Optional[pd.Timedelta]:
    if index is None or len(index) < 3:
        return None
    d = pd.Series(index).diff().dropna()
    if d.empty:
        return None
    return d.median()


def timeframe_label(dt: Optional[pd.Timedelta]) -> str:
    if dt is None or dt <= pd.Timedelta(0):
        return "—"
    sec = dt.total_seconds()
    cand = [
        ("1min", 60),
        ("5min", 300),
        ("15min", 900),
        ("30min", 1800),
        ("1H", 3600),
        ("4H", 14400),
        ("1D", 86400),
    ]
    for name, s in cand:
        if abs(sec - s) / s < 0.05:
            return name
    if sec < 60:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{sec/60:.2f}min"
    if sec < 86400:
        return f"{sec/3600:.2f}h"
    return f"{sec/86400:.3f}D"


def bars_per_day_from_dt(dt: Optional[pd.Timedelta]) -> Optional[float]:
    if dt is None or dt <= pd.Timedelta(0):
        return None
    sec = dt.total_seconds()
    if sec == 0:
        return None
    return float(86400.0 / sec)


def ann_factor_from_index(index: pd.DatetimeIndex, trading_days: int = 252) -> float:
    dt = infer_dt(index)
    bpd = bars_per_day_from_dt(dt)
    if bpd is None:
        return float(trading_days)
    if dt is not None and dt >= pd.Timedelta(days=1):
        return float(trading_days)
    return float(trading_days) * float(bpd)


# ============================================================
# Indicadores / métricas
# ============================================================
def compute_adx_atr(df: pd.DataFrame, n: int = 14) -> Tuple[pd.Series, pd.Series]:
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    close = df["Close"].astype(float)
    high = df["High"].astype(float).fillna(close)
    low = df["Low"].astype(float).fillna(close)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean()

    plus_di = 100 * (plus_dm_s / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm_s / atr.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx, atr


def rolling_r2_from_close(close: pd.Series, win: int = 200) -> pd.Series:
    close = close.dropna()
    if len(close) < win + 5:
        return pd.Series(index=close.index, dtype=float)
    y = np.log(close.astype(float))
    x = pd.Series(np.arange(len(y)), index=y.index, dtype=float)
    r = y.rolling(win).corr(x)
    return (r ** 2).rename("R2")


def underwater_curve(close: pd.Series) -> pd.Series:
    close = close.dropna()
    if close.empty:
        return pd.Series(dtype=float)
    return (close / close.cummax() - 1).rename("DD")


def drawdown_events(
    close: pd.Series,
    min_new_high: float = 0.0,
    min_dd: float = 0.0,
) -> pd.DataFrame:
    """
    Eventos: Peak -> Trough -> Recovery.
    min_new_high: histéresis para ignorar micro-peaks (ej 0.002 = 0.2%)
    min_dd: filtra eventos menores (ej 0.02 = solo DD >= 2%)
    """
    close = close.dropna()
    if close.empty:
        return pd.DataFrame()

    events = []
    peak_price = float(close.iloc[0])
    peak_date = close.index[0]
    trough_price = peak_price
    trough_date = peak_date
    in_dd = False

    for dt, price in close.iloc[1:].items():
        price = float(price)
        is_new_peak = price >= peak_price * (1.0 + float(min_new_high))

        if is_new_peak:
            if in_dd:
                ddp = trough_price / peak_price - 1.0
                events.append({"Peak": peak_date, "Trough": trough_date, "Recovery": dt, "DD%": ddp})
                in_dd = False
            peak_price = price
            peak_date = dt
            trough_price = price
            trough_date = dt
        else:
            if not in_dd:
                in_dd = True
                trough_price = price
                trough_date = dt
            if price < trough_price:
                trough_price = price
                trough_date = dt

    if in_dd:
        ddp = trough_price / peak_price - 1.0
        events.append({"Peak": peak_date, "Trough": trough_date, "Recovery": pd.NaT, "DD%": ddp})

    ev = pd.DataFrame(events)
    if ev.empty:
        return ev

    if min_dd > 0:
        ev = ev.loc[ev["DD%"] <= -abs(float(min_dd))].copy()

    ev["Peak"] = pd.to_datetime(ev["Peak"])
    ev["Trough"] = pd.to_datetime(ev["Trough"])
    ev["Recovery"] = pd.to_datetime(ev["Recovery"])

    ev["Dur Peak->Trough"] = ev["Trough"] - ev["Peak"]
    ev["Dur Trough->Recovery"] = ev["Recovery"] - ev["Trough"]
    ev["Dur Peak->Recovery"] = ev["Recovery"] - ev["Peak"]

    ev = ev.sort_values("DD%").reset_index(drop=True)
    return ev


def compute_metrics(
    df: pd.DataFrame,
    trading_days: int = 252,
    trend_win: int = 200,
    trend_lookback_days: int = 180,
) -> Optional[dict]:
    close = df["Close"].dropna()
    if len(close) < max(250, trend_win + 20):
        return None

    ann = ann_factor_from_index(close.index, trading_days=trading_days)

    rets = close.pct_change().dropna()
    if rets.empty:
        return None

    span_years = max((close.index[-1] - close.index[0]).total_seconds() / (365.25 * 86400.0), 1e-9)
    total_ret = float(close.iloc[-1] / close.iloc[0] - 1.0)
    cagr = float((close.iloc[-1] / close.iloc[0]) ** (1.0 / span_years) - 1.0)

    mean_ann = float(rets.mean() * ann)
    vol_ann = float(rets.std(ddof=0) * np.sqrt(ann))
    sharpe = float(mean_ann / vol_ann) if vol_ann > 0 else np.nan

    dd = underwater_curve(close)
    mdd = float(dd.min()) if not dd.empty else np.nan
    calmar = float(cagr / abs(mdd)) if pd.notna(mdd) and mdd < 0 else np.nan
    dd_current = float(dd.iloc[-1]) if not dd.empty else np.nan

    avg_range_pct = float(((df["High"] - df["Low"]).abs() / df["Close"]).replace([np.inf, -np.inf], np.nan).dropna().mean())

    adx, atr = compute_adx_atr(df, 14)
    adx_last = float(adx.dropna().iloc[-1]) if not adx.dropna().empty else np.nan
    atr_last = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else np.nan
    atr_pct = float(atr_last / close.iloc[-1]) if close.iloc[-1] != 0 else np.nan

    r2_series = rolling_r2_from_close(close, win=trend_win)
    r2_last = float(r2_series.dropna().iloc[-1]) if not r2_series.dropna().empty else np.nan

    label = "Mixto"
    if pd.notna(adx_last) and pd.notna(r2_last):
        if adx_last >= 25 and r2_last >= 0.20:
            label = "Tendencial"
        elif adx_last <= 20 and r2_last < 0.20:
            label = "Lateral"

    dt = infer_dt(close.index)
    bpd = bars_per_day_from_dt(dt) or 1.0
    lookback_bars = int(max(100, trend_lookback_days * bpd))
    sl = df.iloc[-lookback_bars:] if len(df) > lookback_bars else df

    adx_lb, _ = compute_adx_atr(sl, 14)
    r2_lb = rolling_r2_from_close(sl["Close"], win=min(trend_win, max(50, int(0.5 * lookback_bars))))

    z = pd.DataFrame({"ADX": adx_lb, "R2": r2_lb}).dropna()
    if z.empty:
        tend_share = np.nan
        lat_share = np.nan
    else:
        tend = (z["ADX"] >= 25) & (z["R2"] >= 0.20)
        lat = (z["ADX"] <= 20) & (z["R2"] < 0.20)
        tend_share = float(tend.mean())
        lat_share = float(lat.mean())

    vol_mean = float(df["Volume"].replace(0, np.nan).dropna().mean()) if df["Volume"].notna().any() else np.nan

    return {
        "Precio": float(close.iloc[-1]),
        "Retorno total": total_ret,
        "CAGR": cagr,
        "Mean ann": mean_ann,
        "Vol anual": vol_ann,
        "Sharpe": sharpe,
        "MaxDD": mdd,
        "DD actual": dd_current,
        "Calmar": calmar,
        "Avg rango%": avg_range_pct,
        "ATR14%": atr_pct,
        "ADX14": adx_last,
        "R2": r2_last,
        "% Tend": tend_share,
        "% Lat": lat_share,
        "Vol prom": vol_mean,
        "Tipo": label,
        "Barras": int(len(close)),
        "Desde": close.index.min(),
        "Hasta": close.index.max(),
        "AnnFactor": float(ann),
        "TF": timeframe_label(dt),
    }


def weekly_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    x = df.copy()
    idx = x.index
    week_id = (idx - pd.to_timedelta(idx.weekday, unit="D")).normalize()
    x = x.assign(_week=week_id)

    agg = x.groupby("_week").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    ).dropna(subset=["Close"])

    agg.index.name = "Week"
    agg["Ret semana"] = agg["Close"].pct_change()
    agg["Rango semana %"] = (agg["High"] - agg["Low"]) / agg["Close"].replace(0, np.nan)
    return agg


def this_week_summary(df: pd.DataFrame, trading_days: int = 252) -> Optional[dict]:
    if df.empty or df["Close"].dropna().shape[0] < 10:
        return None

    w = weekly_aggregation(df)
    if w.empty or w.shape[0] < 3:
        return None

    last = w.iloc[-1]
    ann = ann_factor_from_index(df.index, trading_days=trading_days)
    last_week = w.index[-1]
    mask = (df.index >= last_week) & (df.index < last_week + pd.Timedelta(days=7))
    intr = df.loc[mask, "Close"].dropna().pct_change().dropna()
    vol_week_ann = float(intr.std(ddof=0) * np.sqrt(ann)) if intr.shape[0] > 5 else np.nan

    return {
        "Precio fin semana": float(last["Close"]),
        "Retorno semana": float(last["Ret semana"]) if pd.notna(last["Ret semana"]) else np.nan,
        "Rango semana %": float(last["Rango semana %"]) if pd.notna(last["Rango semana %"]) else np.nan,
        "Vol semana (ann)": vol_week_ann,
        "Volumen semana (suma)": float(last["Volume"]) if pd.notna(last["Volume"]) else np.nan,
        "WeekStart": w.index[-1],
    }


def week_anomaly_scores(df: pd.DataFrame, lookback_weeks: int = 52) -> Optional[dict]:
    w = weekly_aggregation(df)
    if w.empty or w.shape[0] < max(10, lookback_weeks // 2):
        return None

    hist = w.iloc[:-1].copy()
    cur = w.iloc[-1].copy()

    ret_hist = hist["Ret semana"].dropna()
    if ret_hist.shape[0] < 8:
        z_ret = np.nan
    else:
        mu = ret_hist.tail(lookback_weeks).mean()
        sd = ret_hist.tail(lookback_weeks).std(ddof=0)
        z_ret = float((cur["Ret semana"] - mu) / sd) if sd and pd.notna(cur["Ret semana"]) else np.nan

    vol_hist = hist["Volume"].replace(0, np.nan).dropna()
    vol_ma = float(vol_hist.tail(lookback_weeks).mean()) if vol_hist.shape[0] else np.nan
    vol_ratio = float(cur["Volume"] / vol_ma) if vol_ma and pd.notna(cur["Volume"]) else np.nan

    rng_hist = hist["Rango semana %"].replace([np.inf, -np.inf], np.nan).dropna()
    if rng_hist.shape[0] >= 10 and pd.notna(cur["Rango semana %"]):
        pct_rng = float((rng_hist < cur["Rango semana %"]).mean())
    else:
        pct_rng = np.nan

    return {"Z Ret semana": z_ret, "Vol ratio vs MA": vol_ratio, "Pct rango semana": pct_rng}


def rolling_vol_peaks(close: pd.Series, win: int, top_n: int, ann: float):
    ret = close.pct_change().dropna()
    if len(ret) < win + 50:
        return None, None, None
    roll = ret.rolling(win).std(ddof=0) * np.sqrt(ann)
    peaks = roll.dropna().nlargest(top_n)
    table = pd.DataFrame(
        {
            "Fecha": peaks.index,
            "Vol rolling (ann)": peaks.values,
            "Ret 1": ret.reindex(peaks.index).values,
            "Ret 5": close.pct_change(5).reindex(peaks.index).values,
        }
    )
    return roll, peaks, table


# ============================================================
# Correlación / Clustering / Portafolio
# ============================================================
def shrink_cov(cov: pd.DataFrame, lam: float = 0.10, jitter: float = 1e-10) -> pd.DataFrame:
    cov = cov.copy()
    diag = np.diag(np.diag(cov.values))
    shr = (1.0 - lam) * cov.values + lam * diag
    shr = shr + np.eye(shr.shape[0]) * jitter
    return pd.DataFrame(shr, index=cov.index, columns=cov.columns)


def inverse_vol_weights(vol: pd.Series) -> pd.Series:
    v = vol.replace(0, np.nan).dropna()
    if v.empty:
        return pd.Series(dtype=float)
    w = 1.0 / v
    w = w / w.sum()
    return w


def min_var_weights_unconstrained(cov: pd.DataFrame) -> pd.Series:
    cov_ = cov.values
    n = cov_.shape[0]
    ones = np.ones((n, 1))
    try:
        inv = np.linalg.inv(cov_)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov_)
    w = inv @ ones
    w = w / float(ones.T @ inv @ ones)
    return pd.Series(w.flatten(), index=cov.index)


def risk_parity_weights(cov: pd.DataFrame, max_iter: int = 5000, tol: float = 1e-10) -> pd.Series:
    C = cov.values
    n = C.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        port_var = float(w @ C @ w)
        mrc = C @ w
        rc = w * mrc
        target = port_var / n
        diff = rc - target
        if np.max(np.abs(diff)) < tol:
            break
        w *= target / (rc + 1e-16)
        w = np.clip(w, 1e-12, None)
        w /= w.sum()
    return pd.Series(w, index=cov.index)


def cluster_order_from_corr(corr: pd.DataFrame) -> Optional[np.ndarray]:
    if not SCIPY_OK or corr.shape[0] < 3:
        return None
    dist = np.sqrt(0.5 * (1.0 - corr.fillna(0.0)))
    dist_cond = squareform(dist.values, checks=False)
    Z = linkage(dist_cond, method="average")
    return leaves_list(Z)


def cluster_labels_from_corr(corr: pd.DataFrame, k: int = 4) -> Optional[pd.Series]:
    if not SCIPY_OK or corr.shape[0] < 3:
        return None
    dist = np.sqrt(0.5 * (1.0 - corr.fillna(0.0)))
    dist_cond = squareform(dist.values, checks=False)
    Z = linkage(dist_cond, method="average")
    labels = fcluster(Z, t=k, criterion="maxclust")
    return pd.Series(labels, index=corr.index, name="Cluster")


# ============================================================
# Executive helpers (insights + semáforo + PDF)
# ============================================================
def safe_imshow(df: pd.DataFrame, title: str):
    try:
        fig = px.imshow(df, text_auto=".2f", aspect="auto", title=title)
    except TypeError:
        fig = px.imshow(df, aspect="auto", title=title)
    return fig


def build_risk_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary is None or summary.empty:
        return pd.DataFrame()

    s = summary.copy()

    # percentiles
    s["Vol_pct"] = s["Vol anual"].rank(pct=True)
    s["DD_bad"] = (-s["DD actual"]).rank(pct=True)  # más alto = peor
    s["MDD_bad"] = (-s["MaxDD"]).rank(pct=True)

    def classify(row):
        vol_pct = row.get("Vol_pct", np.nan)
        dd = row.get("DD actual", np.nan)
        mdd = row.get("MaxDD", np.nan)

        red = False
        yellow = False

        if pd.notna(mdd) and mdd <= -0.30:
            red = True
        if pd.notna(dd) and dd <= -0.10 and pd.notna(vol_pct) and vol_pct >= 0.70:
            red = True
        if pd.notna(vol_pct) and vol_pct >= 0.85 and pd.notna(dd) and dd <= -0.06:
            red = True

        if not red:
            if pd.notna(mdd) and mdd <= -0.20:
                yellow = True
            if pd.notna(dd) and dd <= -0.06:
                yellow = True
            if pd.notna(vol_pct) and vol_pct >= 0.70:
                yellow = True

        if red:
            return "🔴 Alto", "Reducir exposición / cobertura"
        if yellow:
            return "🟡 Medio", "Operar con tamaño moderado"
        return "🟢 Bajo", "Ok para operar"

    out = []
    for sym, row in s.iterrows():
        level, action = classify(row)
        out.append({
            "Símbolo": sym,
            "Riesgo": level,
            "Acción": action,
            "Tipo": row.get("Tipo", "—"),
            "Vol anual": row.get("Vol anual", np.nan),
            "DD actual": row.get("DD actual", np.nan),
            "MaxDD": row.get("MaxDD", np.nan),
            "CAGR": row.get("CAGR", np.nan),
            "Score": row.get("Score", np.nan),
            "TF": row.get("TF", "—"),
        })

    out_df = pd.DataFrame(out).set_index("Símbolo")
    return out_df


def build_executive_insights(summary: pd.DataFrame, week_sev: pd.DataFrame, reco: dict, start: str, end: str) -> List[str]:
    bullets: List[str] = []
    bullets.append(f"Rango analizado: **{start} → {end}**.")

    if summary is not None and not summary.empty:
        top_vol = summary["Vol anual"].sort_values(ascending=False).head(1)
        if not top_vol.empty:
            sym = top_vol.index[0]
            bullets.append(f"Activo más volátil (anualizado): **{sym}** (Vol={top_vol.iloc[0]:.1%}).")

        if "Score" in summary.columns and summary["Score"].notna().any():
            top_score = summary["Score"].sort_values(ascending=False).head(1)
            sym = top_score.index[0]
            row = summary.loc[sym]
            bullets.append(
                f"Mejor perfil riesgo/retorno (Score): **{sym}** "
                f"(Score={row['Score']:.2f}, CAGR={row['CAGR']:.1%}, Sharpe={row['Sharpe']:.2f}, MaxDD={row['MaxDD']:.1%})."
            )

        if "Tipo" in summary.columns:
            tend = int((summary["Tipo"] == "Tendencial").sum())
            lat = int((summary["Tipo"] == "Lateral").sum())
            mix = int((summary["Tipo"] == "Mixto").sum())
            bullets.append(f"Clasificación: **{tend} tendenciales**, **{lat} laterales**, **{mix} mixtos** (según ADX/R²).")

    if week_sev is not None and not week_sev.empty:
        sym = week_sev.index[0]
        r = week_sev.iloc[0]
        ret = r.get("Retorno semana", np.nan)
        sev = r.get("Severidad", np.nan)
        bullets.append(
            f"Esta semana, el movimiento más relevante fue **{sym}** "
            f"(Ret semana={ret:.1%} | Severidad={sev:.2f})."
        )

    if reco and reco.get("selected"):
        bullets.append(f"Portafolio sugerido (diversificación): **{', '.join(reco['selected'])}**.")
        if reco.get("note"):
            bullets.append(f"Método: {reco['note']}")

    bullets.append("Recuerda: el 'volumen' es **TickVol** (indicador relativo, no volumen real).")
    return bullets


def compute_portfolio_metrics_from_reco(rets_df: pd.DataFrame, reco: dict, trading_days: int) -> dict:
    out = {"vol": np.nan, "sharpe": np.nan, "maxdd": np.nan}
    if rets_df is None or rets_df.empty:
        return out
    w = reco.get("weights")
    sel = reco.get("selected", [])
    if w is None or getattr(w, "empty", True) or not sel:
        return out

    common = [c for c in sel if c in rets_df.columns and c in w.index]
    if len(common) < 2:
        return out

    R = rets_df[common].dropna(how="any")
    if R.shape[0] < 200:
        return out

    ww = w.reindex(common).astype(float)
    ww = ww / ww.sum()
    port_lr = (R * ww).sum(axis=1)  # log-ret
    ann = ann_factor_from_index(port_lr.index, trading_days=trading_days)

    mean = float(port_lr.mean() * ann)
    vol = float(port_lr.std(ddof=0) * np.sqrt(ann))
    sharpe = mean / vol if vol > 0 else np.nan

    port_curve = np.exp(port_lr.cumsum())
    dd = underwater_curve(port_curve)
    maxdd = float(dd.min()) if not dd.empty else np.nan

    out.update({"vol": vol, "sharpe": sharpe, "maxdd": maxdd})
    return out


def df_to_table_data(df: pd.DataFrame, cols: List[str], max_rows: int = 12) -> List[List[str]]:
    if df is None or df.empty:
        return [["—"]]
    x = df.copy()
    x = x.head(max_rows)
    data = [ ["Símbolo"] + cols ]
    for sym, row in x.iterrows():
        r = [sym]
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, (float, np.floating)):
                if "Vol" in c or "CAGR" in c or "DD" in c or "MaxDD" in c or "%" in c:
                    r.append(f"{v*100:.1f}%")
                else:
                    r.append(f"{v:.2f}")
            else:
                r.append(str(v))
        data.append(r)
    return data


def make_pdf_report(
    title: str,
    start: str,
    end: str,
    symbols: List[str],
    tf_labels: List[str],
    bullets: List[str],
    summary: pd.DataFrame,
    risk_df: pd.DataFrame,
    week_sev: pd.DataFrame,
    reco: dict,
    port_metrics: dict,
) -> bytes:
    if not REPORTLAB_OK:
        return b""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    story = []
    story.append(Paragraph(title, styles["Title"]))
    gen = datetime.now(TZ_CDMX).strftime("%Y-%m-%d %H:%M CDMX")
    story.append(Paragraph(f"Generado: {gen}", styles["Normal"]))
    story.append(Paragraph(f"Rango: {start} → {end}", styles["Normal"]))
    story.append(Paragraph(f"Activos: {', '.join(symbols)}", styles["Normal"]))
    story.append(Paragraph(f"Temporalidad detectada: {', '.join(tf_labels) if tf_labels else '—'}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Executive Brief", styles["Heading2"]))
    for b in bullets[:12]:
        story.append(Paragraph(f"• {b}", styles["Normal"]))
    story.append(Spacer(1, 12))

    if summary is not None and not summary.empty:
        story.append(Paragraph("Top volatilidad", styles["Heading2"]))
        top_vol = summary.sort_values("Vol anual", ascending=False)
        data = df_to_table_data(top_vol, ["Vol anual", "CAGR", "Sharpe", "MaxDD", "Tipo"], max_rows=10)
        t = Table(data, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

        story.append(Paragraph("Top rentabilidad (Score)", styles["Heading2"]))
        top_score = summary.sort_values("Score", ascending=False) if "Score" in summary.columns else summary
        data = df_to_table_data(top_score, ["Score", "CAGR", "Vol anual", "Sharpe", "MaxDD", "Tipo"], max_rows=10)
        t = Table(data, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

    if risk_df is not None and not risk_df.empty:
        story.append(Paragraph("Semáforo de riesgo (resumen)", styles["Heading2"]))
        data = df_to_table_data(risk_df, ["Riesgo", "Acción", "Tipo", "Vol anual", "DD actual"], max_rows=12)
        t = Table(data, hAlign="LEFT", colWidths=[70, 70, 120, 70, 70, 70])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

    if week_sev is not None and not week_sev.empty:
        story.append(Paragraph("Esta semana (top severidad)", styles["Heading2"]))
        cols = [c for c in ["Retorno semana", "Rango semana %", "Volumen semana (suma)", "Z Ret semana", "Severidad"] if c in week_sev.columns]
        data = df_to_table_data(week_sev, cols, max_rows=8)
        t = Table(data, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
        ]))
        story.append(t)

    story.append(PageBreak())

    story.append(Paragraph("Portafolio", styles["Title"]))
    if reco and reco.get("selected"):
        story.append(Paragraph(f"Selección sugerida: {', '.join(reco['selected'])}", styles["Normal"]))
        if reco.get("note"):
            story.append(Paragraph(f"Método: {reco['note']}", styles["Normal"]))
        story.append(Spacer(1, 10))

        w = reco.get("weights")
        if w is not None and not getattr(w, "empty", True):
            wdf = w.to_frame("Peso").copy()
            data = [["Símbolo", "Peso"]] + [[idx, f"{float(val)*100:.1f}%"] for idx, val in wdf["Peso"].items()]
            t = Table(data, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE", (0,0), (-1,-1), 9),
            ]))
            story.append(t)
            story.append(Spacer(1, 10))

        if port_metrics:
            story.append(Paragraph("Métricas del portafolio (aprox.)", styles["Heading2"]))
            story.append(Paragraph(f"Vol anual: {port_metrics.get('vol', np.nan)*100:.1f}%  |  Sharpe: {port_metrics.get('sharpe', np.nan):.2f}  |  MaxDD: {port_metrics.get('maxdd', np.nan)*100:.1f}%", styles["Normal"]))
            story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("No se pudo generar portafolio recomendado (faltó traslape o activos).", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Notas", styles["Heading2"]))
    story.append(Paragraph("• El 'volumen' proviene de TICKVOL (tick volume). Úsalo como señal relativa, no como volumen real.", styles["Normal"]))
    story.append(Paragraph("• Las métricas son descriptivas; no constituyen recomendación financiera.", styles["Normal"]))

    doc.build(story)
    return buf.getvalue()


# ============================================================
# Formatting helpers
# ============================================================
def fmt_pct(x: float, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x*100:.{digits}f}%"


def fmt_num(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:.{digits}f}"


# ============================================================
# SESSION STATE INIT
# ============================================================
if "processed" not in st.session_state:
    st.session_state.processed = False
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "analysis_hash" not in st.session_state:
    st.session_state.analysis_hash = None
if "processed_file_names" not in st.session_state:
    st.session_state.processed_file_names = []
if "series_raw" not in st.session_state:
    st.session_state.series_raw = {}
if "meta_df" not in st.session_state:
    st.session_state.meta_df = pd.DataFrame()
if "ranges" not in st.session_state:
    st.session_state.ranges = None


# ============================================================
# Sidebar — Modo + Carga
# ============================================================
st.sidebar.header("👔 Modo de uso")
mode = st.sidebar.radio(
    "¿Para quién es esta vista?",
    ["👔 Patrón (simple)", "🧠 Analista (detallado)"],
    index=0,
)
SIMPLE = mode.startswith("👔")

st.sidebar.markdown("---")
st.sidebar.header("Paso 1 — Sube CSVs")
files = st.sidebar.file_uploader(
    "Sube CSV MT5 (misma temporalidad)",
    type=["csv", "txt"],
    accept_multiple_files=True,
)

if not files:
    st.info("1) Sube CSVs en la barra lateral.\n\n2) Presiona **📥 Procesar CSVs**.")
    st.stop()

current_names = sorted([f.name for f in files])

# Si cambian archivos, invalidamos
if st.session_state.processed and current_names != st.session_state.processed_file_names:
    st.session_state.processed = False
    st.session_state.analysis = None
    st.session_state.analysis_hash = None
    st.session_state.series_raw = {}
    st.session_state.meta_df = pd.DataFrame()
    st.session_state.ranges = None
    st.sidebar.warning("Detecté cambio en archivos. Vuelve a procesar CSVs.")

with st.sidebar.expander("Símbolo por archivo (opcional)", expanded=False):
    overrides = {}
    for f in files:
        guess = infer_symbol_from_filename(f.name)
        sym = st.text_input(f.name, value=guess, key=f"sym_{f.name}")
        overrides[f.name] = sym.strip().upper()

st.sidebar.markdown("---")
colb1, colb2 = st.sidebar.columns(2)
btn_process_side = colb1.button("📥 Procesar", type="primary")
btn_reset = colb2.button("🧹 Reset")

if btn_reset:
    st.session_state.processed = False
    st.session_state.analysis = None
    st.session_state.analysis_hash = None
    st.session_state.series_raw = {}
    st.session_state.meta_df = pd.DataFrame()
    st.session_state.ranges = None
    st.session_state.processed_file_names = []
    rerun()

# Barra de progreso + botones en main
st.markdown("### ✅ Flujo recomendado")
stage = 0 if not st.session_state.processed else (1 if st.session_state.analysis is None else 2)
progress = {0: 0.33, 1: 0.66, 2: 1.0}[stage]
st.progress(progress)
if stage == 0:
    st.info("**Paso 1/3:** Sube CSVs → presiona **📥 Procesar CSVs**.")
elif stage == 1:
    st.info("**Paso 2/3:** Ajusta rango → presiona **▶️ Iniciar análisis**.")
else:
    st.success("**Paso 3/3:** Resultados listos. Abre **📌 Resumen Ejecutivo**.")

col_main_a, col_main_b, col_main_c = st.columns([1, 1, 2])
with col_main_a:
    btn_process_main = st.button("📥 Procesar CSVs", type="primary", key="process_main")
with col_main_b:
    btn_analyze_main = st.button("▶️ Iniciar análisis", type="primary", key="analyze_main")
with col_main_c:
    st.caption("Tip: si cambias archivos, vuelve a **Procesar**. Si cambias parámetros, vuelve a **Iniciar**.")

process_now = bool(btn_process_side or btn_process_main)

# ============================================================
# Procesar CSVs (solo cuando se presiona)
# ============================================================
def process_files(files_list, overrides_map) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, dict]:
    series_raw: Dict[str, pd.DataFrame] = {}
    meta_rows = []

    for f in files_list:
        sym = overrides_map.get(f.name, infer_symbol_from_filename(f.name))
        raw, info = load_and_prepare_bytes(f.getvalue())
        x = to_indexed_ohlcv(raw)

        if not x.empty:
            if sym in series_raw:
                comb = pd.concat([series_raw[sym], x]).sort_index()
                comb = comb[~comb.index.duplicated(keep="last")]
                series_raw[sym] = comb
            else:
                series_raw[sym] = x

        dt = infer_dt(x.index) if not x.empty else None
        meta_rows.append(
            {
                "Archivo": f.name,
                "Símbolo": sym,
                "Barras": int(len(x)),
                "Desde": (x.index.min().strftime("%Y-%m-%d %H:%M") if not x.empty else "—"),
                "Hasta": (x.index.max().strftime("%Y-%m-%d %H:%M") if not x.empty else "—"),
                "Temporalidad (est.)": timeframe_label(dt),
                "Δt mediana": str(dt) if dt is not None else "—",
                "Enc": info["encoding"],
                "Hdr": info["hdr"],
                "Sep": info["sep"],
            }
        )

    meta_df = pd.DataFrame(meta_rows)

    if not series_raw:
        ranges = {"gmin": None, "gmax": None, "common_start": None, "common_end": None}
        return series_raw, meta_df, ranges

    symbols_all = sorted(series_raw.keys())
    gmin = min(series_raw[s].index.min() for s in symbols_all)
    gmax = max(series_raw[s].index.max() for s in symbols_all)
    common_start = max(series_raw[s].index.min() for s in symbols_all)
    common_end = min(series_raw[s].index.max() for s in symbols_all)

    ranges = {"gmin": gmin, "gmax": gmax, "common_start": common_start, "common_end": common_end}
    return series_raw, meta_df, ranges


if process_now:
    with st.spinner("Procesando CSVs..."):
        sraw, mdf, ranges = process_files(files, overrides)
        st.session_state.series_raw = sraw
        st.session_state.meta_df = mdf
        st.session_state.ranges = ranges
        st.session_state.processed = True
        st.session_state.analysis = None
        st.session_state.analysis_hash = None
        st.session_state.processed_file_names = current_names

# ============================================================
# Mostrar estado de carga (sin análisis)
# ============================================================
st.subheader("📦 Estado de carga")
if not st.session_state.processed:
    st.warning("Aún no se han procesado los CSVs. Presiona **📥 Procesar CSVs**.")
    st.stop()

meta_df = st.session_state.meta_df
st.dataframe(meta_df, use_container_width=True)

series_raw = st.session_state.series_raw
if not series_raw:
    st.error("No se pudo cargar ningún símbolo. Revisa formato/archivos.")
    st.stop()

symbols_all = sorted(series_raw.keys())
ranges = st.session_state.ranges
gmin, gmax = ranges["gmin"], ranges["gmax"]
common_start, common_end = ranges["common_start"], ranges["common_end"]

# Validación temporalidad (modo estricto)
dt_labels = []
for s in symbols_all:
    dt = infer_dt(series_raw[s].index)
    dt_labels.append(timeframe_label(dt))
unique_labels = sorted(set([x for x in dt_labels if x != "—"]))

# ============================================================
# Sidebar — Parámetros (simple por default)
# ============================================================
st.sidebar.markdown("---")
st.sidebar.header("Paso 2 — Configura rango")

modo_estricto = st.sidebar.checkbox("Exigir misma temporalidad", value=True)
if modo_estricto and len(unique_labels) > 1:
    st.error(f"Temporalidades detectadas: {unique_labels}. En modo estricto deben ser iguales.")
    st.stop()

st.sidebar.caption(f"Rango global: {gmin:%Y-%m-%d} → {gmax:%Y-%m-%d}")
st.sidebar.caption(f"Rango común:  {common_start:%Y-%m-%d} → {common_end:%Y-%m-%d}")

if "start_date" not in st.session_state or st.session_state.start_date is None:
    st.session_state.start_date = gmin.date()
if "end_date" not in st.session_state or st.session_state.end_date is None:
    st.session_state.end_date = gmax.date()

if st.sidebar.button("📌 Usar rango común", help="Recorta para que todos los activos tengan traslape perfecto."):
    st.session_state.start_date = common_start.date()
    st.session_state.end_date = common_end.date()

start = st.sidebar.date_input("Inicio", value=st.session_state.start_date)
end = st.sidebar.date_input("Fin", value=st.session_state.end_date)

if start > end:
    st.sidebar.error("Inicio no puede ser después de Fin.")
    st.stop()

trading_days = st.sidebar.selectbox("Días/año (anualización)", [252, 365], index=0)

with st.sidebar.expander("⚙️ Ajustes avanzados", expanded=not SIMPLE):
    trend_win = st.slider("Ventana R² (barras)", 50, 600, 200)
    trend_lookback_days = st.slider("Lookback % Tend/% Lat (días)", 30, 365, 180)
    roll_vol_days = st.slider("Vol rolling (días)", 1, 180, 30)
    roll_corr_days = st.slider("Rolling corr (días)", 1, 365, 90)
    top_dd = st.selectbox("Top drawdowns", [3, 5, 10], index=1)
    top_peaks = st.selectbox("Top picos vol", [5, 10, 20, 30], index=1)
    min_new_high = st.slider("Ignorar micro-peaks (nuevo high mínimo %)", 0.0, 1.0, 0.20, 0.05) / 100.0
    min_dd_event = st.slider("Solo eventos DD >= (%)", 0.0, 20.0, 1.0, 0.5) / 100.0
    use_common_for_corr = st.checkbox("Correlación/portafolio: usar rango común", value=True)

    # slider robusto para k
    n_assets = len(symbols_all)
    if n_assets < 3:
        portfolio_k = 2
        st.caption("Clustering requiere ≥3 activos. k fijo=2.")
    else:
        k_min = 2
        k_max = min(10, n_assets)
        if "portfolio_k" not in st.session_state:
            st.session_state.portfolio_k = min(4, k_max)
        st.session_state.portfolio_k = int(np.clip(st.session_state.portfolio_k, k_min, k_max))
        portfolio_k = st.slider(
            "Clusters sugeridos (k)",
            min_value=k_min,
            max_value=k_max,
            value=st.session_state.portfolio_k,
            key="portfolio_k",
        )

# Defaults si SIMPLE
if SIMPLE:
    trend_win = 200
    trend_lookback_days = 180
    roll_vol_days = 30
    roll_corr_days = 90
    top_dd = 5
    top_peaks = 10
    min_new_high = 0.002
    min_dd_event = 0.01
    use_common_for_corr = True
    portfolio_k = min(4, max(2, len(symbols_all)))

# ============================================================
# Análisis (SOLO con botón)
# ============================================================
def run_analysis(series_raw_: Dict[str, pd.DataFrame], params_: dict) -> dict:
    start_ts = pd.Timestamp(params_["start"])
    end_ts = pd.Timestamp(params_["end"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    data: Dict[str, pd.DataFrame] = {}
    for s, df in series_raw_.items():
        m = (df.index >= start_ts) & (df.index <= end_ts)
        data[s] = df.loc[m].copy()

    symbols = [s for s in sorted(data.keys()) if not data[s].empty]

    metrics_rows = []
    rets = {}
    week_rows = []

    for s in symbols:
        m = compute_metrics(
            data[s],
            trading_days=params_["trading_days"],
            trend_win=params_["trend_win"],
            trend_lookback_days=params_["trend_lookback_days"],
        )
        if m:
            m["Símbolo"] = s
            metrics_rows.append(m)

        close = data[s]["Close"].dropna()
        if close.shape[0] >= 200:
            rets[s] = np.log(close).diff()

        wk = this_week_summary(data[s], trading_days=params_["trading_days"])
        if wk:
            an = week_anomaly_scores(data[s], lookback_weeks=52) or {}
            wk.update(an)
            wk["Símbolo"] = s
            week_rows.append(wk)

    summary = pd.DataFrame(metrics_rows).set_index("Símbolo") if metrics_rows else pd.DataFrame()
    weekdf = pd.DataFrame(week_rows).set_index("Símbolo") if week_rows else pd.DataFrame()
    rets_df = pd.DataFrame(rets) if rets else pd.DataFrame()

    if not summary.empty:
        def pct_rank(x: pd.Series, asc=True) -> pd.Series:
            return x.rank(pct=True, ascending=asc)

        summary["Score"] = (
            0.35 * pct_rank(summary["Sharpe"], True) +
            0.35 * pct_rank(summary["Calmar"], True) +
            0.20 * pct_rank(summary["CAGR"], True) -
            0.10 * pct_rank(summary["Vol anual"], True)
        )

    week_sev = pd.DataFrame()
    if not weekdf.empty:
        tmp = weekdf.copy()
        tmp["Severidad"] = 0.0
        if "Z Ret semana" in tmp.columns:
            tmp["Severidad"] += tmp["Z Ret semana"].abs().fillna(0.0)
        if "Pct rango semana" in tmp.columns:
            tmp["Severidad"] += (tmp["Pct rango semana"] - 0.5).abs().fillna(0.0)
        if "Vol ratio vs MA" in tmp.columns:
            tmp["Severidad"] += (np.log(tmp["Vol ratio vs MA"]).abs()).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        week_sev = tmp.sort_values("Severidad", ascending=False)

    corr = rets_df.corr(min_periods=200) if (not rets_df.empty and rets_df.shape[1] >= 2) else pd.DataFrame()
    clusters = cluster_labels_from_corr(corr, k=params_["portfolio_k"]) if not corr.empty else None

    # Portafolio recomendado
    reco = {"selected": [], "weights": pd.Series(dtype=float), "note": ""}
    if not rets_df.empty and not summary.empty and rets_df.shape[1] >= 2:
        selected = []
        if clusters is not None and SCIPY_OK and corr.shape[0] >= 3:
            for cl in sorted(clusters.unique()):
                members = clusters[clusters == cl].index.tolist()
                cand = summary.loc[summary.index.intersection(members)].copy()
                if "Score" in cand.columns and not cand["Score"].dropna().empty:
                    selected.append(cand["Score"].sort_values(ascending=False).index[0])
                else:
                    selected.append(members[0])
        else:
            selected = summary["Score"].sort_values(ascending=False).index.tolist()[: min(5, len(summary))]

        selected = list(dict.fromkeys(selected))
        R = rets_df[selected].dropna(how="any")
        if R.shape[0] >= 200 and R.shape[1] >= 2:
            cov = shrink_cov(R.cov(), lam=0.10, jitter=1e-10)
            w = risk_parity_weights(cov).sort_values(ascending=False)
            reco["selected"] = selected
            reco["weights"] = w
            reco["note"] = "Risk Parity (long-only) sobre covarianza shrink."
        else:
            reco["selected"] = selected
            reco["note"] = "Poco traslape para pesos robustos. Sugerencia: usar rango común."

    risk_df = build_risk_table(summary)
    return {
        "data": data,
        "symbols": symbols,
        "summary": summary,
        "risk_df": risk_df,
        "weekdf": weekdf,
        "week_sev": week_sev,
        "rets_df": rets_df,
        "corr": corr,
        "clusters": clusters,
        "reco_portfolio": reco,
    }


params = {
    "start": str(start),
    "end": str(end),
    "trading_days": int(trading_days),
    "trend_win": int(trend_win),
    "trend_lookback_days": int(trend_lookback_days),
    "roll_vol_days": int(roll_vol_days),
    "roll_corr_days": int(roll_corr_days),
    "top_dd": int(top_dd),
    "top_peaks": int(top_peaks),
    "min_new_high": float(min_new_high),
    "min_dd_event": float(min_dd_event),
    "use_common_for_corr": bool(use_common_for_corr),
    "portfolio_k": int(portfolio_k),
    "symbols": tuple(symbols_all),
    "files": tuple(current_names),
    "simple_mode": bool(SIMPLE),
}

def params_hash(d: dict) -> str:
    blob = repr(sorted(d.items())).encode("utf-8")
    return hashlib.md5(blob).hexdigest()

cur_hash = params_hash(params)

st.sidebar.markdown("---")
btn_analyze_side = st.sidebar.button("▶️ Iniciar análisis", type="primary")
run_now = bool(btn_analyze_side or btn_analyze_main)

if run_now:
    with st.spinner("Analizando..."):
        st.session_state.analysis = run_analysis(series_raw, params)
        st.session_state.analysis_hash = cur_hash

if st.session_state.analysis is None:
    st.warning("Listo para analizar. Presiona **▶️ Iniciar análisis**.")
    st.stop()

if st.session_state.analysis_hash != cur_hash:
    st.warning("Cambiaste parámetros. Los resultados mostrados pueden ser anteriores. Presiona ▶️ para recalcular.")

res = st.session_state.analysis
data = res["data"]
symbols = res["symbols"]
summary = res["summary"]
risk_df = res["risk_df"]
weekdf = res["weekdf"]
week_sev = res["week_sev"]
rets_df = res["rets_df"]
reco = res.get("reco_portfolio", {"selected": [], "weights": pd.Series(dtype=float), "note": ""})

if not symbols:
    st.warning("No hay datos en el rango seleccionado. Cambia fechas y vuelve a analizar.")
    st.stop()

if use_common_for_corr and not rets_df.empty:
    rets_df_corr = rets_df.loc[(rets_df.index >= common_start) & (rets_df.index <= common_end)].copy()
else:
    rets_df_corr = rets_df

# ============================================================
# Tabs (patrón ve menos)
# ============================================================
if SIMPLE:
    tab_exec, tab_week, tab_port, tab_corr = st.tabs(["📌 Resumen Ejecutivo", "🗓️ Semana", "🧩 Portafolio", "🔗 Correlación"])
else:
    tab_exec, tab_week, tab_port, tab_corr, tab_dd, tab_gs, tab_peaks = st.tabs(
        ["📌 Resumen Ejecutivo", "🗓️ Semana", "🧩 Portafolio", "🔗 Correlación", "📉 Drawdowns", "🥇 Oro vs 🥈 Plata", "🧨 Picos vol"]
    )

with tab_exec:
    st.subheader("📌 Resumen Ejecutivo")
    st.caption("Aquí está todo lo importante: semáforo, ranking, semana y portafolio sugerido.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Activos", str(len(symbols)))
    c2.metric("Temporalidad", ", ".join(unique_labels) if unique_labels else "—")
    c3.metric("Rango", f"{start} → {end}")
    c4.metric("Overlap común", f"{common_start.date()} → {common_end.date()}")

    bullets = build_executive_insights(summary, week_sev, reco, str(start), str(end))
    st.markdown("### 🧾 Executive Brief")
    st.markdown("\n".join([f"- {b}" for b in bullets]))

    st.markdown("---")
    st.markdown("### 🚦 Semáforo de riesgo (por activo)")
    if risk_df is None or risk_df.empty:
        st.info("No hay métricas suficientes para semáforo (falta historia).")
    else:
        # Filtro rápido
        colf1, colf2 = st.columns([1, 2])
        show_only = colf1.multiselect("Mostrar", ["🔴 Alto", "🟡 Medio", "🟢 Bajo"], default=["🔴 Alto", "🟡 Medio", "🟢 Bajo"])
        tmp = risk_df.copy()
        tmp = tmp[tmp["Riesgo"].isin(show_only)]
        st.dataframe(
            tmp[["Riesgo", "Acción", "Tipo", "Vol anual", "DD actual", "MaxDD", "CAGR", "Score", "TF"]].sort_values(["Riesgo", "Score"], ascending=[True, False]),
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("### 🔥 Ranking rápido")
    if summary.empty:
        st.warning("No hay suficientes barras para calcular métricas (mínimo ~250).")
    else:
        colL, colR = st.columns(2)
        with colL:
            st.markdown("**Top volatilidad**")
            st.dataframe(
                summary.sort_values("Vol anual", ascending=False)[["Vol anual", "CAGR", "Sharpe", "Calmar", "MaxDD", "Tipo"]].head(8),
                use_container_width=True
            )
        with colR:
            st.markdown("**Top rentabilidad (Score)**")
            st.dataframe(
                summary.sort_values("Score", ascending=False)[["Score", "CAGR", "Vol anual", "Sharpe", "Calmar", "MaxDD", "Tipo"]].head(8),
                use_container_width=True
            )

        x = summary.replace([np.inf, -np.inf], np.nan).dropna(subset=["Vol anual", "CAGR"])
        if not x.empty:
            fig = px.scatter(
                x, x="Vol anual", y="CAGR", text=x.index,
                hover_data=["Sharpe", "Calmar", "MaxDD", "Tipo", "Score", "TF"],
                title="Riesgo vs Retorno (CAGR vs Vol anual)"
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🗓️ Esta semana (top severidad)")
    if week_sev is None or week_sev.empty:
        st.info("No hay suficiente data semanal en el rango actual.")
    else:
        show_cols = [c for c in ["WeekStart", "Retorno semana", "Rango semana %", "Volumen semana (suma)", "Z Ret semana", "Vol ratio vs MA", "Pct rango semana", "Severidad"] if c in week_sev.columns]
        st.dataframe(week_sev.head(10)[show_cols], use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧩 Portafolio recomendado (diversificación por correlación)")
    if not reco.get("selected"):
        st.info("No pude construir un recomendado. Revisa que haya 2+ activos con retornos suficientes.")
    else:
        st.success(f"Selección sugerida: **{', '.join(reco['selected'])}**")
        if reco.get("weights") is not None and not reco["weights"].empty:
            st.dataframe(reco["weights"].to_frame("Peso"), use_container_width=True)

        # métricas del portafolio recomendado
        pm = compute_portfolio_metrics_from_reco(rets_df_corr, reco, trading_days=trading_days)
        cpm1, cpm2, cpm3 = st.columns(3)
        cpm1.metric("Vol anual (port)", fmt_pct(pm["vol"], 1) if pd.notna(pm["vol"]) else "—")
        cpm2.metric("Sharpe (port)", f"{pm['sharpe']:.2f}" if pd.notna(pm["sharpe"]) else "—")
        cpm3.metric("MaxDD (port)", fmt_pct(pm["maxdd"], 1) if pd.notna(pm["maxdd"]) else "—")
        st.caption(reco.get("note", ""))

    st.markdown("---")
    st.markdown("### 📤 Exportar")
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)

    if not summary.empty:
        col_d1.download_button(
            "⬇️ Resumen (CSV)",
            data=summary.to_csv().encode("utf-8"),
            file_name="resumen_activos.csv",
            mime="text/csv",
        )
    if weekdf is not None and not weekdf.empty:
        col_d2.download_button(
            "⬇️ Semana (CSV)",
            data=weekdf.to_csv().encode("utf-8"),
            file_name="semana_activos.csv",
            mime="text/csv",
        )
    if reco.get("weights") is not None and not reco["weights"].empty:
        col_d3.download_button(
            "⬇️ Pesos Portafolio (CSV)",
            data=reco["weights"].to_csv().encode("utf-8"),
            file_name="pesos_portafolio.csv",
            mime="text/csv",
        )

    if REPORTLAB_OK:
        pm = compute_portfolio_metrics_from_reco(rets_df_corr, reco, trading_days=trading_days)
        pdf_bytes = make_pdf_report(
            title="MT5 Portfolio Lab — Reporte Ejecutivo",
            start=str(start),
            end=str(end),
            symbols=symbols,
            tf_labels=unique_labels,
            bullets=bullets,
            summary=summary,
            risk_df=risk_df,
            week_sev=week_sev,
            reco=reco,
            port_metrics=pm,
        )
        col_d4.download_button(
            "📄 Reporte PDF",
            data=pdf_bytes,
            file_name="reporte_ejecutivo_mt5.pdf",
            mime="application/pdf",
        )
    else:
        col_d4.button("📄 Reporte PDF", disabled=True)
        st.warning("Para PDF, agrega `reportlab` a tu requirements.txt en Streamlit Cloud.")

    st.info("Nota: MT5 suele usar **Tick Volume**. Para comparar entre activos, usa **Vol/MA** o z-score (no volumen absoluto).")


with tab_week:
    st.subheader("🗓️ Semana — detalle")
    if week_sev is None or week_sev.empty:
        st.info("No hay suficiente data semanal en el rango actual.")
    else:
        st.dataframe(week_sev, use_container_width=True)
        st.caption("Ordenado por Severidad (movimientos raros y/o volumen inusual).")


with tab_port:
    st.subheader("🧩 Portafolio")
    st.caption("Recomendado arriba. Abajo puedes construir manual o por cluster.")

    if rets_df_corr.empty or rets_df_corr.shape[1] < 2:
        st.info("Necesitas 2+ activos con retornos suficientes.")
    else:
        corrp = rets_df_corr.corr(min_periods=200)
        cols = list(corrp.columns)

        st.markdown("### ✅ Recomendado")
        if reco.get("selected"):
            st.write("Selección:", ", ".join(reco["selected"]))
            if reco.get("weights") is not None and not reco["weights"].empty:
                st.dataframe(reco["weights"].to_frame("Peso"), use_container_width=True)

        with st.expander("🧪 Builder manual (avanzado)", expanded=not SIMPLE):
            mode_sel = st.radio("Selección", ["Manual", "1 por cluster (sugerido)"], horizontal=True)
            lbl = cluster_labels_from_corr(corrp, k=portfolio_k) if (SCIPY_OK and corrp.shape[0] >= 3) else None

            if mode_sel == "Manual" or lbl is None or summary.empty:
                default_sel = cols[: min(5, len(cols))]
                selected = st.multiselect("Activos", options=cols, default=default_sel)
            else:
                pick = []
                for cl in sorted(lbl.unique()):
                    members = lbl[lbl == cl].index.tolist()
                    cand = summary.loc[summary.index.intersection(members)].copy()
                    if "Score" in cand.columns and not cand["Score"].dropna().empty:
                        pick.append(cand["Score"].sort_values(ascending=False).index[0])
                    else:
                        pick.append(members[0])
                selected = st.multiselect("Activos (auto)", options=cols, default=pick)

            if len(selected) < 2:
                st.info("Selecciona al menos 2 activos.")
            else:
                R = rets_df_corr[selected].dropna(how="any")
                if R.shape[0] < 200:
                    st.warning("Poco traslape. Usa rango común o sube CSVs con mismas fechas.")
                else:
                    cov = shrink_cov(R.cov(), lam=0.10, jitter=1e-10)
                    method = st.selectbox("Método de pesos", ["Risk Parity (long-only)", "Inverse Vol (long-only)", "Min Var (puede tener negativos)"])

                    if method.startswith("Risk Parity"):
                        w = risk_parity_weights(cov)
                    elif method.startswith("Inverse Vol"):
                        w = inverse_vol_weights(R.std(ddof=0))
                    else:
                        w = min_var_weights_unconstrained(cov)

                    st.markdown("#### Pesos")
                    wdf = w.to_frame("Peso").sort_values("Peso", ascending=False)
                    st.dataframe(wdf, use_container_width=True)

                    port_lr = (R * w).sum(axis=1)
                    port = np.exp(port_lr.cumsum())

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=port.index, y=port.values, mode="lines", name="Portfolio"))
                    fig.update_layout(title="Curva del portafolio (base 1.0)", height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)


with tab_corr:
    st.subheader("🔗 Correlación")
    if rets_df_corr.empty or rets_df_corr.shape[1] < 2:
        st.info("Para correlación necesitas 2+ símbolos con retornos suficientes.")
    else:
        corr2 = rets_df_corr.corr(min_periods=200)
        st.plotly_chart(safe_imshow(corr2, "Matriz de correlación"), use_container_width=True)

        cols = list(rets_df_corr.columns)
        if "corrA" not in st.session_state or st.session_state.corrA not in cols:
            st.session_state.corrA = cols[0]
        if "corrB" not in st.session_state or st.session_state.corrB not in cols:
            st.session_state.corrB = cols[1] if len(cols) > 1 else cols[0]

        colA, colB = st.columns(2)
        a = colA.selectbox("A", options=cols, key="corrA")
        b = colB.selectbox("B", options=cols, key="corrB")

        if a != b:
            ab = rets_df_corr[[a, b]].dropna()
            if ab.shape[0] < 50:
                st.info("Poca data alineada para rolling corr.")
            else:
                dt = infer_dt(ab.index)
                bpd = bars_per_day_from_dt(dt) or 1.0
                roll_corr_win = max(10, int(roll_corr_days * bpd))
                rc = ab[a].rolling(roll_corr_win).corr(ab[b])

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rc.index, y=rc.values, mode="lines"))
                fig.add_hline(y=0)
                fig.update_layout(title=f"Rolling Corr (~{roll_corr_days} días): {a} vs {b}", height=260, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

        if SCIPY_OK and corr2.shape[0] >= 3 and not SIMPLE:
            st.markdown("### 🧩 Clusters (ordenando matriz)")
            ord2 = cluster_order_from_corr(corr2)
            if ord2 is not None:
                ordered = corr2.iloc[ord2, ord2]
                st.plotly_chart(safe_imshow(ordered, "Correlación ordenada (clusters)"), use_container_width=True)
            lbl = cluster_labels_from_corr(corr2, k=portfolio_k)
            if lbl is not None:
                st.dataframe(lbl.to_frame(), use_container_width=True)


# ============================================================
# Tabs extra (solo Analista)
# ============================================================
if not SIMPLE:
    with tab_dd:
        st.subheader("📉 Drawdowns (detalle)")

        if "dd_sym" not in st.session_state or st.session_state.dd_sym not in symbols:
            st.session_state.dd_sym = symbols[0]
        sym = st.selectbox("Símbolo", options=symbols, key="dd_sym")

        df = data[sym]
        close = df["Close"].dropna()
        dd = underwater_curve(close)

        c1, c2, c3 = st.columns(3)
        c1.metric("Precio", fmt_num(float(close.iloc[-1])) if not close.empty else "—")
        c2.metric("MaxDD", fmt_pct(float(dd.min())) if not dd.empty else "—")
        c3.metric("DD actual", fmt_pct(float(dd.iloc[-1])) if not dd.empty else "—")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines"))
        fig.update_layout(title=f"{sym} – Precio", height=320, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines"))
        fig.update_layout(title=f"{sym} – Underwater (Drawdown)", height=240, margin=dict(l=20, r=20, t=50, b=20))
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        ev = drawdown_events(close, min_new_high=min_new_high, min_dd=min_dd_event)
        st.markdown(f"### Top {top_dd} drawdowns (filtrados)")
        st.dataframe(ev.head(top_dd) if not ev.empty else pd.DataFrame(), use_container_width=True)

    with tab_gs:
        st.subheader("🥇 Oro vs 🥈 Plata — últimos 5 años (precio + Vol/MA)")

        gold_guess = None
        silver_guess = None
        for s in symbols:
            su = s.upper()
            if gold_guess is None and ("XAU" in su or "GOLD" in su):
                gold_guess = s
            if silver_guess is None and ("XAG" in su or "SILV" in su):
                silver_guess = s

        col1, col2 = st.columns(2)
        gold_sym = col1.selectbox("Oro (XAU)", options=symbols, index=symbols.index(gold_guess) if gold_guess in symbols else 0)
        silver_sym = col2.selectbox("Plata (XAG)", options=symbols, index=symbols.index(silver_guess) if silver_guess in symbols else (1 if len(symbols) > 1 else 0))

        dfG = data[gold_sym].copy()
        dfS = data[silver_sym].copy()

        if dfG.empty or dfS.empty:
            st.warning("No hay data suficiente para alguno de los dos.")
        else:
            end_dt = min(dfG.index.max(), dfS.index.max())
            start_5y = end_dt - pd.Timedelta(days=int(365.25 * 5))
            dfG = dfG.loc[dfG.index >= start_5y]
            dfS = dfS.loc[dfS.index >= start_5y]

            join = dfG[["Close", "Volume"]].rename(columns={"Close": "G_Close", "Volume": "G_Vol"}).join(
                dfS[["Close", "Volume"]].rename(columns={"Close": "S_Close", "Volume": "S_Vol"}),
                how="inner",
            ).dropna(subset=["G_Close", "S_Close"])

            if join.shape[0] < 200:
                st.info("Poco traslape en la ventana 5y.")
            else:
                g_norm = join["G_Close"] / join["G_Close"].iloc[0]
                s_norm = join["S_Close"] / join["S_Close"].iloc[0]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=join.index, y=g_norm, mode="lines", name=gold_sym))
                fig.add_trace(go.Scatter(x=join.index, y=s_norm, mode="lines", name=silver_sym))
                fig.update_layout(title="Precio normalizado (5 años)", height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

                dt = infer_dt(join.index)
                bpd = bars_per_day_from_dt(dt) or 1.0
                win = int(max(20, 20 * bpd))
                g_vrel = join["G_Vol"] / join["G_Vol"].rolling(win).mean()
                s_vrel = join["S_Vol"] / join["S_Vol"].rolling(win).mean()

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=join.index, y=g_vrel, mode="lines", name=f"{gold_sym} Vol/MA"))
                fig.add_trace(go.Scatter(x=join.index, y=s_vrel, mode="lines", name=f"{silver_sym} Vol/MA"))
                fig.update_layout(title="Volumen relativo (TickVol): Vol/MA(≈20 días)", height=260, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

    with tab_peaks:
        st.subheader("🧨 Picos de volatilidad + drilldown")

        if "pvol_sym" not in st.session_state or st.session_state.pvol_sym not in symbols:
            st.session_state.pvol_sym = symbols[0]
        sym = st.selectbox("Activo", options=symbols, key="pvol_sym")

        df = data[sym]
        close = df["Close"].dropna()

        if close.shape[0] < 300:
            st.info("Poca historia para picos. Amplía el rango.")
        else:
            ann = ann_factor_from_index(close.index, trading_days=trading_days)
            dt = infer_dt(close.index)
            bpd = bars_per_day_from_dt(dt) or 1.0
            win = int(max(10, roll_vol_days * bpd))

            roll, peaks, table = rolling_vol_peaks(close, win=win, top_n=int(top_peaks), ann=ann)
            if roll is None:
                st.info("Poca historia para picos con esa ventana.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines"))
                for d in peaks.index:
                    fig.add_vline(x=d, line_dash="dash", opacity=0.25)
                fig.update_layout(title=f"{sym} – Vol rolling (win={win} barras)", height=280, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(table, use_container_width=True)

                if not table.empty:
                    pick = st.selectbox("Evento (fecha)", options=list(table["Fecha"].astype(str)), index=0)
                    event_dt = pd.to_datetime(pick)
                    window_days = st.slider("Ventana alrededor (días)", 1, 30, 7)
                    left = event_dt - pd.Timedelta(days=window_days)
                    right = event_dt + pd.Timedelta(days=window_days)
                    sub = df.loc[(df.index >= left) & (df.index <= right)].copy()

                    if not sub.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=sub.index, y=sub["Close"], mode="lines"))
                        fig.add_vline(x=event_dt, line_dash="dash")
                        fig.update_layout(title=f"{sym} – Precio alrededor del evento", height=280, margin=dict(l=20, r=20, t=50, b=20))
                        st.plotly_chart(fig, use_container_width=True)

                        sub_v = sub["Volume"].replace(0, np.nan)
                        vrel = sub_v / sub_v.rolling(max(10, int(5 * bpd))).mean()
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=vrel.index, y=vrel.values, mode="lines"))
                        fig.add_vline(x=event_dt, line_dash="dash")
                        fig.update_layout(title="Volumen relativo local (Vol/MA)", height=220, margin=dict(l=20, r=20, t=50, b=20))
                        st.plotly_chart(fig, use_container_width=True)
