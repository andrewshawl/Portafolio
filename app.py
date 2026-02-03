#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MT5 Portfolio Lab — CSV-only (SIN resampling de temporalidad)
=============================================================
Objetivo: que el equipo suba todos los CSV en la MISMA temporalidad (1m/5m/15m/1h/etc)
y el análisis se ejecuta SOLO cuando el usuario presiona "▶️ Iniciar / Recalcular análisis".

Incluye:
- Loader estable (igual al analyzer): detecta <DATE>, soporta TAB/espacios, UTF-16/UTF-8, CDMX tz-naive
- Validación: misma temporalidad (estimada) + rango común de fechas
- Ranking: volatilidad vs rentabilidad (Sharpe/Calmar/CAGR) + percentiles + score
- Lateral vs Tendencial: ADX + R² (último + % del tiempo en lookback)
- Drawdowns: underwater curve + Top N eventos peak→trough→recovery + filtros anti-micro-peaks
- Semana: retorno/rango/volumen + “qué tan raro” (z-score vs historial semanal)
- Oro vs Plata (últimos 5 años): precio normalizado + volumen relativo (Vol/MA y z-score)
- Picos de volatilidad: top N + drilldown de ventana alrededor del evento
- Correlación + rolling corr + clustering (si SciPy)
- Builder de portafolio: selección 1 por cluster + pesos (risk parity / inverse vol / min-var unconstrained)
"""

from __future__ import annotations

import io
import re
import hashlib
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

# ---------------- UI ----------------
st.set_page_config(page_title="MT5 Portfolio Lab", page_icon="📈", layout="wide")
st.title("📈 MT5 Portfolio Lab (MT5 CSV only) — sin resample")
st.caption("El análisis corre solo cuando le picas ▶️ Iniciar. Sin Yahoo. CSV MT5. Timezone CDMX estable.")

def do_rerun():
    # compat: st.rerun() (nuevo) vs st.experimental_rerun() (viejo)
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

TZ_CDMX = pytz.timezone("America/Mexico_City")

# ============================================================
# Loader MT5 (igual al analyzer)
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

    # ✅ key fix: convertir a CDMX y volverlo tz-naive (manteniendo hora local)
    dt_cdmx_naive = df["datetime_utc"].dt.tz_convert(TZ_CDMX).dt.tz_localize(None)

    df = df.assign(datetime_cdmx=dt_cdmx_naive).sort_values("datetime_cdmx")
    df["range_pts"] = df["High"] - df["Low"]

    out = df[["datetime_cdmx", "Open", "High", "Low", "Close", "Volume", "range_pts"]].copy()
    out = out.drop_duplicates(subset=["datetime_cdmx"], keep="last")
    return out, info

# ============================================================
# Helpers de temporalidad / anualización
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
    cand = [("1min",60),("5min",300),("15min",900),("30min",1800),("1H",3600),("4H",14400),("1D",86400)]
    for name, s in cand:
        if abs(sec - s)/s < 0.05:
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

def to_indexed_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    x = raw.copy()
    x = x.set_index("datetime_cdmx").sort_index()
    x.index.name = "datetime"
    x = x[~x.index.duplicated(keep="last")]
    return x

# ============================================================
# Indicadores / métricas
# ============================================================
def compute_adx_atr(df: pd.DataFrame, n: int = 14) -> Tuple[pd.Series, pd.Series]:
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    close = df["Close"].astype(float)
    high  = df["High"].astype(float).fillna(close)
    low   = df["Low"].astype(float).fillna(close)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm= np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_dm_s  = pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean()

    plus_di  = 100*(plus_dm_s/atr.replace(0, np.nan))
    minus_di = 100*(minus_dm_s/atr.replace(0, np.nan))

    dx = 100*(plus_di - minus_di).abs()/(plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/n, adjust=False).mean()
    return adx, atr

def rolling_r2_from_close(close: pd.Series, win: int = 200) -> pd.Series:
    close = close.dropna()
    if len(close) < win + 5:
        return pd.Series(index=close.index, dtype=float)
    y = np.log(close.astype(float))
    x = pd.Series(np.arange(len(y)), index=y.index, dtype=float)
    r = y.rolling(win).corr(x)
    return (r**2).rename("R2")

def underwater_curve(close: pd.Series) -> pd.Series:
    close = close.dropna()
    if close.empty:
        return pd.Series(dtype=float)
    return (close / close.cummax() - 1).rename("DD")

def drawdown_events(
    close: pd.Series,
    min_new_high: float = 0.0,   # ej 0.002 = 0.2% para ignorar micro-peaks
    min_dd: float = 0.0          # ej 0.02 = solo eventos con al menos -2%
) -> pd.DataFrame:
    close = close.dropna()
    if close.empty:
        return pd.DataFrame()

    events = []
    peak_price = float(close.iloc[0])
    peak_date  = close.index[0]
    trough_price = peak_price
    trough_date  = peak_date
    in_dd = False

    for dt, price in close.iloc[1:].items():
        price = float(price)
        is_new_peak = price >= peak_price * (1.0 + float(min_new_high))

        if is_new_peak:
            if in_dd:
                ddp = trough_price/peak_price - 1.0
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
        ddp = trough_price/peak_price - 1.0
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
    trend_lookback_days: int = 180
) -> Optional[dict]:
    close = df["Close"].dropna()
    if len(close) < max(250, trend_win + 20):
        return None

    ann = ann_factor_from_index(close.index, trading_days=trading_days)

    rets = close.pct_change().dropna()
    if rets.empty:
        return None

    span_years = max((close.index[-1] - close.index[0]).total_seconds() / (365.25*86400.0), 1e-9)
    total_ret = float(close.iloc[-1]/close.iloc[0] - 1.0)
    cagr = float((close.iloc[-1]/close.iloc[0])**(1.0/span_years) - 1.0)

    mean_ann = float(rets.mean() * ann)
    vol_ann = float(rets.std(ddof=0) * np.sqrt(ann))
    sharpe = float(mean_ann/vol_ann) if vol_ann > 0 else np.nan

    dd = underwater_curve(close)
    mdd = float(dd.min()) if not dd.empty else np.nan
    calmar = float(cagr/abs(mdd)) if pd.notna(mdd) and mdd < 0 else np.nan
    dd_current = float(dd.iloc[-1]) if not dd.empty else np.nan

    avg_range_pct = float(((df["High"]-df["Low"]).abs()/df["Close"]).replace([np.inf,-np.inf], np.nan).dropna().mean())

    adx, atr = compute_adx_atr(df, 14)
    adx_last = float(adx.dropna().iloc[-1]) if not adx.dropna().empty else np.nan
    atr_last = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else np.nan
    atr_pct = float(atr_last/close.iloc[-1]) if close.iloc[-1] != 0 else np.nan

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
    r2_lb = rolling_r2_from_close(sl["Close"], win=min(trend_win, max(50, int(0.5*lookback_bars))))

    z = pd.DataFrame({"ADX": adx_lb, "R2": r2_lb}).dropna()
    if z.empty:
        tend_share = np.nan
        lat_share = np.nan
    else:
        tend = (z["ADX"] >= 25) & (z["R2"] >= 0.20)
        lat  = (z["ADX"] <= 20) & (z["R2"] < 0.20)
        tend_share = float(tend.mean())
        lat_share  = float(lat.mean())

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
        "dt": str(dt) if dt is not None else "—",
    }

def weekly_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    x = df.copy()
    idx = x.index
    week_id = (idx - pd.to_timedelta(idx.weekday, unit="D")).normalize()
    x = x.assign(_week=week_id)

    agg = x.groupby("_week").agg(
        Open=("Open","first"),
        High=("High","max"),
        Low=("Low","min"),
        Close=("Close","last"),
        Volume=("Volume","sum"),
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
    if w.empty or w.shape[0] < max(10, lookback_weeks//2):
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
    vol_ratio = float(cur["Volume"]/vol_ma) if vol_ma and pd.notna(cur["Volume"]) else np.nan

    rng_hist = hist["Rango semana %"].replace([np.inf,-np.inf], np.nan).dropna()
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
    table = pd.DataFrame({
        "Fecha": peaks.index,
        "Vol rolling (ann)": peaks.values,
        "Ret 1": ret.reindex(peaks.index).values,
        "Ret 5": close.pct_change(5).reindex(peaks.index).values,
    })
    return roll, peaks, table

# ============================================================
# Portfolio helpers
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
    order = leaves_list(Z)
    return order

def cluster_labels_from_corr(corr: pd.DataFrame, k: int = 4) -> Optional[pd.Series]:
    if not SCIPY_OK or corr.shape[0] < 3:
        return None
    dist = np.sqrt(0.5 * (1.0 - corr.fillna(0.0)))
    dist_cond = squareform(dist.values, checks=False)
    Z = linkage(dist_cond, method="average")
    labels = fcluster(Z, t=k, criterion="maxclust")
    return pd.Series(labels, index=corr.index, name="Cluster")

# ============================================================
# UI — carga de archivos (SIN análisis automático)
# ============================================================
st.sidebar.header("CSV MT5")
files = st.sidebar.file_uploader("Sube varios CSV", type=["csv","txt"], accept_multiple_files=True)

if not files:
    st.info("Sube tus CSV(s).")
    st.stop()

with st.sidebar.expander("Símbolo por archivo (opcional)", expanded=False):
    overrides = {}
    for f in files:
        guess = infer_symbol_from_filename(f.name)
        sym = st.text_input(f.name, value=guess, key=f"sym_{f.name}")
        overrides[f.name] = sym.strip().upper()

series_raw: Dict[str, pd.DataFrame] = {}
meta_rows = []

for f in files:
    sym = overrides.get(f.name, infer_symbol_from_filename(f.name))
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
    meta_rows.append({
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
    })

meta_df = pd.DataFrame(meta_rows)

st.subheader("Estado de carga (sin análisis aún)")
st.dataframe(meta_df, use_container_width=True)

if not series_raw:
    st.error("No se pudo cargar ningún símbolo.")
    st.stop()

symbols_all = sorted(series_raw.keys())

dt_labels = []
for s in symbols_all:
    dt = infer_dt(series_raw[s].index)
    dt_labels.append(timeframe_label(dt))
unique_labels = sorted(set([x for x in dt_labels if x != "—"]))

st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros (antes de correr)")
modo_estricto = st.sidebar.checkbox("Modo estricto: exigir misma temporalidad", value=True)

if modo_estricto and len(unique_labels) > 1:
    st.error(f"Temporalidades detectadas: {unique_labels}. En modo estricto deben ser iguales. "
             f"Sube CSVs en la misma temporalidad o desactiva modo estricto.")
    st.stop()

gmin = min(series_raw[s].index.min() for s in symbols_all)
gmax = max(series_raw[s].index.max() for s in symbols_all)

common_start = max(series_raw[s].index.min() for s in symbols_all)
common_end   = min(series_raw[s].index.max() for s in symbols_all)

st.sidebar.caption(f"Rango global: {gmin:%Y-%m-%d} → {gmax:%Y-%m-%d}")
st.sidebar.caption(f"Rango común:  {common_start:%Y-%m-%d} → {common_end:%Y-%m-%d}")

if "start_date" not in st.session_state:
    st.session_state.start_date = gmin.date()
if "end_date" not in st.session_state:
    st.session_state.end_date = gmax.date()

if st.sidebar.button("📌 Usar rango común"):
    st.session_state.start_date = common_start.date()
    st.session_state.end_date = common_end.date()

start = st.sidebar.date_input("Inicio", value=st.session_state.start_date)
end   = st.sidebar.date_input("Fin", value=st.session_state.end_date)

trading_days = st.sidebar.selectbox("Días de trading/año (anualización)", [252, 365], index=0)
trend_win = st.sidebar.slider("Ventana R² (barras) para tendencia", 50, 600, 200)
trend_lookback_days = st.sidebar.slider("Lookback para % Tend/% Lat (días)", 30, 365, 180)

roll_vol_days = st.sidebar.slider("Ventana vol rolling (días)", 1, 180, 30)
roll_corr_days = st.sidebar.slider("Ventana rolling corr (días)", 1, 365, 90)

top_dd = st.sidebar.selectbox("Top drawdowns", [3,5,10], index=1)
top_peaks = st.sidebar.selectbox("Top picos vol", [5,10,20,30], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Drawdown (anti-ruido)")
min_new_high = st.sidebar.slider("Ignorar micro-peaks: nuevo high mínimo (%)", 0.0, 1.0, 0.20, 0.05) / 100.0
min_dd_event = st.sidebar.slider("Solo eventos DD más grandes que (%)", 0.0, 20.0, 1.0, 0.5) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("Correlación / Portafolio")
use_common_for_corr = st.sidebar.checkbox("Para correlación/portafolio usar solo rango común", value=True)
portfolio_k = st.sidebar.slider(
    "Clusters sugeridos (k)",
    2,
    min(10, max(2, len(symbols_all))),
    min(4, max(2, len(symbols_all)))
)

# ============================================================
# Botones: correr / reset (clave)
# ============================================================
if "analysis" not in st.session_state:
    st.session_state.analysis = None
    st.session_state.analysis_params_hash = None

def params_hash(d: dict) -> str:
    blob = repr(sorted(d.items())).encode("utf-8")
    return hashlib.md5(blob).hexdigest()

params = {
    "start": str(start),
    "end": str(end),
    "trading_days": trading_days,
    "trend_win": trend_win,
    "trend_lookback_days": trend_lookback_days,
    "roll_vol_days": roll_vol_days,
    "roll_corr_days": roll_corr_days,
    "top_dd": top_dd,
    "top_peaks": int(top_peaks),
    "min_new_high": float(min_new_high),
    "min_dd_event": float(min_dd_event),
    "use_common_for_corr": bool(use_common_for_corr),
    "portfolio_k": int(portfolio_k),
    "symbols": tuple(symbols_all),
}

col_run, col_reset = st.sidebar.columns(2)
run_now = col_run.button("▶️ Iniciar / Recalcular", type="primary")
reset = col_reset.button("🧹 Reset")

if reset:
    st.session_state.analysis = None
    st.session_state.analysis_params_hash = None
    do_rerun()

cur_hash = params_hash(params)
if st.session_state.analysis is not None and st.session_state.analysis_params_hash != cur_hash:
    st.sidebar.warning("Hay cambios pendientes. Presiona ▶️ para recalcular.")

# ============================================================
# Ejecutar análisis SOLO con botón
# ============================================================
def run_analysis(series_raw: Dict[str, pd.DataFrame], params: dict) -> dict:
    start_ts = pd.Timestamp(params["start"])
    end_ts = pd.Timestamp(params["end"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    data: Dict[str, pd.DataFrame] = {}
    for s, df in series_raw.items():
        m = (df.index >= start_ts) & (df.index <= end_ts)
        data[s] = df.loc[m].copy()

    symbols = [s for s in sorted(data.keys()) if not data[s].empty]

    metrics_rows = []
    rets = {}
    week_rows = []

    for s in symbols:
        m = compute_metrics(
            data[s],
            trading_days=params["trading_days"],
            trend_win=params["trend_win"],
            trend_lookback_days=params["trend_lookback_days"]
        )
        if m:
            m["Símbolo"] = s
            metrics_rows.append(m)

        close = data[s]["Close"].dropna()
        if close.shape[0] >= 200:
            rets[s] = np.log(close).diff()  # log returns

        wk = this_week_summary(data[s], trading_days=params["trading_days"])
        if wk:
            an = week_anomaly_scores(data[s], lookback_weeks=52) or {}
            wk.update(an)
            wk["Símbolo"] = s
            week_rows.append(wk)

    summary = pd.DataFrame(metrics_rows).set_index("Símbolo") if metrics_rows else pd.DataFrame()
    weekdf = pd.DataFrame(week_rows).set_index("Símbolo") if week_rows else pd.DataFrame()
    rets_df = pd.DataFrame(rets) if rets else pd.DataFrame()

    if not summary.empty:
        def pct_rank(s: pd.Series, ascending=True) -> pd.Series:
            x = s.copy()
            return x.rank(pct=True, ascending=ascending)

        score = (
            0.35 * pct_rank(summary["Sharpe"], ascending=True) +
            0.35 * pct_rank(summary["Calmar"], ascending=True) +
            0.20 * pct_rank(summary["CAGR"], ascending=True) -
            0.10 * pct_rank(summary["Vol anual"], ascending=True)
        )
        summary["Score"] = score

        for col in ["CAGR","Vol anual","Sharpe","Calmar","MaxDD","ATR14%"]:
            if col in summary.columns:
                summary[col + " pct"] = summary[col].rank(pct=True, ascending=True)

    corr = rets_df.corr(min_periods=200) if not rets_df.empty and rets_df.shape[1] >= 2 else pd.DataFrame()
    order = cluster_order_from_corr(corr) if not corr.empty else None
    clusters = cluster_labels_from_corr(corr, k=params["portfolio_k"]) if not corr.empty else None

    return {
        "data": data,
        "symbols": symbols,
        "summary": summary,
        "weekdf": weekdf,
        "rets_df": rets_df,
        "corr": corr,
        "order": order,
        "clusters": clusters,
        "common_start": common_start,
        "common_end": common_end,
    }

if run_now:
    with st.spinner("Analizando..."):
        st.session_state.analysis = run_analysis(series_raw, params)
        st.session_state.analysis_params_hash = cur_hash

if st.session_state.analysis is None:
    st.info("Configura fechas/parámetros y presiona **▶️ Iniciar / Recalcular**.")
    st.stop()

res = st.session_state.analysis
data = res["data"]
symbols = res["symbols"]
summary = res["summary"]
weekdf = res["weekdf"]
rets_df = res["rets_df"]

if not symbols:
    st.warning("No hay datos en el rango seleccionado. Cambia Inicio/Fin y recalcula.")
    st.stop()

if use_common_for_corr and not rets_df.empty:
    cs = res["common_start"]
    ce = res["common_end"]
    rets_df_corr = rets_df.loc[(rets_df.index >= cs) & (rets_df.index <= ce)].copy()
else:
    rets_df_corr = rets_df

# ============================================================
# Tabs principales
# ============================================================
tabA, tabB, tabC, tabD, tabE, tabF, tabG = st.tabs([
    "🏁 Dashboard", "📉 Drawdowns + Tendencia", "🔗 Correlación + Clusters",
    "🥇 Oro vs 🥈 Plata (5y)", "🧨 Picos de volatilidad", "🗓️ Esta semana", "🧩 Portafolio"
])

with tabA:
    st.subheader("Ranking: volatilidad vs rentabilidad")
    if summary.empty:
        st.warning("Cargó, pero faltan barras para métricas (mínimo ~250). Amplía rango o sube más historia.")
    else:
        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown("**Más volátiles (Vol anual)**")
            st.dataframe(
                summary.sort_values("Vol anual", ascending=False)[
                    ["Precio","CAGR","Vol anual","Sharpe","Calmar","MaxDD","ATR14%","Tipo","Score"]
                ].head(10),
                use_container_width=True
            )
        with col2:
            st.markdown("**Más rentables (Score)**")
            st.dataframe(
                summary.sort_values("Score", ascending=False)[
                    ["Precio","CAGR","Vol anual","Sharpe","Calmar","MaxDD","ATR14%","Tipo","Score"]
                ].head(10),
                use_container_width=True
            )

        x = summary.copy()
        x = x.replace([np.inf,-np.inf], np.nan).dropna(subset=["Vol anual","CAGR"])
        if not x.empty:
            fig = px.scatter(
                x, x="Vol anual", y="CAGR", text=x.index,
                hover_data=["Sharpe","Calmar","MaxDD","Tipo","ATR14%","Score"],
                title="Riesgo vs Retorno (CAGR vs Vol anual)"
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=420, margin=dict(l=20,r=20,t=60,b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Clasificación (último punto)")
        st.dataframe(
            summary[["Tipo","ADX14","R2","% Tend","% Lat"]].sort_values("% Tend", ascending=False),
            use_container_width=True
        )

with tabB:
    st.subheader("Drawdown (Underwater) + eventos peak→trough→recovery")
    sym = st.selectbox("Símbolo", options=symbols, index=0, key="dd_sym")
    df = data[sym]
    close = df["Close"].dropna()

    dd = underwater_curve(close)
    mdd = float(dd.min()) if not dd.empty else np.nan
    dd_cur = float(dd.iloc[-1]) if not dd.empty else np.nan
    peak_price = float(close.cummax().iloc[-1]) if not close.empty else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio", f"{close.iloc[-1]:.5f}" if not close.empty else "—")
    c2.metric("MaxDD (hist)", f"{mdd:.1%}" if pd.notna(mdd) else "—")
    c3.metric("DD actual", f"{dd_cur:.1%}" if pd.notna(dd_cur) else "—")
    c4.metric("Último peak", f"{peak_price:.5f}" if pd.notna(peak_price) else "—")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines", name="Close"))
    fig.update_layout(title=f"{sym} – Precio", height=320, margin=dict(l=20,r=20,t=50,b=20))
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="DD"))
    fig.update_layout(title=f"{sym} – Underwater (Drawdown)", height=240, margin=dict(l=20,r=20,t=50,b=20))
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    ev = drawdown_events(close, min_new_high=min_new_high, min_dd=min_dd_event)
    st.markdown(f"### Top {top_dd} drawdowns (filtrados)")
    if ev.empty:
        st.info("No se encontraron eventos con los filtros actuales.")
    else:
        st.dataframe(ev.head(top_dd), use_container_width=True)

    st.markdown("### Tendencia vs lateral (ADX + R² rolling)")
    adx, _atr = compute_adx_atr(df, 14)
    r2s = rolling_r2_from_close(close, win=trend_win)
    z = pd.DataFrame({"ADX": adx, "R2": r2s}).dropna()
    if z.empty:
        st.info("No hay suficientes barras para tendencia con esa ventana.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=z.index, y=z["ADX"], mode="lines", name="ADX14"))
        fig.add_hline(y=25, line_dash="dash", opacity=0.4)
        fig.add_hline(y=20, line_dash="dash", opacity=0.4)
        fig.update_layout(title="ADX14", height=220, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=z.index, y=z["R2"], mode="lines", name=f"R² rolling({trend_win})"))
        fig.add_hline(y=0.20, line_dash="dash", opacity=0.4)
        fig.update_layout(title="R² (rolling)", height=220, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

with tabC:
    st.subheader("Correlación (returns) + rolling corr + clustering")
    if rets_df_corr.empty or rets_df_corr.shape[1] < 2:
        st.info("Para correlación necesitas 2+ símbolos con retornos suficientes.")
    else:
        corr2 = rets_df_corr.corr(min_periods=200)
        st.plotly_chart(px.imshow(corr2, text_auto=".2f", aspect="auto", title="Matriz de correlación"), use_container_width=True)

        colA, colB = st.columns(2)
        a = colA.selectbox("A", options=list(rets_df_corr.columns), index=0, key="corrA")
        b = colB.selectbox("B", options=list(rets_df_corr.columns), index=1, key="corrB")

        if a != b:
            ab = rets_df_corr[[a,b]].dropna()
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
                fig.update_layout(title=f"Rolling Corr ({roll_corr_win} barras): {a} vs {b}", height=260, margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Clustering (ordenando matriz por linkage)")
        if SCIPY_OK and corr2.shape[0] >= 3:
            ord2 = cluster_order_from_corr(corr2)
            if ord2 is not None:
                ordered = corr2.iloc[ord2, ord2]
                st.plotly_chart(px.imshow(ordered, text_auto=".2f", aspect="auto", title="Correlación ordenada (clusters)"), use_container_width=True)
            lbl = cluster_labels_from_corr(corr2, k=portfolio_k)
            if lbl is not None:
                st.dataframe(lbl.to_frame(), use_container_width=True)
        else:
            st.info("Clustering requiere SciPy y 3+ activos.")

with tabD:
    st.subheader("🥇 Oro vs 🥈 Plata — últimos 5 años (precio + volumen relativo)")
    candidates = {s.upper(): s for s in symbols}
    gold_sym = None
    silver_sym = None
    for s in symbols:
        su = s.upper()
        if gold_sym is None and ("XAU" in su or "GOLD" in su):
            gold_sym = s
        if silver_sym is None and ("XAG" in su or "SILV" in su):
            silver_sym = s

    col1, col2 = st.columns(2)
    gold_sym = col1.selectbox("Oro (XAU)", options=symbols, index=symbols.index(gold_sym) if gold_sym in symbols else 0, key="gold")
    silver_sym = col2.selectbox("Plata (XAG)", options=symbols, index=symbols.index(silver_sym) if silver_sym in symbols else (1 if len(symbols)>1 else 0), key="silver")

    dfG = data[gold_sym].copy()
    dfS = data[silver_sym].copy()

    end_dt = min(dfG.index.max(), dfS.index.max())
    start_5y = end_dt - pd.Timedelta(days=int(365.25*5))
    dfG = dfG.loc[dfG.index >= start_5y]
    dfS = dfS.loc[dfS.index >= start_5y]

    if dfG.empty or dfS.empty:
        st.warning("No hay suficiente historia de uno de los dos en los últimos 5 años dentro del rango actual.")
    else:
        join = dfG[["Close","Volume"]].rename(columns={"Close":"G_Close","Volume":"G_Vol"}).join(
            dfS[["Close","Volume"]].rename(columns={"Close":"S_Close","Volume":"S_Vol"}),
            how="inner"
        ).dropna(subset=["G_Close","S_Close"])

        if join.shape[0] < 200:
            st.info("Poco traslape entre oro y plata en la ventana de 5 años.")
        else:
            g_norm = join["G_Close"]/join["G_Close"].iloc[0]
            s_norm = join["S_Close"]/join["S_Close"].iloc[0]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=join.index, y=g_norm, mode="lines", name=gold_sym))
            fig.add_trace(go.Scatter(x=join.index, y=s_norm, mode="lines", name=silver_sym))
            fig.update_layout(title="Precio normalizado (5 años)", height=300, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

            dt = infer_dt(join.index)
            bpd = bars_per_day_from_dt(dt) or 1.0
            win = int(max(20, 20*bpd))  # ~20 días
            g_vrel = join["G_Vol"] / join["G_Vol"].rolling(win).mean()
            s_vrel = join["S_Vol"] / join["S_Vol"].rolling(win).mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=join.index, y=g_vrel, mode="lines", name=f"{gold_sym} Vol/MA"))
            fig.add_trace(go.Scatter(x=join.index, y=s_vrel, mode="lines", name=f"{silver_sym} Vol/MA"))
            fig.update_layout(title="Volumen relativo (TickVol): Vol / MA(≈20 días)", height=260, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

            annG = ann_factor_from_index(join.index, trading_days=trading_days)
            retG = join["G_Close"].pct_change().dropna()
            retS = join["S_Close"].pct_change().dropna()
            winv = int(max(10, roll_vol_days*bpd))
            volG = retG.rolling(winv).std(ddof=0) * np.sqrt(annG)
            volS = retS.rolling(winv).std(ddof=0) * np.sqrt(annG)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=volG.index, y=volG.values, mode="lines", name=f"{gold_sym} Vol"))
            fig.add_trace(go.Scatter(x=volS.index, y=volS.values, mode="lines", name=f"{silver_sym} Vol"))
            fig.update_layout(title=f"Vol rolling anualizada (ventana ≈ {roll_vol_days} días)", height=260, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

with tabE:
    st.subheader("Picos de volatilidad (top) + drilldown")
    sym = st.selectbox("Activo", options=symbols, index=0, key="pvol_sym")
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
            fig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines", name="Vol rolling"))
            for d in peaks.index:
                fig.add_vline(x=d, line_dash="dash", opacity=0.25)
            fig.update_layout(title=f"{sym} – Vol rolling (win={win} barras)", height=280, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(table, use_container_width=True)

            st.markdown("### Drilldown de un evento")
            pick = st.selectbox("Evento (fecha)", options=list(table["Fecha"].astype(str)), index=0, key="event_pick")
            event_dt = pd.to_datetime(pick)
            window_days = st.slider("Ventana alrededor (días)", 1, 30, 7)
            left = event_dt - pd.Timedelta(days=window_days)
            right = event_dt + pd.Timedelta(days=window_days)
            sub = df.loc[(df.index >= left) & (df.index <= right)].copy()
            if sub.empty:
                st.info("No hay datos en esa ventana.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sub.index, y=sub["Close"], mode="lines"))
                fig.add_vline(x=event_dt, line_dash="dash")
                fig.update_layout(title=f"{sym} – Precio alrededor del evento", height=280, margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig, use_container_width=True)

                sub_v = sub["Volume"].replace(0, np.nan)
                vrel = sub_v / sub_v.rolling(max(10, int(5*bpd))).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=vrel.index, y=vrel.values, mode="lines"))
                fig.add_vline(x=event_dt, line_dash="dash")
                fig.update_layout(title="Volumen relativo local (Vol / MA)", height=220, margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig, use_container_width=True)

with tabF:
    st.subheader("Qué pasó esta semana (por activo)")
    if weekdf.empty:
        st.info("No hay suficiente data para semana en el rango actual.")
    else:
        tmp = weekdf.copy()
        tmp["Severidad"] = 0.0
        if "Z Ret semana" in tmp.columns:
            tmp["Severidad"] += tmp["Z Ret semana"].abs().fillna(0.0)
        if "Pct rango semana" in tmp.columns:
            tmp["Severidad"] += (tmp["Pct rango semana"] - 0.5).abs().fillna(0.0)
        if "Vol ratio vs MA" in tmp.columns:
            tmp["Severidad"] += (np.log(tmp["Vol ratio vs MA"]).abs()).replace([np.inf,-np.inf], 0.0).fillna(0.0)

        show_cols = [
            "WeekStart","Precio fin semana","Retorno semana","Rango semana %","Vol semana (ann)","Volumen semana (suma)",
            "Z Ret semana","Vol ratio vs MA","Pct rango semana","Severidad"
        ]
        show_cols = [c for c in show_cols if c in tmp.columns]
        st.dataframe(tmp.sort_values("Severidad", ascending=False)[show_cols], use_container_width=True)

with tabG:
    st.subheader("Portafolio por correlación (clusters) + pesos")
    if rets_df_corr.empty or rets_df_corr.shape[1] < 2:
        st.info("Necesitas 2+ activos con retornos para construir portafolio.")
    else:
        corrp = rets_df_corr.corr(min_periods=200)
        cols = list(corrp.columns)

        lbl = cluster_labels_from_corr(corrp, k=portfolio_k) if (SCIPY_OK and corrp.shape[0] >= 3) else None

        st.markdown("### Selección de activos")
        mode = st.radio("Modo selección", ["Manual", "1 por cluster (sugerido)"], horizontal=True)

        if mode == "Manual" or lbl is None or summary.empty:
            selected = st.multiselect("Activos", options=cols, default=cols[:min(5,len(cols))])
        else:
            pick = []
            for cl in sorted(lbl.unique()):
                members = lbl[lbl==cl].index.tolist()
                cand = summary.loc[summary.index.intersection(members)].copy()
                if cand.empty:
                    pick.append(members[0])
                else:
                    pick.append(cand["Score"].sort_values(ascending=False).index[0])
            selected = st.multiselect("Activos (auto)", options=cols, default=pick)

        if len(selected) < 2:
            st.info("Selecciona al menos 2 activos.")
        else:
            R = rets_df_corr[selected].dropna(how="any")
            if R.shape[0] < 200:
                st.warning("Poco traslape entre activos seleccionados. "
                           "Sugerencia: subir CSVs en mismas fechas o activar rango común.")
            else:
                cov = shrink_cov(R.cov(), lam=0.10, jitter=1e-10)

                method = st.selectbox("Método de pesos", ["Risk Parity (long-only)", "Inverse Vol (long-only)", "Min Var (puede tener negativos)"])
                if method.startswith("Risk Parity"):
                    w = risk_parity_weights(cov)
                elif method.startswith("Inverse Vol"):
                    w = inverse_vol_weights(R.std(ddof=0))
                else:
                    w = min_var_weights_unconstrained(cov)

                st.markdown("### Pesos sugeridos")
                wdf = w.to_frame("Peso").sort_values("Peso", ascending=False)
                st.dataframe(wdf, use_container_width=True)

                port_ret = (R * w).sum(axis=1)
                port = (1.0 + port_ret).cumprod()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=port.index, y=port.values, mode="lines", name="Portfolio"))
                fig.update_layout(title="Curva del portafolio (base 1.0)", height=300, margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig, use_container_width=True)

                annp = ann_factor_from_index(port.index, trading_days=trading_days)
                vol = float(port_ret.std(ddof=0) * np.sqrt(annp))
                mean = float(port_ret.mean() * annp)
                sharpe = mean/vol if vol > 0 else np.nan
                ddp = float(underwater_curve(port).min())

                c1, c2, c3 = st.columns(3)
                c1.metric("Vol anual (port)", f"{vol:.1%}" if pd.notna(vol) else "—")
                c2.metric("Sharpe (port)", f"{sharpe:.2f}" if pd.notna(sharpe) else "—")
                c3.metric("MaxDD (port)", f"{ddp:.1%}" if pd.notna(ddp) else "—")
