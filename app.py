#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MT5 Portfolio Lab (CSV-only) — versión estable (loader calcado del analyzer)
===========================================================================
Soporta export MT5 tipo:
<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
- Delimitado por TAB o por espacios
- UTF-16 / UTF-8
- Multi-símbolo (subes varios archivos)

Incluye:
- Resumen por activo: retorno, CAGR, vol, Sharpe, MaxDD, ATR%, ADX, R2, volumen (tickvol)
- Clasificación lateral vs tendencial (ADX+R2)
- Top N peores drawdowns peak->trough por activo
- Qué pasó esta semana por activo
- Correlación normal + rolling corr
- Clustering (ordenando por linkage) sin matplotlib
- Regímenes Calm/Mid/Stress (por vol rolling del benchmark) + Stress–Calm
- Eventos extremos: picos de volatilidad rolling para cualquier activo
- Comparador 2 activos: normalizado + ratio + volumen MA
"""

from __future__ import annotations
import io, re
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# clustering (sin matplotlib)
try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ---------------- UI ----------------
st.set_page_config(page_title="MT5 Portfolio Lab", page_icon="📈", layout="wide")
st.title("📈 MT5 Portfolio Lab (MT5 CSV only)")
st.caption("Loader calcado del analyzer que SÍ te parsea. Sin Yahoo. Multi-símbolo.")

TZ_CDMX = pytz.timezone("America/Mexico_City")

# ============================================================
# 1) Loader MT5 (calcado + fallback)
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
    # BOM
    if b.startswith(b"\xff\xfe") or b.startswith(b"\xfe\xff"):
        return "utf-16"
    if b.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    # heurística: bytes nulos -> utf-16
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
    # XAUUSD_M1_2024 -> XAUUSD
    sym = base.split("_")[0].upper()
    sym = re.split(r"[,\s;()\-]+", sym)[0]
    return sym.upper()

@st.cache_data(show_spinner=False)
def load_mt5_bytes(file_bytes: bytes) -> Tuple[pd.DataFrame, dict]:
    """
    Devuelve df con columnas:
    datetime_utc, datetime_cdmx, Open, High, Low, Close, Volume, range_pts
    """
    enc = detect_encoding(file_bytes)
    head = first_line_text(file_bytes, enc)
    is_date_format = head.startswith("<DATE>")

    info = {"encoding": enc, "detected_header": "<DATE>" if is_date_format else "other", "sep_used": None}

    bio = io.BytesIO(file_bytes)

    if is_date_format:
        # Intento 1 (igualito que tu analyzer): sep="\t"
        df = None
        try:
            bio.seek(0)
            df = pd.read_csv(bio, sep="\t", usecols=USECOLS, encoding=enc).rename(columns=RENAME)
            info["sep_used"] = "\\t"
        except Exception:
            df = None

        # Intento 2: whitespace (por si el export trae espacios)
        if df is None or df.empty:
            bio.seek(0)
            df = pd.read_csv(bio, sep=r"\s+", engine="python", usecols=USECOLS, encoding=enc).rename(columns=RENAME)
            info["sep_used"] = "\\s+"

        # datetime (igualito que tu analyzer)
        dt_utc = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            format="%Y.%m.%d %H:%M:%S",
            utc=True,
            errors="coerce",
        )
        df = df.assign(datetime_utc=dt_utc).dropna(subset=["datetime_utc"])

    else:
        # fallback alterno (por compatibilidad)
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

    # Convertir a CDMX (como en tu analyzer)
    dt_cdmx = df["datetime_utc"].dt.tz_convert(TZ_CDMX)
    df = df.assign(datetime_cdmx=dt_cdmx)

    # numéricos
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    df = df.sort_values("datetime_utc")
    df["range_pts"] = df["High"] - df["Low"]

    return df[["datetime_utc", "datetime_cdmx", "Open", "High", "Low", "Close", "Volume", "range_pts"]], info

# ============================================================
# 2) Resample OHLCV
# ============================================================
def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.set_index("datetime_cdmx").copy()
    out = pd.DataFrame({
        "Open":  x["Open"].resample(rule).first(),
        "High":  x["High"].resample(rule).max(),
        "Low":   x["Low"].resample(rule).min(),
        "Close": x["Close"].resample(rule).last(),
        "Volume":x["Volume"].resample(rule).sum(min_count=1),
        "range_pts": x["range_pts"].resample(rule).max(),  # rango máximo por barra (útil)
    }).dropna(subset=["Close"])
    out.index.name = "datetime_cdmx"
    return out

# ============================================================
# 3) Indicadores / métricas
# ============================================================
def ann_factor(rule: str) -> int:
    if rule == "1D": return 252
    if rule == "1H": return 252 * 24
    if rule == "15T": return 252 * 96
    if rule == "5T": return 252 * 288
    if rule == "1T": return 252 * 1440
    return 252

def compute_adx_atr(df: pd.DataFrame, n=14):
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

def trend_r2(close: pd.Series) -> float:
    close = close.dropna()
    if len(close) < 200:
        return np.nan
    y = np.log(close.values)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope*x + intercept
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return float(1 - ss_res/ss_tot) if ss_tot > 0 else np.nan

def max_drawdown(close: pd.Series) -> float:
    close = close.dropna()
    if close.empty:
        return np.nan
    dd = close/close.cummax() - 1
    return float(dd.min())

def drawdown_events(close: pd.Series) -> pd.DataFrame:
    close = close.dropna()
    if close.empty:
        return pd.DataFrame()

    events = []
    peak_price = close.iloc[0]
    peak_date  = close.index[0]
    trough_price = peak_price
    trough_date  = peak_date
    in_dd = False

    for dt, price in close.iloc[1:].items():
        if price >= peak_price:
            if in_dd:
                events.append({
                    "Peak": peak_date,
                    "Trough": trough_date,
                    "Recovery": dt,
                    "DD%": (trough_price/peak_price - 1),
                })
                in_dd = False
            peak_price = price
            peak_date  = dt
            trough_price = price
            trough_date  = dt
        else:
            if not in_dd:
                in_dd = True
                trough_price = price
                trough_date  = dt
            if price < trough_price:
                trough_price = price
                trough_date  = dt

    if in_dd:
        events.append({
            "Peak": peak_date,
            "Trough": trough_date,
            "Recovery": pd.NaT,
            "DD%": (trough_price/peak_price - 1),
        })

    ev = pd.DataFrame(events)
    if ev.empty:
        return ev
    ev["Days Peak->Trough"] = (pd.to_datetime(ev["Trough"]) - pd.to_datetime(ev["Peak"])).dt.days
    ev = ev.sort_values("DD%").reset_index(drop=True)
    return ev

def compute_metrics(df: pd.DataFrame, rule: str) -> Optional[dict]:
    close = df["Close"].dropna()
    if len(close) < 200:
        return None

    rets = close.pct_change().dropna()
    ann = ann_factor(rule)

    days = (close.index[-1] - close.index[0]).days
    years = max(days/365.25, 1e-9)

    total_ret = float(close.iloc[-1]/close.iloc[0] - 1)
    cagr = float((close.iloc[-1]/close.iloc[0])**(1/years) - 1)
    vol_ann = float(rets.std()*np.sqrt(ann))
    sharpe = float(cagr/vol_ann) if vol_ann > 0 else np.nan
    mdd = max_drawdown(close)

    avg_range = float(((df["High"]-df["Low"]).abs()/df["Close"]).replace([np.inf,-np.inf], np.nan).dropna().mean())

    adx, atr = compute_adx_atr(df, 14)
    adx_last = float(adx.dropna().iloc[-1]) if not adx.dropna().empty else np.nan
    atr_last = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else np.nan
    atr_pct  = float(atr_last/close.iloc[-1]) if close.iloc[-1] != 0 else np.nan

    r2 = trend_r2(close)

    label = "Mixto"
    if pd.notna(adx_last) and pd.notna(r2):
        if adx_last >= 25 and r2 >= 0.20:
            label = "Tendencial"
        elif adx_last <= 20 and r2 < 0.20:
            label = "Lateral"

    vol_mean = float(df["Volume"].replace(0, np.nan).dropna().mean()) if df["Volume"].notna().any() else np.nan

    return {
        "Precio": float(close.iloc[-1]),
        "Retorno total": total_ret,
        "CAGR": cagr,
        "Vol anual": vol_ann,
        "Sharpe": sharpe,
        "MaxDD": mdd,
        "Avg rango%": avg_range,
        "ATR14%": atr_pct,
        "ADX14": adx_last,
        "R2": r2,
        "Vol prom": vol_mean,
        "Tipo": label,
        "Barras": int(len(close)),
        "Desde": close.index.min(),
        "Hasta": close.index.max(),
    }

def rolling_vol_peaks(close: pd.Series, win: int, top_n: int, ann: int):
    ret = close.pct_change().dropna()
    if len(ret) < win + 50:
        return None, None, None
    roll = ret.rolling(win).std()*np.sqrt(ann)
    peaks = roll.dropna().nlargest(top_n)
    table = pd.DataFrame({
        "Fecha": peaks.index,
        "Vol rolling (ann)": peaks.values,
        "Ret 1": ret.reindex(peaks.index).values,
        "Ret 5": close.pct_change(5).reindex(peaks.index).values,
    })
    return roll, peaks, table


# ============================================================
# UI Controls
# ============================================================
st.sidebar.header("CSV MT5")
files = st.sidebar.file_uploader("Sube varios CSV", type=["csv","txt"], accept_multiple_files=True)

freq = st.sidebar.selectbox("Frecuencia (resample)", ["1min","5min","15min","1H","1D"], index=3)
rule = {"1min":"1T","5min":"5T","15min":"15T","1H":"1H","1D":"1D"}[freq]
ann = ann_factor(rule)

roll_vol_days = st.sidebar.slider("Ventana vol rolling (días)", 1, 180, 30)
roll_corr_days = st.sidebar.slider("Ventana rolling corr (días)", 1, 365, 90)
top_dd = st.sidebar.selectbox("Top drawdowns", [3,5,10], index=1)
top_peaks = st.sidebar.selectbox("Top picos vol", [5,10,20,30], index=1)

bars_per_day = {"1T":1440,"5T":288,"15T":96,"1H":24,"1D":1}[rule]
roll_vol_win = max(10, int(roll_vol_days * bars_per_day))
roll_corr_win = max(10, int(roll_corr_days * bars_per_day))
st.sidebar.caption(f"Equivalencia: vol={roll_vol_win} barras, corr={roll_corr_win} barras")

if not files:
    st.info("Sube tus CSV(s). Con este loader ya no debería existir el ‘Barras=0’.")
    st.stop()

with st.sidebar.expander("Símbolo por archivo (opcional)", expanded=False):
    overrides = {}
    for f in files:
        guess = infer_symbol_from_filename(f.name)
        sym = st.text_input(f.name, value=guess, key=f"sym_{f.name}")
        overrides[f.name] = sym.strip().upper()

# ============================================================
# Load / Resample
# ============================================================
series: Dict[str, pd.DataFrame] = {}
meta_rows = []

for f in files:
    sym = overrides.get(f.name, infer_symbol_from_filename(f.name))
    b = f.getvalue()

    raw, info = load_mt5_bytes(b)
    if raw.empty:
        meta_rows.append({"Archivo": f.name, "Símbolo": sym, "Barras": 0, "Enc": info["encoding"], "Hdr": info["detected_header"], "Sep": info["sep_used"]})
        continue

    rs = resample_ohlcv(raw, rule)
    if not rs.empty:
        series[sym] = rs if sym not in series else pd.concat([series[sym], rs]).sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]

    meta_rows.append({
        "Archivo": f.name, "Símbolo": sym,
        "Barras": int(len(rs)),
        "Desde": (rs.index.min().strftime("%Y-%m-%d %H:%M") if not rs.empty else "—"),
        "Hasta": (rs.index.max().strftime("%Y-%m-%d %H:%M") if not rs.empty else "—"),
        "Enc": info["encoding"], "Hdr": info["detected_header"], "Sep": info["sep_used"]
    })

st.subheader(f"Estado de carga (resample a {freq})")
st.dataframe(pd.DataFrame(meta_rows), use_container_width=True)

if not series:
    st.error("No se pudo cargar ningún símbolo. Si tu analyzer lo carga, aquí también debería; revisa que el archivo realmente empiece con <DATE>.")
    st.stop()

symbols = sorted(series.keys())
gmin = min(series[s].index.min() for s in symbols).date()
gmax = max(series[s].index.max() for s in symbols).date()

st.sidebar.markdown("---")
start = st.sidebar.date_input("Inicio", value=gmin)
end   = st.sidebar.date_input("Fin", value=gmax)

data = {}
for s in symbols:
    df = series[s]
    mask = (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    data[s] = df.loc[mask].copy()

symbols = [s for s in symbols if not data[s].empty]
benchmark = st.sidebar.selectbox("Benchmark (estrés)", options=symbols, index=0)

# ============================================================
# Metrics / Returns
# ============================================================
metrics_rows = []
rets = {}
week_rows = []

for s in symbols:
    m = compute_metrics(data[s], rule)
    if m:
        m["Símbolo"] = s
        metrics_rows.append(m)

    close = data[s]["Close"].dropna()
    if len(close) >= 200:
        rets[s] = close.pct_change()

    # semana
    if len(close) >= 50:
        last_dt = close.index[-1]
        week_start = (last_dt - pd.Timedelta(days=last_dt.weekday())).normalize()
        w = data[s].loc[data[s].index >= week_start]
        if not w.empty and w["Close"].dropna().shape[0] >= 2:
            w_close = w["Close"].dropna()
            week_rows.append({
                "Símbolo": s,
                "Retorno semana": float(w_close.iloc[-1]/w_close.iloc[0] - 1),
                "Rango semana %": float((w["High"].max() - w["Low"].min())/w_close.iloc[-1]),
                "Vol semana (ann)": float(w_close.pct_change().dropna().std()*np.sqrt(ann)) if w_close.shape[0] > 3 else np.nan,
                "Vol semana (suma)": float(w["Volume"].replace(0,np.nan).dropna().sum()) if w["Volume"].notna().any() else np.nan
            })

summary = pd.DataFrame(metrics_rows).set_index("Símbolo") if metrics_rows else pd.DataFrame()
weekdf  = pd.DataFrame(week_rows).set_index("Símbolo") if week_rows else pd.DataFrame()
rets_df = pd.DataFrame(rets).dropna(how="any") if rets else pd.DataFrame()

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["✅ Resumen", "🔍 Detalle + DD", "🔗 Correlaciones", "🌡 Estrés vs Calma", "🧨 Eventos + Comparador"])

with tab1:
    st.subheader("Resumen por activo")
    if summary.empty:
        st.warning("Cargó bien, pero faltan barras para métricas (mínimo ~200). Usa 1min/5min/15min o amplía el rango.")
    else:
        st.dataframe(summary.sort_values("Vol anual", ascending=False), use_container_width=True)

    st.subheader("Qué pasó esta semana")
    if weekdf.empty:
        st.info("No hay suficiente data para semana en el rango actual.")
    else:
        st.dataframe(weekdf, use_container_width=True)

with tab2:
    sym = st.selectbox("Símbolo", options=symbols, index=0)
    close = data[sym]["Close"].dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines"))
    fig.update_layout(title=f"{sym} – Precio", height=320, margin=dict(l=20,r=20,t=50,b=20))
    st.plotly_chart(fig, use_container_width=True)

    dd = close/close.cummax()-1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines"))
    fig.update_layout(title=f"{sym} – Drawdown", height=240, margin=dict(l=20,r=20,t=50,b=20))
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    ev = drawdown_events(close)
    st.markdown(f"### Top {top_dd} drawdowns (peak→trough)")
    st.dataframe(ev.head(top_dd), use_container_width=True)

with tab3:
    if rets_df.empty or rets_df.shape[1] < 2:
        st.info("Para correlación necesitas 2+ símbolos con suficientes barras (>=200).")
    else:
        corr = rets_df.corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", title="Matriz de correlación"), use_container_width=True)

        colA, colB = st.columns(2)
        a = colA.selectbox("A", options=list(rets_df.columns), index=0)
        b = colB.selectbox("B", options=list(rets_df.columns), index=1)
        if a != b:
            ab = rets_df[[a,b]].dropna()
            if len(ab) >= roll_corr_win + 50:
                rc = ab[a].rolling(roll_corr_win).corr(ab[b])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rc.index, y=rc.values, mode="lines"))
                fig.add_hline(y=0)
                fig.update_layout(title=f"Rolling Corr ({roll_corr_win} barras): {a} vs {b}", height=260, margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Poca data alineada para rolling corr con esa ventana. Baja ventana o amplía rango.")

        st.markdown("### Clustering (ordenando heatmap por linkage)")
        if SCIPY_OK and rets_df.shape[1] >= 3:
            # distancia por correlación
            dist = np.sqrt(0.5*(1 - corr.fillna(0)))
            dist_cond = squareform(dist.values, checks=False)
            Z = linkage(dist_cond, method="average")
            order = leaves_list(Z)
            ordered = corr.iloc[order, order]
            st.plotly_chart(px.imshow(ordered, text_auto=".2f", aspect="auto", title="Correlación ordenada (clusters)"), use_container_width=True)
        else:
            st.info("Clustering requiere SciPy y 3+ activos.")

with tab4:
    if rets_df.empty or benchmark not in rets_df.columns:
        st.info("No hay retornos suficientes para estrés.")
    else:
        bret = rets_df[benchmark].dropna()
        vol = bret.rolling(roll_vol_win).std().dropna()
        if len(vol) < 300:
            st.info("Poca historia para regímenes con tu ventana. Baja ventana o amplía rango.")
        else:
            q_low, q_high = vol.quantile([0.40, 0.70])
            reg = pd.Series(index=vol.index, dtype="object")
            reg[vol < q_low] = "Calm"
            reg[(vol >= q_low) & (vol < q_high)] = "Mid"
            reg[vol >= q_high] = "Stress"

            st.dataframe(reg.value_counts().to_frame("barras"), use_container_width=True)

            calm_idx = reg[reg=="Calm"].index
            stress_idx = reg[reg=="Stress"].index

            r_calm = rets_df.loc[rets_df.index.intersection(calm_idx)]
            r_stress = rets_df.loc[rets_df.index.intersection(stress_idx)]

            if len(r_calm) < 200 or len(r_stress) < 200:
                st.info("Muy pocos puntos Calm/Stress para comparar correlaciones.")
            else:
                c_calm = r_calm.corr()
                c_stress = r_stress.corr()
                diff = (c_stress - c_calm).fillna(0)
                st.plotly_chart(px.imshow(diff, text_auto=".2f", aspect="auto", title="Stress - Calm"), use_container_width=True)
                st.caption("Rojo = en estrés se alinean más. Azul = en estrés se desacoplan.")

with tab5:
    st.subheader("Picos de volatilidad")
    sym = st.selectbox("Activo", options=symbols, index=0, key="pvol")
    close = data[sym]["Close"].dropna()

    roll, peaks, table = rolling_vol_peaks(close, win=roll_vol_win, top_n=int(top_peaks), ann=ann)
    if roll is None:
        st.info("Poca historia para picos con esa ventana.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines"))
        for d in peaks.index:
            fig.add_vline(x=d, line_dash="dash", opacity=0.35)
        fig.update_layout(title=f"{sym} – Vol rolling ({roll_vol_win} barras)", height=280, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(table, use_container_width=True)

    st.markdown("---")
    st.subheader("Comparador 2 activos")
    col1, col2 = st.columns(2)
    a = col1.selectbox("A", options=symbols, index=0, key="cmpA")
    b = col2.selectbox("B", options=symbols, index=1 if len(symbols)>1 else 0, key="cmpB")

    dfA = data[a][["Close","Volume"]].rename(columns={"Close":"A_Close","Volume":"A_Vol"})
    dfB = data[b][["Close","Volume"]].rename(columns={"Close":"B_Close","Volume":"B_Vol"})
    df = dfA.join(dfB, how="inner").dropna(subset=["A_Close","B_Close"])

    if len(df) < 200:
        st.info("No hay suficiente traslape entre ambos activos en el rango actual.")
    else:
        na = df["A_Close"]/df["A_Close"].iloc[0]
        nb = df["B_Close"]/df["B_Close"].iloc[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=na, mode="lines", name=a))
        fig.add_trace(go.Scatter(x=df.index, y=nb, mode="lines", name=b))
        fig.update_layout(title="Precio normalizado", height=280, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

        ratio = df["A_Close"]/df["B_Close"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=ratio.values, mode="lines"))
        fig.update_layout(title=f"Ratio {a}/{b}", height=240, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        if df["A_Vol"].notna().any():
            fig.add_trace(go.Scatter(x=df.index, y=df["A_Vol"].rolling(20).mean(), mode="lines", name=f"{a} Vol MA20"))
        if df["B_Vol"].notna().any():
            fig.add_trace(go.Scatter(x=df.index, y=df["B_Vol"].rolling(20).mean(), mode="lines", name=f"{b} Vol MA20"))
        fig.update_layout(title="Volumen (MA20) usando TICKVOL", height=240, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)
