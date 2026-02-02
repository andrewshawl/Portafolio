
# app.py
# =========================================
# Trading Lab – Streamlit Web App (ROBUSTA EN STREAMLIT CLOUD)
# - Incluye básicos + extras (EURUSD, BTCUSD, XAUUSD, USOIL, US500, etc.)
# - Descarga "bulk" para evitar rate-limit y luego fallbacks por símbolo
# - Soporta session con curl_cffi (muy recomendado para Yahoo)
# - NO se cae si Yahoo falla: muestra diagnóstico y deja usar carga manual (CSV)
# - Top N peores drawdowns por activo (peak->trough con fechas y duración)
# - Volumen, volatilidad, recorrido (rango/ATR), precio, drawdown
# - Lateral vs Tendencial (ADX + R²)
# - Correlaciones + rolling corr + clustering
# - Regímenes Calm/Mid/Stress + Stress-Calm
# - Oro vs Plata 5 años (volumen usando futuros GC=F / SI=F)
# =========================================

import warnings
warnings.filterwarnings("ignore")

import time
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import plotly.express as px
import plotly.graph_objects as go

# SciPy (clustering opcional)
try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ----------------------------
# Config app
# ----------------------------
st.set_page_config(page_title="Trading Lab", page_icon="📈", layout="wide")
st.title("📈 Trading Lab – Dashboard Multi-Activos (MT5 / Proxies)")
st.caption(
    "Fuente: Yahoo Finance (yfinance). "
    "En Streamlit Cloud a veces Yahoo rate-limitea o bloquea IPs compartidas; "
    "esta app intenta minimizar requests y trae diagnóstico si falla."
)

# ----------------------------
# Recomendación requirements (para Cloud)
# ----------------------------
with st.expander("✅ Requisitos recomendados (Streamlit Cloud)", expanded=False):
    st.code(
        "streamlit\n"
        "yfinance\n"
        "curl_cffi\n"
        "requests\n"
        "plotly\n"
        "scipy\n"
        "pandas\n"
        "numpy\n"
    )
    st.caption("Nota: yfinance a veces necesita curl_cffi para que Yahoo acepte las requests.")

# ----------------------------
# Universo (BÁSICOS + EXTRAS)
# Proxies estables:
# - US500: SPY (o ^GSPC)
# - NAS100: QQQ (o NQ=F)
# - VIX: VXX (porque ^VIX falla seguido)
# - DXY: UUP (porque DXY directo falla seguido)
# - XAU/XAG: GC=F / SI=F para OHLC+volumen
# ----------------------------
ASSETS = {
    # ===== BÁSICOS =====
    "EURUSD": {"mt5": "EURUSD", "primary": "EURUSD=X", "fallbacks": []},
    "USDJPY": {"mt5": "USDJPY", "primary": "JPY=X", "fallbacks": []},
    "XAUUSD (Oro)": {"mt5": "XAUUSD", "primary": "GC=F", "fallbacks": ["XAUUSD=X"]},
    "USOIL / WTI": {"mt5": "USOIL", "primary": "CL=F", "fallbacks": ["BZ=F"]},
    "US500 (proxy)": {"mt5": "US500", "primary": "SPY", "fallbacks": ["^GSPC", "ES=F"]},
    "BTCUSD": {"mt5": "BTCUSD", "primary": "BTC-USD", "fallbacks": []},

    # ===== EXTRAS =====
    "XAGUSD (Plata)": {"mt5": "XAGUSD", "primary": "SI=F", "fallbacks": ["XAGUSD=X"]},
    "Nasdaq100": {"mt5": "NAS100", "primary": "QQQ", "fallbacks": ["NQ=F", "^NDX"]},
    "USDCHF": {"mt5": "USDCHF", "primary": "CHF=X", "fallbacks": ["USDCHF=X"]},
    "EURGBP": {"mt5": "EURGBP", "primary": "EURGBP=X", "fallbacks": []},
    "GBPUSD": {"mt5": "GBPUSD", "primary": "GBPUSD=X", "fallbacks": []},
    "VIX (proxy)": {"mt5": "VIX", "primary": "VXX", "fallbacks": ["^VIX", "VX=F"]},
    "USDMXN": {"mt5": "USDMXN", "primary": "MXN=X", "fallbacks": ["USDMXN=X"]},
    "Ethereum": {"mt5": "ETHUSD", "primary": "ETH-USD", "fallbacks": []},
    "DXY (proxy)": {"mt5": "DXY", "primary": "UUP", "fallbacks": ["DX-Y.NYB", "DX=F", "^DXY"]},
    "Copper (Cobre)": {"mt5": "COPPER", "primary": "HG=F", "fallbacks": []},
    "AUDCAD": {"mt5": "AUDCAD", "primary": "AUDCAD=X", "fallbacks": []},
    "NZDCAD": {"mt5": "NZDCAD", "primary": "NZDCAD=X", "fallbacks": []},
}

GOLD_FUT = "GC=F"
SILV_FUT = "SI=F"

# ----------------------------
# Helpers fechas
# ----------------------------
def infer_dates(preset: str):
    today = date.today()
    if preset == "6M":  return today - timedelta(days=183), today
    if preset == "1Y":  return today - timedelta(days=365), today
    if preset == "2Y":  return today - timedelta(days=365*2), today
    if preset == "5Y":  return today - timedelta(days=int(365.25*5)), today
    if preset == "10Y": return today - timedelta(days=int(365.25*10)), today
    if preset == "MAX": return date(1990,1,1), today
    return None, None

# ----------------------------
# Session (curl_cffi si existe)
# ----------------------------
UAS = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
]

def make_session():
    # yfinance trabaja mejor con curl_cffi en muchos entornos. citeturn0search4turn0search10
    try:
        from curl_cffi import requests as crequests  # type: ignore
        s = crequests.Session()
        s.headers.update({"User-Agent": random.choice(UAS)})
        return s, "curl_cffi"
    except Exception:
        try:
            import requests
            s = requests.Session()
            s.headers.update({"User-Agent": random.choice(UAS)})
            return s, "requests"
        except Exception:
            return None, "none"

SESSION, SESSION_KIND = make_session()

# ----------------------------
# Descarga robusta (bulk + fallback)
# ----------------------------
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
    df = df.copy()
    for col in ["Open","High","Low","Close","Volume"]:
        if col not in df.columns:
            df[col] = np.nan
    df = df[["Open","High","Low","Close","Volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["Close"])
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def bulk_download(tickers: list[str], start, end):
    """
    Descarga en 1 request (lo que más ayuda contra rate-limit).
    Devuelve dict[ticker] = df
    """
    if not tickers:
        return {}

    kwargs = dict(start=start, end=end, interval="1d", auto_adjust=True, progress=False, threads=False)
    if SESSION is not None:
        kwargs["session"] = SESSION

    last_err = None
    for attempt in range(3):
        try:
            raw = yf.download(tickers, group_by="ticker", **kwargs)
            if raw is None or raw.empty:
                last_err = "empty"
                time.sleep(1.0 * (attempt+1))
                continue

            out = {}
            if isinstance(raw.columns, pd.MultiIndex):
                # columnas: (ticker, field)
                for t in tickers:
                    if t in raw.columns.get_level_values(0):
                        part = raw[t].copy()
                        out[t] = normalize_ohlcv(part)
            else:
                # single ticker
                out[tickers[0]] = normalize_ohlcv(raw)

            return out
        except Exception as e:
            last_err = str(e)
            time.sleep(1.5 * (attempt+1))
            continue

    return {"__error__": pd.DataFrame({"error":[str(last_err)]})}

def download_one(ticker: str, start, end):
    kwargs = dict(start=start, end=end, interval="1d", auto_adjust=True, progress=False, threads=False)
    if SESSION is not None:
        kwargs["session"] = SESSION

    for attempt in range(3):
        try:
            df = yf.download(ticker, **kwargs)
            df = normalize_ohlcv(df)
            return df
        except Exception:
            time.sleep(1.5*(attempt+1))
    return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

# ----------------------------
# Métricas (vol, ATR, ADX, drawdowns, etc.)
# ----------------------------
def compute_adx_atr(df: pd.DataFrame, n=14):
    if df.empty or df["Close"].dropna().empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    close = df["Close"].astype(float).copy()
    high = df["High"].astype(float).copy().fillna(close)
    low  = df["Low"].astype(float).copy().fillna(close)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm= np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean()
    minus_dm_s= pd.Series(minus_dm,index=df.index).ewm(alpha=1/n, adjust=False).mean()

    plus_di = 100*(plus_dm_s/atr.replace(0, np.nan))
    minus_di= 100*(minus_dm_s/atr.replace(0, np.nan))
    dx = 100*(plus_di - minus_di).abs()/(plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/n, adjust=False).mean()
    return adx, atr

def trend_r2(close: pd.Series):
    close = close.dropna()
    if len(close) < 40:
        return np.nan, np.nan
    y = np.log(close.values)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope*x + intercept
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return float(r2), float(slope)

def max_drawdown(close: pd.Series):
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
                dd_pct = trough_price/peak_price - 1
                events.append({
                    "Peak Date": peak_date,
                    "Trough Date": trough_date,
                    "Recovery Date": dt,
                    "Drawdown %": dd_pct,
                    "Peak Price": peak_price,
                    "Trough Price": trough_price,
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
        dd_pct = trough_price/peak_price - 1
        events.append({
            "Peak Date": peak_date,
            "Trough Date": trough_date,
            "Recovery Date": pd.NaT,
            "Drawdown %": dd_pct,
            "Peak Price": peak_price,
            "Trough Price": trough_price,
        })

    ev = pd.DataFrame(events)
    if ev.empty:
        return ev

    ev["Days Peak->Trough"] = (pd.to_datetime(ev["Trough Date"]) - pd.to_datetime(ev["Peak Date"])).dt.days
    end_dt = close.index[-1]
    rec = pd.to_datetime(ev["Recovery Date"])
    rec_filled = rec.fillna(end_dt)
    ev["Days Trough->Recovery"] = (rec_filled - pd.to_datetime(ev["Trough Date"])).dt.days
    ev["Days Total"] = (rec_filled - pd.to_datetime(ev["Peak Date"])).dt.days

    ev = ev.sort_values("Drawdown %").reset_index(drop=True)
    return ev

def compute_metrics(df: pd.DataFrame):
    if df.empty or df["Close"].dropna().empty:
        return None

    close = df["Close"].dropna()
    rets = close.pct_change().dropna()

    days = (close.index[-1] - close.index[0]).days
    years = max(days/365.25, 1e-9)

    total_ret = float(close.iloc[-1]/close.iloc[0] - 1)
    cagr = float((close.iloc[-1]/close.iloc[0])**(1/years) - 1) if close.iloc[0] > 0 else np.nan
    vol_ann = float(rets.std()*np.sqrt(252)) if len(rets) > 10 else np.nan
    sharpe = float(cagr/vol_ann) if pd.notna(vol_ann) and vol_ann > 0 else np.nan
    mdd = max_drawdown(close)

    high = df["High"].fillna(close)
    low  = df["Low"].fillna(close)
    avg_range = float(((high-low).abs()/close).replace([np.inf,-np.inf], np.nan).dropna().mean())

    adx, atr = compute_adx_atr(df, n=14)
    adx_last = float(adx.dropna().iloc[-1]) if not adx.dropna().empty else np.nan
    atr_last = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else np.nan
    atr_pct  = float(atr_last/close.iloc[-1]) if pd.notna(atr_last) and close.iloc[-1] != 0 else np.nan

    r2, _ = trend_r2(close)

    vol_s = df["Volume"].replace(0, np.nan).dropna()
    vol_mean = float(vol_s.mean()) if not vol_s.empty else np.nan
    vol_last = float(vol_s.iloc[-1]) if not vol_s.empty else np.nan

    label = "Mixto"
    if pd.notna(adx_last) and pd.notna(r2):
        if adx_last >= 25 and r2 >= 0.20:
            label = "Tendencial"
        elif adx_last <= 20 and r2 < 0.20 and abs(total_ret) < 0.08:
            label = "Lateral"

    return {
        "Precio": float(close.iloc[-1]),
        "Retorno total": total_ret,
        "CAGR": cagr,
        "Vol anualizada": vol_ann,
        "Sharpe (~rf0)": sharpe,
        "MaxDD": mdd,
        "Avg rango diario %": avg_range,
        "ATR14 %": atr_pct,
        "ADX14": adx_last,
        "R2 tendencia": float(r2) if pd.notna(r2) else np.nan,
        "Volumen prom": vol_mean,
        "Volumen último": vol_last,
        "Clasificación": label,
        "Obs": int(len(close)),
    }

def week_metrics(df: pd.DataFrame):
    if df.empty or df["Close"].dropna().empty:
        return None
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    close = df["Close"].dropna()
    high  = df["High"].fillna(close)
    low   = df["Low"].fillna(close)

    last_dt = close.index[-1]
    week_start = (last_dt - timedelta(days=last_dt.weekday()))
    w = df[df.index >= week_start]
    if w.empty or w["Close"].dropna().empty:
        w = df.tail(5)

    w_close = w["Close"].dropna()
    w_high = w["High"].fillna(w_close).dropna()
    w_low  = w["Low"].fillna(w_close).dropna()

    w_ret = float(w_close.iloc[-1]/w_close.iloc[0] - 1) if len(w_close) >= 2 else np.nan
    w_range = float((w_high.max() - w_low.min())/w_close.iloc[-1]) if len(w_close) >= 2 else np.nan
    w_vol = float(w_close.pct_change().dropna().std()*np.sqrt(252)) if len(w_close) > 3 else np.nan

    vol_s = w["Volume"].replace(0, np.nan).dropna()
    w_volsum = float(vol_s.sum()) if not vol_s.empty else np.nan

    return {
        "Semana inicio": w_close.index[0].date(),
        "Semana fin": w_close.index[-1].date(),
        "Retorno semana": w_ret,
        "Rango semana %": w_range,
        "Vol semana (ann)": w_vol,
        "Volumen semana (suma)": w_volsum,
    }

def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8-sig")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("⚙️ Configuración")

all_names = list(ASSETS.keys())
default = ["EURUSD","USDJPY","XAUUSD (Oro)","USOIL / WTI","US500 (proxy)","BTCUSD",
           "Nasdaq100","VIX (proxy)","DXY (proxy)","USDMXN","USDCHF","GBPUSD","Ethereum","Copper (Cobre)","XAGUSD (Plata)"]
default = [x for x in default if x in all_names]

selected = st.sidebar.multiselect("Activos", options=all_names, default=default)

period = st.sidebar.selectbox("Periodo", ["6M","1Y","2Y","5Y","10Y","MAX","Custom"], index=3)
if period == "Custom":
    c1, c2 = st.sidebar.columns(2)
    start_date = c1.date_input("Inicio", value=date.today()-timedelta(days=int(365.25*5)))
    end_date   = c2.date_input("Fin", value=date.today())
else:
    start_date, end_date = infer_dates(period)

min_rows = st.sidebar.slider("Mínimo de velas para contar", 10, 150, 25, step=5)
top_dd_n = st.sidebar.selectbox("Top drawdowns por activo", [3,5,10], index=1)
roll_vol_win = st.sidebar.slider("Ventana vol rolling (días)", 10, 90, 30, step=5)
roll_corr_win= st.sidebar.slider("Ventana corr rolling (días)", 30, 180, 90, step=10)

if not selected:
    st.warning("Selecciona al menos un activo.")
    st.stop()

benchmark = st.sidebar.selectbox("Benchmark para regímenes", options=selected, index=0)

# Carga manual (fallback)
st.sidebar.markdown("---")
st.sidebar.subheader("🧯 Fallback: Cargar CSV propio")
st.sidebar.caption("Si Yahoo está bloqueando, sube un CSV con columnas: date,symbol,open,high,low,close,volume")
uploaded = st.sidebar.file_uploader("CSV (opcional)", type=["csv"])

# ----------------------------
# Cargar datos
# ----------------------------
data = {}     # asset_name -> OHLCV df
used = {}     # asset_name -> ticker usado
meta_rows = []

# 1) Si hay CSV, cargarlo y usarlo como fuente principal
if uploaded is not None:
    try:
        u = pd.read_csv(uploaded)
        u.columns = [c.strip().lower() for c in u.columns]
        needed = {"date","symbol","open","high","low","close"}
        if not needed.issubset(set(u.columns)):
            st.sidebar.error("CSV inválido. Necesita columnas: date,symbol,open,high,low,close (volume opcional).")
        else:
            u["date"] = pd.to_datetime(u["date"]).dt.tz_localize(None)
            if "volume" not in u.columns:
                u["volume"] = np.nan
            # filtro periodo
            u = u[(u["date"] >= pd.to_datetime(start_date)) & (u["date"] <= pd.to_datetime(end_date))]
            for a in selected:
                sym = ASSETS[a]["mt5"]
                part = u[u["symbol"].astype(str).str.upper() == sym.upper()].copy()
                if part.empty:
                    continue
                part = part.sort_values("date")
                part = part.set_index("date")[["open","high","low","close","volume"]]
                part.columns = ["Open","High","Low","Close","Volume"]
                part = normalize_ohlcv(part)
                if len(part) > 0:
                    data[a] = part
                    used[a] = f"CSV:{sym}"
    except Exception as e:
        st.sidebar.error(f"No pude leer el CSV: {e}")

# 2) Completar lo que falte con Yahoo (bulk + fallbacks)
need_yahoo = [a for a in selected if a not in data]

if need_yahoo:
    st.info(f"Descargando Yahoo (modo bulk, session={SESSION_KIND})…")
    primary_tickers = []
    asset_by_ticker = {}
    for a in need_yahoo:
        t = ASSETS[a]["primary"]
        primary_tickers.append(t)
        asset_by_ticker[t] = a

    bulk = bulk_download(primary_tickers, start_date, end_date)

    # si bulk devolvió error
    if "__error__" in bulk:
        st.error(
            "Yahoo no respondió (rate-limit/bloqueo). "
            "Sugerencias: 1) agrega curl_cffi en requirements, 2) recarga en 5-10 min, 3) usa el CSV fallback."
        )
    else:
        for t, df in bulk.items():
            a = asset_by_ticker.get(t)
            if a is None:
                continue
            df = normalize_ohlcv(df)
            if len(df) >= min_rows:
                data[a] = df
                used[a] = t

    # fallbacks por símbolo (solo para los que faltan)
    still = [a for a in need_yahoo if a not in data]
    if still:
        st.warning("Algunos tickers primarios no bajaron; probando fallbacks…")
        for a in still:
            ok = False
            for t in ASSETS[a]["fallbacks"]:
                df = download_one(t, start_date, end_date)
                if len(df) >= min_rows:
                    data[a] = df
                    used[a] = t
                    ok = True
                    break
            if not ok:
                data[a] = pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
                used[a] = "N/A"

# Meta table
for a in selected:
    df = data.get(a, pd.DataFrame())
    meta_rows.append({
        "Activo": a,
        "MT5": ASSETS[a]["mt5"],
        "Fuente": used.get(a, "N/A"),
        "Filas": int(len(df)) if df is not None else 0,
        "Última fecha": (df.index.max().date().isoformat() if df is not None and not df.empty else "—")
    })

meta_df = pd.DataFrame(meta_rows)

ok_assets = [a for a in selected if a in data and data[a] is not None and len(data[a]) >= min_rows and not data[a].empty]
failed_assets = [a for a in selected if a not in ok_assets]

# Mostrar estado
st.subheader("Estado de descarga")
st.dataframe(meta_df, use_container_width=True)

if failed_assets:
    st.warning("Estos activos no tuvieron data suficiente (pero la app sigue):")
    st.write(", ".join(failed_assets))

if len(ok_assets) == 0:
    st.error(
        "No bajó data suficiente de Yahoo (probable rate-limit/bloqueo). "
        "Solución rápida: usa el CSV fallback en el sidebar o vuelve a intentar más tarde."
    )
    st.stop()

if benchmark not in ok_assets:
    benchmark = ok_assets[0]
    st.info(f"Benchmark sin data suficiente; uso {benchmark}.")

# ----------------------------
# Métricas y returns
# ----------------------------
rows = []
week_rows = []
rets_cols = {}

for a in ok_assets:
    df = data[a]
    m = compute_metrics(df)
    if m is not None:
        m["Activo"] = a
        m["MT5"] = ASSETS[a]["mt5"]
        m["Fuente"] = used.get(a, "N/A")
        rows.append(m)

    w = week_metrics(df)
    if w is not None:
        w["Activo"] = a
        week_rows.append(w)

    close = df["Close"].dropna()
    if len(close) >= 30:
        rets_cols[a] = close.pct_change()

summary = pd.DataFrame(rows).set_index("Activo").sort_values("Vol anualizada", ascending=False)
weekdf = pd.DataFrame(week_rows).set_index("Activo") if week_rows else pd.DataFrame()
rets_df = pd.DataFrame(rets_cols)

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "✅ Resumen",
    "🔍 Detalle + Drawdowns",
    "🔗 Correlaciones",
    "🌡️ Regímenes",
    "🥇 Oro vs Plata"
])

with tab1:
    st.subheader("Resumen por activo")
    try:
        st.dataframe(
            summary.style.format({
                "Precio":"{:,.4f}",
                "Retorno total":"{:.2%}",
                "CAGR":"{:.2%}",
                "Vol anualizada":"{:.2%}",
                "Sharpe (~rf0)":"{:,.2f}",
                "MaxDD":"{:.2%}",
                "Avg rango diario %":"{:.2%}",
                "ATR14 %":"{:.2%}",
                "ADX14":"{:,.2f}",
                "R2 tendencia":"{:,.2f}",
                "Volumen prom":"{:,.0f}",
                "Volumen último":"{:,.0f}",
                "Obs":"{:,.0f}",
            }),
            use_container_width=True
        )
    except Exception:
        st.dataframe(summary, use_container_width=True)

    st.markdown("### Rankings rápidos")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("🔥 Más volátiles")
        st.dataframe(summary["Vol anualizada"].sort_values(ascending=False).head(7).to_frame("Vol"), use_container_width=True)
    with c2:
        st.write("💰 Más 'rentables' (CAGR)")
        st.dataframe(summary["CAGR"].sort_values(ascending=False).head(7).to_frame("CAGR"), use_container_width=True)
    with c3:
        st.write("⚖️ Mejor Sharpe (~rf0)")
        st.dataframe(summary["Sharpe (~rf0)"].sort_values(ascending=False).head(7).to_frame("Sharpe"), use_container_width=True)

    st.markdown("### Lateral vs Tendencial (ADX + R²)")
    cls = summary[["Clasificación","ADX14","R2 tendencia","Retorno total","Vol anualizada","ATR14 %","Avg rango diario %","Fuente"]].sort_values(
        ["Clasificación","ADX14"], ascending=[True, False]
    )
    st.dataframe(cls, use_container_width=True)

    st.markdown("### ¿Qué pasó esta semana?")
    if not weekdf.empty:
        try:
            st.dataframe(weekdf.style.format({
                "Retorno semana":"{:.2%}",
                "Rango semana %":"{:.2%}",
                "Vol semana (ann)":"{:.2%}",
                "Volumen semana (suma)":"{:,.0f}",
            }), use_container_width=True)
        except Exception:
            st.dataframe(weekdf, use_container_width=True)
    else:
        st.info("No hay suficiente data para tabla semanal.")

    st.download_button("⬇️ Descargar resumen (CSV)", data=csv_bytes(summary), file_name="summary.csv", mime="text/csv")
    if not weekdf.empty:
        st.download_button("⬇️ Descargar semanal (CSV)", data=csv_bytes(weekdf), file_name="weekly.csv", mime="text/csv")

with tab2:
    st.subheader("Detalle por activo + Top drawdowns (peak → trough)")
    asset = st.selectbox("Elige un activo", options=list(summary.index), index=0)
    df = data[asset]
    close = df["Close"].dropna()

    m = summary.loc[asset]
    a1,a2,a3,a4,a5,a6 = st.columns(6)
    a1.metric("Precio", f"{m['Precio']:,.4f}")
    a2.metric("Retorno total", f"{m['Retorno total']*100:,.2f}%")
    a3.metric("Vol anual", f"{m['Vol anualizada']*100:,.2f}%" if pd.notna(m["Vol anualizada"]) else "—")
    a4.metric("MaxDD", f"{m['MaxDD']*100:,.2f}%")
    a5.metric("ADX14", f"{m['ADX14']:,.1f}" if pd.notna(m["ADX14"]) else "—")
    a6.metric("Clasificación", str(m["Clasificación"]))

    st.caption(f"MT5: {ASSETS[asset]['mt5']} | Fuente: {m['Fuente']} | Obs: {m['Obs']}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines"))
    fig.update_layout(title=f"{asset} – Precio", height=360, margin=dict(l=20,r=20,t=50,b=20))
    st.plotly_chart(fig, use_container_width=True)

    ret = close.pct_change().dropna()
    if len(ret) > roll_vol_win + 10:
        rv = ret.rolling(roll_vol_win).std()*np.sqrt(252)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rv.index, y=rv.values, mode="lines"))
        fig.update_layout(title=f"{asset} – Vol rolling {roll_vol_win}d (ann)", height=280, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

    dd = close/close.cummax()-1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines"))
    fig.update_layout(title=f"{asset} – Drawdown (desde máximos)", height=280, margin=dict(l=20,r=20,t=50,b=20))
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    ev = drawdown_events(close)
    st.markdown(f"### 🧨 Top {top_dd_n} peores drawdowns históricos")
    if ev.empty:
        st.info("No se detectaron eventos de drawdown (data corta o muy estable).")
    else:
        top_ev = ev.head(top_dd_n)[["Peak Date","Trough Date","Recovery Date","Drawdown %","Days Peak->Trough","Days Trough->Recovery","Days Total","Peak Price","Trough Price"]]
        st.dataframe(top_ev, use_container_width=True)
        st.download_button("⬇️ Descargar drawdowns (CSV)", data=csv_bytes(ev), file_name=f"drawdowns_{asset}.csv", mime="text/csv")

with tab3:
    st.subheader("Correlaciones + Rolling Corr")
    valid_cols = [c for c in rets_df.columns if rets_df[c].dropna().shape[0] >= 60]
    if len(valid_cols) < 2:
        st.info("Necesitas al menos 2 activos con retornos suficientes para correlación.")
    else:
        r = rets_df[valid_cols].copy()
        corr = r.corr(min_periods=60)
        st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", title="Matriz de correlación"), use_container_width=True)

        colA, colB = st.columns(2)
        a = colA.selectbox("Activo A", options=valid_cols, index=0)
        b = colB.selectbox("Activo B", options=valid_cols, index=1 if len(valid_cols)>1 else 0)
        if a != b:
            ab = pd.concat([r[a], r[b]], axis=1).dropna()
            if len(ab) < roll_corr_win + 60:
                st.info("Muy pocos datos alineados para esa ventana de rolling corr. Baja la ventana o sube periodo.")
            else:
                rc = ab[a].rolling(roll_corr_win).corr(ab[b])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rc.index, y=rc.values, mode="lines"))
                fig.add_hline(y=0)
                fig.update_layout(title=f"Rolling Corr ({roll_corr_win}d): {a} vs {b}", height=300, margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Clustering (opcional)")
        if SCIPY_OK and len(valid_cols) >= 3:
            corr2 = corr.fillna(0)
            dist = np.sqrt(0.5*(1-corr2))
            dist_cond = squareform(dist.values, checks=False)
            Z = linkage(dist_cond, method="average")

            import matplotlib.pyplot as plt
            fig2 = plt.figure(figsize=(12,4))
            dendrogram(Z, labels=corr2.columns)
            plt.title("Clustering jerárquico (distancia por correlación)")
            st.pyplot(fig2, clear_figure=True)
        else:
            st.info("Clustering requiere SciPy y al menos 3 activos con data suficiente.")

with tab4:
    st.subheader("Regímenes Calm/Mid/Stress (según vol del benchmark)")
    if benchmark not in rets_df.columns or rets_df[benchmark].dropna().shape[0] < 180:
        st.info("Benchmark no tiene suficiente historia para regímenes. Usa 2Y/5Y o cambia benchmark.")
    else:
        bench = rets_df[benchmark].dropna()
        win = 30
        vol = bench.rolling(win).std().dropna()
        if len(vol) < 180:
            st.info("No hay suficiente historia para separar regímenes con esa ventana.")
        else:
            q_low, q_high = vol.quantile([0.40, 0.70])
            reg = pd.Series(index=vol.index, dtype="object")
            reg[vol < q_low] = "Calm"
            reg[(vol >= q_low) & (vol < q_high)] = "Mid"
            reg[vol >= q_high] = "Stress"

            st.dataframe(reg.value_counts().to_frame("días"), use_container_width=True)

            valid_cols = [c for c in rets_df.columns if rets_df[c].dropna().shape[0] >= 120]
            if len(valid_cols) < 2:
                st.info("No hay suficientes activos con retornos para correlación por régimen.")
            else:
                r = rets_df[valid_cols].copy()
                calm_idx = reg[reg=="Calm"].index
                stress_idx = reg[reg=="Stress"].index
                r_calm = r.loc[r.index.intersection(calm_idx)].dropna(how="all")
                r_stress = r.loc[r.index.intersection(stress_idx)].dropna(how="all")

                if r_calm.shape[0] < 60 or r_stress.shape[0] < 60:
                    st.info("Muy pocos días Calm/Stress. Usa 5Y o baja min_rows.")
                else:
                    c_calm = r_calm.corr(min_periods=60)
                    c_stress = r_stress.corr(min_periods=60)
                    diff = (c_stress - c_calm).fillna(0)
                    st.plotly_chart(px.imshow(diff, text_auto=".2f", aspect="auto", title="Cambio de correlación: Stress - Calm"), use_container_width=True)
                    st.caption("Rojo = en estrés se alinean más. Azul = en estrés se desacoplan.")

with tab5:
    st.subheader("Oro vs Plata (5 años) – precio + volumen (futuros GC=F / SI=F)")
    end = date.today()
    start = end - timedelta(days=int(365.25*5)+10)

    # Descarga directa (single) para no complicar
    g = download_one(GOLD_FUT, start, end)
    s = download_one(SILV_FUT, start, end)

    if g.empty or s.empty or g["Close"].dropna().empty or s["Close"].dropna().empty:
        st.warning("No pude descargar GC=F o SI=F. (Yahoo bloqueado o rate-limit).")
    else:
        df = pd.DataFrame({"Gold": g["Close"].dropna(), "Silver": s["Close"].dropna()}).dropna()
        norm = df/df.iloc[0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=norm.index, y=norm["Gold"], mode="lines", name="Gold (norm)"))
        fig.add_trace(go.Scatter(x=norm.index, y=norm["Silver"], mode="lines", name="Silver (norm)"))
        fig.update_layout(title="Precio normalizado (5 años)", height=340, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

        ratio = df["Gold"]/df["Silver"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ratio.index, y=ratio.values, mode="lines", name="Gold/Silver"))
        fig.update_layout(title="Ratio Oro/Plata", height=260, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

        gv = g["Volume"].replace(0, np.nan).dropna()
        sv = s["Volume"].replace(0, np.nan).dropna()
        fig = go.Figure()
        if not gv.empty:
            fig.add_trace(go.Scatter(x=gv.index, y=gv.rolling(20).mean(), mode="lines", name="Gold Vol MA20"))
        if not sv.empty:
            fig.add_trace(go.Scatter(x=sv.index, y=sv.rolling(20).mean(), mode="lines", name="Silver Vol MA20"))
        fig.update_layout(title="Volumen MA20 (futuros)", height=260, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Disclaimer: análisis estadístico, no asesoría financiera.")
