# app.py
# =========================================
# Trading Lab – Streamlit Web App (robusta)
# - Universo: básicos + extras (incluye EURUSD, BTC, XAUUSD, USOIL, US500, etc.)
# - Descarga robusta con proxies estables (Yahoo)
# - No se cae si fallan símbolos: sigue con los que sí
# - Top N peores drawdowns por activo (peak->trough, fechas, duración)
# - Volumen/volatilidad/recorrido/precio + lateral vs tendencial (ADX + R²)
# - Correlaciones + rolling corr + clustering
# - Regímenes Calm/Mid/Stress + Stress-Calm
# - Oro vs Plata 5 años (volumen usando futuros GC=F / SI=F)
# =========================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta, datetime

# SciPy para clustering (si no está, no pasa nada)
try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# -----------------------------------------
# CONFIG
# -----------------------------------------
st.set_page_config(page_title="Trading Lab", page_icon="📈", layout="wide")
st.title("📈 Trading Lab – Dashboard Multi-Activos (MT5 / Proxies)")
st.caption(
    "Fuente: Yahoo Finance (yfinance). "
    "Nota: en FX spot / algunos índices, el 'volumen' puede venir vacío o no ser volumen real."
)

# -----------------------------------------
# UNIVERSO (MT5 símbolos + proxies Yahoo estables)
# -----------------------------------------
# Regla práctica:
# - Índices: usa ETFs (SPY/QQQ) para OHLC confiable
# - VIX: usa VXX (ETF) porque ^VIX puede fallar según región
# - DXY: usa UUP (ETF) porque DXY a veces falla
# - Oro/Plata: para OHLC+volumen usa futuros GC=F / SI=F
# - USOIL: usa CL=F
# - FX: usa los pares =X cuando existan (o CHF=X, JPY=X, MXN=X)
ASSETS = {
    # ===== BÁSICOS =====
    "EURUSD": {"mt5": "EURUSD", "yahoo": ["EURUSD=X"]},
    "USDJPY": {"mt5": "USDJPY", "yahoo": ["JPY=X"]},
    "XAUUSD (Oro)": {"mt5": "XAUUSD", "yahoo": ["GC=F", "XAUUSD=X"]},     # preferimos futuros
    "USOIL / WTI": {"mt5": "USOIL", "yahoo": ["CL=F", "BZ=F"]},
    "US500 (proxy)": {"mt5": "US500", "yahoo": ["SPY", "^GSPC", "ES=F"]},
    "BTCUSD": {"mt5": "BTCUSD", "yahoo": ["BTC-USD"]},

    # ===== EXTRAS QUE PEDISTE =====
    "XAGUSD (Plata)": {"mt5": "XAGUSD", "yahoo": ["SI=F", "XAGUSD=X"]},   # preferimos futuros
    "Nasdaq100": {"mt5": "NAS100", "yahoo": ["QQQ", "NQ=F", "^NDX"]},
    "USDCHF": {"mt5": "USDCHF", "yahoo": ["CHF=X", "USDCHF=X"]},
    "EURGBP": {"mt5": "EURGBP", "yahoo": ["EURGBP=X"]},
    "GBPUSD": {"mt5": "GBPUSD", "yahoo": ["GBPUSD=X"]},
    "VIX (proxy)": {"mt5": "VIX", "yahoo": ["VXX", "^VIX", "VX=F"]},
    "USDMXN": {"mt5": "USDMXN", "yahoo": ["MXN=X", "USDMXN=X"]},
    "Ethereum": {"mt5": "ETHUSD", "yahoo": ["ETH-USD"]},
    "DXY (proxy)": {"mt5": "DXY", "yahoo": ["UUP", "DX-Y.NYB", "DX=F", "^DXY"]},
    "Copper (Cobre)": {"mt5": "COPPER", "yahoo": ["HG=F"]},
    "AUDCAD": {"mt5": "AUDCAD", "yahoo": ["AUDCAD=X"]},
    "NZDCAD": {"mt5": "NZDCAD", "yahoo": ["NZDCAD=X"]},
}

# Para módulo Oro/Plata con volumen
GOLD_FUT = "GC=F"
SILV_FUT = "SI=F"

# -----------------------------------------
# HELPERS
# -----------------------------------------
def _infer_dates(preset: str):
    today = date.today()
    if preset == "6M":  return today - timedelta(days=183), today
    if preset == "1Y":  return today - timedelta(days=365), today
    if preset == "2Y":  return today - timedelta(days=365*2), today
    if preset == "5Y":  return today - timedelta(days=int(365.25*5)), today
    if preset == "10Y": return today - timedelta(days=int(365.25*10)), today
    if preset == "MAX": return date(1990, 1, 1), today
    return None, None

@st.cache_data(ttl=3600, show_spinner=False)
def download_first_working(aliases, start, end, interval="1d"):
    """
    Prueba aliases en orden y devuelve (df, ticker_usado).
    df con columnas: Open, High, Low, Close, Volume
    """
    for t in aliases:
        try:
            df = yf.download(t, start=start, end=end, interval=interval, auto_adjust=True, progress=False, threads=False)
            if df is None or df.empty:
                continue
            df = df.copy()
            # normalizar columnas
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[~df.index.duplicated(keep="last")]
            df = df.dropna(subset=["Close"])
            return df, t
        except Exception:
            continue
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]), None

def compute_adx_atr(df, n=14):
    if df.empty or df["Close"].dropna().empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    close = df["Close"].astype(float).copy()
    high = df["High"].astype(float).copy().fillna(close)
    low  = df["Low"].astype(float).copy().fillna(close)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean()

    plus_di = 100 * (plus_dm_s / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm_s / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
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
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(r2), float(slope)

def max_drawdown(close: pd.Series):
    close = close.dropna()
    if close.empty:
        return np.nan
    dd = close / close.cummax() - 1
    return float(dd.min())

def drawdown_events(close: pd.Series) -> pd.DataFrame:
    """
    Eventos drawdown: peak -> trough, y recovery si existe.
    Ordenado de peor a mejor (más negativo primero).
    """
    close = close.dropna()
    if close.empty:
        return pd.DataFrame()

    events = []
    peak_price = close.iloc[0]
    peak_date = close.index[0]
    trough_price = peak_price
    trough_date = peak_date
    in_dd = False

    for dt, price in close.iloc[1:].items():
        if price >= peak_price:
            # recovery / nuevo máximo
            if in_dd:
                dd_pct = trough_price / peak_price - 1
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

    # drawdown abierto
    if in_dd:
        dd_pct = trough_price / peak_price - 1
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
    years = max(days / 365.25, 1e-9)

    total_ret = float(close.iloc[-1] / close.iloc[0] - 1)
    cagr = float((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1) if close.iloc[0] > 0 else np.nan
    vol_ann = float(rets.std() * np.sqrt(252)) if len(rets) > 10 else np.nan
    sharpe = float(cagr / vol_ann) if pd.notna(vol_ann) and vol_ann > 0 else np.nan

    mdd = max_drawdown(close)

    high = df["High"].fillna(close)
    low = df["Low"].fillna(close)
    avg_range = float(((high - low).abs() / close).replace([np.inf, -np.inf], np.nan).dropna().mean())

    adx, atr = compute_adx_atr(df, n=14)
    adx_last = float(adx.dropna().iloc[-1]) if not adx.dropna().empty else np.nan
    atr_last = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else np.nan
    atr_pct = float(atr_last / close.iloc[-1]) if pd.notna(atr_last) and close.iloc[-1] != 0 else np.nan

    r2, _ = trend_r2(close)

    vol_s = df["Volume"].replace(0, np.nan).dropna()
    vol_mean = float(vol_s.mean()) if not vol_s.empty else np.nan
    vol_last = float(vol_s.iloc[-1]) if not vol_s.empty else np.nan

    # Clasificación lateral/tendencial (regla práctica)
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
    high = df["High"].fillna(close)
    low = df["Low"].fillna(close)

    last_dt = close.index[-1]
    week_start = (last_dt - timedelta(days=last_dt.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)

    w = df[df.index >= week_start]
    if w.empty or w["Close"].dropna().empty:
        w = df.tail(5)

    w_close = w["Close"].dropna()
    w_high = w["High"].fillna(w_close).dropna()
    w_low = w["Low"].fillna(w_close).dropna()

    w_ret = float(w_close.iloc[-1] / w_close.iloc[0] - 1) if len(w_close) >= 2 else np.nan
    w_range = float((w_high.max() - w_low.min()) / w_close.iloc[-1]) if len(w_close) >= 2 else np.nan
    w_vol = float(w_close.pct_change().dropna().std() * np.sqrt(252)) if len(w_close) > 3 else np.nan

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

# -----------------------------------------
# SIDEBAR
# -----------------------------------------
st.sidebar.header("⚙️ Configuración")

all_assets = list(ASSETS.keys())

default_assets = [
    "EURUSD", "USDJPY", "XAUUSD (Oro)", "USOIL / WTI", "US500 (proxy)", "BTCUSD",
    "Nasdaq100", "USDMXN", "DXY (proxy)", "VIX (proxy)", "XAGUSD (Plata)", "Ethereum"
]
default_assets = [a for a in default_assets if a in all_assets]

selected_assets = st.sidebar.multiselect(
    "Activos",
    options=all_assets,
    default=default_assets
)

period = st.sidebar.selectbox("Periodo", ["6M", "1Y", "2Y", "5Y", "10Y", "MAX", "Custom"], index=3)
if period == "Custom":
    c1, c2 = st.sidebar.columns(2)
    start_date = c1.date_input("Inicio", value=date.today() - timedelta(days=int(365.25 * 5)))
    end_date = c2.date_input("Fin", value=date.today())
else:
    start_date, end_date = _infer_dates(period)

min_rows = st.sidebar.slider("Mínimo de velas (para contar como válido)", 10, 150, 25, step=5)
top_dd_n = st.sidebar.selectbox("Top drawdowns por activo", [3, 5, 10], index=1)
roll_vol_win = st.sidebar.slider("Ventana vol rolling (días)", 10, 90, 30, step=5)
roll_corr_win = st.sidebar.slider("Ventana corr rolling (días)", 30, 180, 90, step=10)

if not selected_assets:
    st.warning("Selecciona al menos 1 activo.")
    st.stop()

benchmark = st.sidebar.selectbox("Benchmark para regímenes", options=selected_assets, index=0)

with st.sidebar.expander("🧩 Notas rápidas"):
    st.write(
        "- US500/NAS100/VIX/DXY aquí son proxies (SPY/QQQ/VXX/UUP) para tener OHLC estable.\n"
        "- Oro/Plata se bajan como futuros (GC=F / SI=F) para que sí exista volumen.\n"
        "- Si Yahoo rate-limitea, recarga o usa un periodo más largo."
    )

# -----------------------------------------
# DESCARGA
# -----------------------------------------
with st.spinner("Bajando datos (Yahoo) y calculando..."):
    data = {}
    meta_rows = []

    for a in selected_assets:
        df, used = download_first_working(ASSETS[a]["yahoo"], start_date, end_date, interval="1d")
        data[a] = df
        meta_rows.append({
            "Activo": a,
            "MT5": ASSETS[a]["mt5"],
            "Yahoo usado": used if used else "N/A",
            "Filas": int(len(df)) if df is not None else 0
        })

meta_df = pd.DataFrame(meta_rows)

# Clasificar válidos vs fallidos
ok_assets = [a for a in selected_assets if data[a] is not None and len(data[a]) >= min_rows and not data[a].empty]
failed_assets = [a for a in selected_assets if a not in ok_assets]

if failed_assets:
    st.warning("Algunos símbolos no bajaron bien de Yahoo. Sigo con los que sí (abajo te dejo cuáles fallaron).")
    st.dataframe(meta_df[meta_df["Activo"].isin(failed_assets)], use_container_width=True)

if len(ok_assets) == 0:
    st.error("No bajó data suficiente para ningún activo. Prueba: periodo 5Y / 10Y o reduce la lista.")
    st.stop()

if benchmark not in ok_assets:
    benchmark = ok_assets[0]
    st.info(f"El benchmark elegido no tuvo data suficiente; uso {benchmark} como referencia.")

# -----------------------------------------
# MÉTRICAS
# -----------------------------------------
rows = []
week_rows = []

for a in ok_assets:
    m = compute_metrics(data[a])
    if m is None:
        continue
    m["Activo"] = a
    m["MT5"] = ASSETS[a]["mt5"]
    m["Yahoo usado"] = meta_df.loc[meta_df["Activo"] == a, "Yahoo usado"].iloc[0]
    rows.append(m)

    w = week_metrics(data[a])
    if w is not None:
        w["Activo"] = a
        week_rows.append(w)

summary = pd.DataFrame(rows).set_index("Activo") if rows else pd.DataFrame()
weekdf = pd.DataFrame(week_rows).set_index("Activo") if week_rows else pd.DataFrame()

if summary.empty:
    st.error("Se bajó data pero no se pudieron calcular métricas. Intenta con otros símbolos/periodo.")
    st.stop()

summary = summary.sort_values("Vol anualizada", ascending=False)

# returns matrix (NO dropna global para que no mate correlación)
rets_df = pd.DataFrame()
for a in ok_assets:
    close = data[a]["Close"].dropna()
    if len(close) >= 30:
        rets_df[a] = close.pct_change()

# -----------------------------------------
# TABS
# -----------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "✅ Resumen",
    "🔍 Detalle + Drawdowns",
    "🔗 Correlaciones",
    "🌡️ Regímenes",
    "🥇 Oro vs Plata"
])

# =========================
# TAB 1: Resumen
# =========================
with tab1:
    st.subheader("Mapa de símbolos (MT5 vs Yahoo)")
    st.dataframe(meta_df[meta_df["Activo"].isin(ok_assets)], use_container_width=True)

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
    cls = summary[["Clasificación","ADX14","R2 tendencia","Retorno total","Vol anualizada","ATR14 %","Avg rango diario %"]].sort_values(
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
        st.info("No se pudo armar tabla semanal con lo descargado.")

    st.markdown("### Descargar")
    st.download_button("⬇️ Descargar resumen (CSV)", data=csv_bytes(summary), file_name="summary.csv", mime="text/csv")
    if not weekdf.empty:
        st.download_button("⬇️ Descargar semanal (CSV)", data=csv_bytes(weekdf), file_name="weekly.csv", mime="text/csv")

# =========================
# TAB 2: Detalle + Drawdowns
# =========================
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

    st.caption(f"MT5: {ASSETS[asset]['mt5']} | Yahoo: {m['Yahoo usado']} | Obs: {m['Obs']}")

    # Precio
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines", name="Close"))
    fig.update_layout(title=f"{asset} – Precio", height=360, margin=dict(l=20,r=20,t=50,b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Vol rolling
    ret = close.pct_change().dropna()
    if len(ret) > roll_vol_win + 10:
        rv = ret.rolling(roll_vol_win).std()*np.sqrt(252)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rv.index, y=rv.values, mode="lines", name="Rolling Vol"))
        fig.update_layout(title=f"{asset} – Vol rolling {roll_vol_win}d (ann)", height=280, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Drawdown curve
    dd = close/close.cummax()-1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown"))
    fig.update_layout(title=f"{asset} – Drawdown (desde máximos)", height=280, margin=dict(l=20,r=20,t=50,b=20))
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    # Top drawdown events
    ev = drawdown_events(close)
    st.markdown(f"### 🧨 Top {top_dd_n} peores drawdowns históricos")
    if ev.empty:
        st.info("No se detectaron eventos de drawdown (o data muy corta).")
    else:
        top_ev = ev.head(top_dd_n).copy()
        show_ev = top_ev[["Peak Date","Trough Date","Recovery Date","Drawdown %","Days Peak->Trough","Days Trough->Recovery","Days Total","Peak Price","Trough Price"]]
        try:
            st.dataframe(show_ev.style.format({
                "Drawdown %":"{:.2%}",
                "Peak Price":"{:,.4f}",
                "Trough Price":"{:,.4f}",
            }), use_container_width=True)
        except Exception:
            st.dataframe(show_ev, use_container_width=True)

        st.download_button("⬇️ Descargar todos los drawdowns (CSV)", data=csv_bytes(ev), file_name=f"drawdowns_{asset}.csv", mime="text/csv")

    # Volumen
    vol_s = df["Volume"].replace(0, np.nan).dropna()
    st.markdown("### Volumen")
    if vol_s.empty:
        st.info("Volumen: N/A (normal en FX spot / proxies).")
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=vol_s.index, y=vol_s.values, name="Volume"))
        fig.update_layout(title=f"{asset} – Volumen", height=260, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 3: Correlaciones
# =========================
with tab3:
    st.subheader("Correlaciones + Rolling Corr + Clustering")
    # necesitamos al menos 2 series de retornos con datos
    valid_cols = [c for c in rets_df.columns if rets_df[c].dropna().shape[0] >= 30]

    if len(valid_cols) < 2:
        st.info("Para correlaciones necesitas al menos 2 activos con retornos suficientes. (Los otros tabs sí sirven).")
    else:
        r = rets_df[valid_cols].copy()
        corr = r.corr(min_periods=60)

        st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", title="Matriz de correlación (pairwise)"), use_container_width=True)

        st.markdown("### Rolling correlation (elige par)")
        colA, colB = st.columns(2)
        a = colA.selectbox("Activo A", options=valid_cols, index=0)
        b = colB.selectbox("Activo B", options=valid_cols, index=1 if len(valid_cols) > 1 else 0)

        if a != b:
            ab = pd.concat([r[a], r[b]], axis=1).dropna()
            if len(ab) < roll_corr_win + 30:
                st.info("Muy pocos datos alineados para rolling corr con esa ventana. Baja la ventana o usa periodo más largo.")
            else:
                rc = ab[a].rolling(roll_corr_win).corr(ab[b])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rc.index, y=rc.values, mode="lines"))
                fig.add_hline(y=0)
                fig.update_layout(title=f"Rolling Corr ({roll_corr_win}d): {a} vs {b}", height=300, margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Clustering (opcional)")
        if SCIPY_OK and len(valid_cols) >= 3:
            corr2 = r.corr(min_periods=60).fillna(0)
            dist = np.sqrt(0.5*(1-corr2))
            dist_cond = squareform(dist.values, checks=False)
            Z = linkage(dist_cond, method="average")

            import matplotlib.pyplot as plt
            fig2 = plt.figure(figsize=(12,4))
            dendrogram(Z, labels=corr2.columns)
            plt.title("Clustering jerárquico (distancia por correlación)")
            st.pyplot(fig2, clear_figure=True)
        else:
            st.info("Clustering requiere SciPy y al menos 3 activos con retornos suficientes.")

# =========================
# TAB 4: Regímenes
# =========================
with tab4:
    st.subheader("Regímenes Calm / Mid / Stress (según vol del benchmark)")

    if benchmark not in rets_df.columns or rets_df[benchmark].dropna().shape[0] < 120:
        st.info("Benchmark no tiene historia suficiente para regímenes. Usa periodo 2Y/5Y o cambia benchmark.")
    else:
        bench = rets_df[benchmark].dropna()
        win = 30
        vol = bench.rolling(win).std().dropna()
        if len(vol) < 120:
            st.info("No hay suficiente historia para separar regímenes con esa ventana.")
        else:
            q_low, q_high = vol.quantile([0.40, 0.70])
            reg = pd.Series(index=vol.index, dtype="object")
            reg[vol < q_low] = "Calm"
            reg[(vol >= q_low) & (vol < q_high)] = "Mid"
            reg[vol >= q_high] = "Stress"

            st.write("Conteo por régimen:")
            st.dataframe(reg.value_counts().to_frame("días"), use_container_width=True)

            # Para correlaciones por régimen necesitamos varios activos con retornos
            valid_cols = [c for c in rets_df.columns if rets_df[c].dropna().shape[0] >= 60]
            if len(valid_cols) < 2:
                st.info("No hay suficientes activos con retornos para comparar correlación por régimen.")
            else:
                r = rets_df[valid_cols].copy()

                calm_idx = reg[reg == "Calm"].index
                stress_idx = reg[reg == "Stress"].index

                r_calm = r.loc[r.index.intersection(calm_idx)]
                r_stress = r.loc[r.index.intersection(stress_idx)]

                if r_calm.shape[0] < 60 or r_stress.shape[0] < 60:
                    st.info("Muy pocos días Calm/Stress para comparar correlaciones. Usa 5Y o baja umbrales.")
                else:
                    c_calm = r_calm.corr(min_periods=60)
                    c_stress = r_stress.corr(min_periods=60)
                    diff = (c_stress - c_calm).fillna(0)

                    st.plotly_chart(px.imshow(diff, text_auto=".2f", aspect="auto", title="Cambio de correlación: Stress - Calm"), use_container_width=True)
                    st.caption("Rojo = en estrés se alinean más. Azul = en estrés se desacoplan.")

# =========================
# TAB 5: Oro vs Plata
# =========================
with tab5:
    st.subheader("Oro vs Plata (5 años) – precio + volumen (futuros)")
    st.caption("Para volumen usamos futuros: Oro (GC=F) y Plata (SI=F).")

    end = date.today()
    start = end - timedelta(days=int(365.25*5) + 10)

    g, g_used = download_first_working([GOLD_FUT], start, end)
    s, s_used = download_first_working([SILV_FUT], start, end)

    if g.empty or s.empty or g["Close"].dropna().empty or s["Close"].dropna().empty:
        st.warning("No pude descargar GC=F o SI=F (Yahoo). Intenta recargar o más tarde.")
    else:
        df = pd.DataFrame({
            "Gold": g["Close"].dropna(),
            "Silver": s["Close"].dropna()
        }).dropna()

        norm = df / df.iloc[0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=norm.index, y=norm["Gold"], mode="lines", name="Gold (norm)"))
        fig.add_trace(go.Scatter(x=norm.index, y=norm["Silver"], mode="lines", name="Silver (norm)"))
        fig.update_layout(title="Precio normalizado (5 años)", height=340, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

        ratio = df["Gold"] / df["Silver"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ratio.index, y=ratio.values, mode="lines"))
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

        # Picos de volatilidad del oro
        st.markdown("### ⚡ Momentos más volátiles del oro (GC=F)")
        top_n = st.slider("Top N picos (oro)", 5, 20, 10, step=1)
        win = st.slider("Ventana vol (oro)", 10, 90, 30, step=5)

        g_close = g["Close"].dropna()
        g_ret = g_close.pct_change().dropna()

        if len(g_ret) > win + 60:
            roll = g_ret.rolling(win).std() * np.sqrt(252)
            peaks = roll.dropna().nlargest(top_n)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines", name="Vol rolling"))
            for d in peaks.index:
                fig.add_vline(x=d, line_dash="dash", opacity=0.5)
            fig.update_layout(title=f"Oro – Vol rolling {win}d (picos marcados)", height=320, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

            table = pd.DataFrame({
                "Fecha": peaks.index.date,
                f"Vol rolling {win}d (ann)": peaks.values,
                "Retorno 1d": g_ret.reindex(peaks.index).values,
                "Retorno 5d": g_close.pct_change(5).reindex(peaks.index).values,
            })
            try:
                st.dataframe(table.style.format({
                    f"Vol rolling {win}d (ann)":"{:.2%}",
                    "Retorno 1d":"{:.2%}",
                    "Retorno 5d":"{:.2%}",
                }), use_container_width=True)
            except Exception:
                st.dataframe(table, use_container_width=True)
        else:
            st.info("No hay suficiente historia para detectar picos con esa ventana.")

st.markdown("---")
st.caption("Disclaimer: análisis estadístico, no asesoría financiera.")
