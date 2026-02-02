# app.py
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

# SciPy (para clustering). Si no está, no tronamos.
try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

st.set_page_config(page_title="Trading Lab", page_icon="📈", layout="wide")
st.title("📈 Trading Lab – Dashboard Multi-Activos (MT5/Proxies)")
st.caption("Fuente: Yahoo Finance (yfinance). Nota: en FX spot / algunos índices el volumen suele venir vacío o no ser volumen real.")

# ------------------------------------------------------------
# UNIVERSO (BÁSICOS + EXTRAS)
# ------------------------------------------------------------
# Mapas/Proxies típicos:
# EURUSD=X -> EURUSD
# JPY=X -> USDJPY (Yahoo lo usa así)
# CL=F -> USOIL/WTI (energía)
# GC=F -> oro (XAUUSD)
# SI=F -> plata (XAGUSD) (volumen real en futuros)
# SPY o ^GSPC -> proxy US500
# BTC-USD -> BTCUSD
# ETH-USD -> ETHUSD
# NQ=F o ^NDX -> Nasdaq100
# ^VIX o VX=F -> VIX
# DX-Y.NYB / DX=F / UUP -> DXY proxy

ASSETS = {
    # ===== BÁSICOS =====
    "EURUSD": {"aliases": ["EURUSD=X"], "mt5": "EURUSD"},
    "USDJPY": {"aliases": ["JPY=X"], "mt5": "USDJPY"},
    "XAUUSD (Oro)": {"aliases": ["XAUUSD=X", "GC=F"], "mt5": "XAUUSD"},
    "USOIL / WTI": {"aliases": ["CL=F", "BZ=F"], "mt5": "USOIL"},
    "US500 (SPX proxy)": {"aliases": ["SPY", "^GSPC", "ES=F"], "mt5": "US500"},
    "BTCUSD": {"aliases": ["BTC-USD"], "mt5": "BTCUSD"},

    # ===== EXTRAS QUE PEDISTE =====
    "XAGUSD (Plata)": {"aliases": ["XAGUSD=X", "SI=F"], "mt5": "XAGUSD"},
    "Nasdaq100": {"aliases": ["NQ=F", "^NDX", "QQQ"], "mt5": "NAS100"},
    "USDCHF": {"aliases": ["CHF=X", "USDCHF=X"], "mt5": "USDCHF"},
    "EURGBP": {"aliases": ["EURGBP=X"], "mt5": "EURGBP"},
    "GBPUSD": {"aliases": ["GBPUSD=X"], "mt5": "GBPUSD"},
    "VIX (Índice)": {"aliases": ["^VIX", "VX=F"], "mt5": "VIX"},
    "USDMXN": {"aliases": ["MXN=X", "USDMXN=X"], "mt5": "USDMXN"},
    "Ethereum": {"aliases": ["ETH-USD"], "mt5": "ETHUSD"},
    "DXY (Índice USD)": {"aliases": ["DX-Y.NYB", "DX=F", "UUP", "^DXY"], "mt5": "DXY"},
    "Copper (Cobre)": {"aliases": ["HG=F"], "mt5": "COPPER"},
    "AUDCAD": {"aliases": ["AUDCAD=X"], "mt5": "AUDCAD"},
    "NZDCAD": {"aliases": ["NZDCAD=X"], "mt5": "NZDCAD"},
}

# Para Oro vs Plata volumen real-ish: futuros
GOLD_FUT = "GC=F"
SILV_FUT = "SI=F"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _infer_dates(preset: str):
    today = date.today()
    if preset == "6M":  return today - timedelta(days=183), today
    if preset == "1Y":  return today - timedelta(days=365), today
    if preset == "2Y":  return today - timedelta(days=365*2), today
    if preset == "5Y":  return today - timedelta(days=int(365.25*5)), today
    if preset == "10Y": return today - timedelta(days=int(365.25*10)), today
    if preset == "MAX": return date(1990,1,1), today
    return None, None

@st.cache_data(ttl=60*60, show_spinner=False)
def download_first_available(aliases, start, end, interval="1d"):
    for t in aliases:
        try:
            df = yf.download(t, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
            if df is None or df.empty:
                continue
            df = df.copy()
            for col in ["Open","High","Low","Close","Volume"]:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[["Open","High","Low","Close","Volume"]]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[~df.index.duplicated(keep="last")]
            df = df.dropna(subset=["Close"])
            if len(df) >= 25:  # umbral flexible
                return df, t
        except Exception:
            continue
    return pd.DataFrame(columns=["Open","High","Low","Close","Volume"]), None

def compute_adx_atr(df, n=14):
    if df.empty:
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

def trend_r2(close):
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
    return r2, slope

def max_drawdown(close):
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

    ev = ev.sort_values("Drawdown %").reset_index(drop=True)  # más negativo primero
    return ev

def compute_metrics(df):
    if df.empty or df["Close"].dropna().empty:
        return None

    close = df["Close"].dropna()
    rets = close.pct_change().dropna()

    days = (close.index[-1] - close.index[0]).days
    years = max(days/365.25, 1e-9)

    total_ret = float(close.iloc[-1]/close.iloc[0] - 1)
    cagr = float((close.iloc[-1]/close.iloc[0])**(1/years) - 1) if close.iloc[0] > 0 else np.nan
    vol_ann = float(rets.std()*np.sqrt(252)) if len(rets) > 10 else np.nan
    sharpe = float(cagr/vol_ann) if (pd.notna(vol_ann) and vol_ann > 0) else np.nan
    mdd = max_drawdown(close)

    high = df["High"].fillna(close)
    low  = df["Low"].fillna(close)
    avg_range = float(((high - low).abs()/close).replace([np.inf,-np.inf], np.nan).dropna().mean())

    adx, atr = compute_adx_atr(df, n=14)
    adx_last = float(adx.dropna().iloc[-1]) if not adx.dropna().empty else np.nan
    atr_last = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else np.nan
    atr_pct  = float(atr_last/close.iloc[-1]) if (pd.notna(atr_last) and close.iloc[-1] != 0) else np.nan

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

def week_metrics(df):
    if df.empty or df["Close"].dropna().empty:
        return None
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    close = df["Close"].dropna()
    high = df["High"].fillna(close)
    low  = df["Low"].fillna(close)

    last_dt = close.index[-1]
    week_start = (last_dt - timedelta(days=last_dt.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
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

def pctfmt(x): return f"{x*100:,.2f}%" if pd.notna(x) else "—"
def numfmt(x, d=2): return f"{x:,.{d}f}" if pd.notna(x) else "—"
def csv_bytes(df): return df.to_csv(index=True).encode("utf-8-sig")

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.header("⚙️ Configuración")

all_names = list(ASSETS.keys())
# default: básicos + algunos extras comunes
default = ["EURUSD","USDJPY","XAUUSD (Oro)","USOIL / WTI","US500 (SPX proxy)","BTCUSD","Nasdaq100","USDMXN","DXY (Índice USD)","VIX (Índice)"]

selected_assets = st.sidebar.multiselect("Activos", options=all_names, default=[x for x in default if x in all_names])

preset = st.sidebar.selectbox("Periodo", ["6M","1Y","2Y","5Y","10Y","MAX","Custom"], index=3)
if preset == "Custom":
    c1, c2 = st.sidebar.columns(2)
    start_date = c1.date_input("Inicio", value=date.today()-timedelta(days=int(365.25*5)))
    end_date   = c2.date_input("Fin", value=date.today())
else:
    start_date, end_date = _infer_dates(preset)

min_rows = st.sidebar.slider("Mínimo de velas por activo", 15, 120, 25, step=5)
top_dd_n = st.sidebar.selectbox("Top drawdowns por activo", [3,5,10], index=1)
roll_vol_win = st.sidebar.slider("Ventana vol rolling (días)", 10, 90, 30, step=5)
roll_corr_win= st.sidebar.slider("Ventana corr rolling (días)", 30, 180, 90, step=10)

if not selected_assets:
    st.warning("Selecciona al menos 1 activo.")
    st.stop()

benchmark = st.sidebar.selectbox("Benchmark para regímenes", options=selected_assets, index=0)

# ------------------------------------------------------------
# Download
# ------------------------------------------------------------
with st.spinner("Bajando datos (Yahoo) y calculando..."):
    data = {}
    meta = []
    for a in selected_assets:
        df, used = download_first_available(ASSETS[a]["aliases"], start_date, end_date, interval="1d")
        data[a] = df
        meta.append({"Activo": a, "MT5": ASSETS[a]["mt5"], "Yahoo usado": used if used else "N/A", "Filas": len(df)})

meta_df = pd.DataFrame(meta)

# Filtra los activos con data suficiente
ok_assets = [a for a in selected_assets if (data[a] is not None and len(data[a]) >= min_rows)]
failed_assets = [a for a in selected_assets if a not in ok_assets]

if failed_assets:
    st.warning("Algunos símbolos no bajaron bien de Yahoo (o el periodo fue muy corto). Sigo con los que sí.")
    st.dataframe(meta_df[meta_df["Activo"].isin(failed_assets)], use_container_width=True)

if len(ok_assets) < 2:
    st.error("Muy pocos activos con datos suficientes. Prueba periodo más largo (2Y/5Y) o quita DXY/VIX primero.")
    st.stop()

# Si benchmark falló, escoger otro
if benchmark not in ok_assets:
    benchmark = ok_assets[0]
    st.info(f"Benchmark no tenía datos suficientes; usé {benchmark} como referencia.")

# Métricas
rows, week_rows = [], []
for a in ok_assets:
    df = data[a]
    m = compute_metrics(df)
    if m is None:
        continue
    m["Activo"] = a
    m["MT5"] = ASSETS[a]["mt5"]
    m["Yahoo usado"] = meta_df.loc[meta_df["Activo"]==a, "Yahoo usado"].iloc[0]
    rows.append(m)

    w = week_metrics(df)
    if w is not None:
        w["Activo"] = a
        week_rows.append(w)

summary = pd.DataFrame(rows).set_index("Activo").sort_values("Vol anualizada", ascending=False)
weekdf  = pd.DataFrame(week_rows).set_index("Activo") if week_rows else pd.DataFrame()

# Retornos matrix
rets = {}
for a in ok_assets:
    c = data[a]["Close"].dropna()
    if len(c) >= 40:
        rets[a] = c.pct_change().dropna()
rets_df = pd.concat(rets, axis=1).dropna(how="any") if len(rets) else pd.DataFrame()

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["✅ Resumen", "🔍 Detalle + Drawdowns", "🔗 Correlaciones", "🌡️ Regímenes", "🥇 Oro vs Plata"])

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
        st.dataframe(summary["Vol anualizada"].head(7).to_frame("Vol anualizada"), use_container_width=True)
    with c2:
        st.write("💰 Más 'rentables' (CAGR)")
        st.dataframe(summary["CAGR"].sort_values(ascending=False).head(7).to_frame("CAGR"), use_container_width=True)
    with c3:
        st.write("⚖️ Mejor Sharpe (rf~0)")
        st.dataframe(summary["Sharpe (~rf0)"].sort_values(ascending=False).head(7).to_frame("Sharpe"), use_container_width=True)

    st.markdown("### Lateral vs Tendencial (regla práctica: ADX + R²)")
    cls = summary[["Clasificación","ADX14","R2 tendencia","Retorno total","Vol anualizada","ATR14 %","Avg rango diario %"]].sort_values(["Clasificación","ADX14"], ascending=[True, False])
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
    a1.metric("Precio", numfmt(m["Precio"],4))
    a2.metric("Retorno total", pctfmt(m["Retorno total"]))
    a3.metric("Vol anual", pctfmt(m["Vol anualizada"]))
    a4.metric("MaxDD", pctfmt(m["MaxDD"]))
    a5.metric("ADX14", numfmt(m["ADX14"],1))
    a6.metric("Clasificación", str(m["Clasificación"]))

    st.caption(f"MT5: {ASSETS[asset]['mt5']} | Yahoo: {m['Yahoo usado']} | Obs: {m['Obs']}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines", name="Close"))
    fig.update_layout(title=f"{asset} – Precio", height=360, margin=dict(l=20,r=20,t=50,b=20))
    st.plotly_chart(fig, use_container_width=True)

    # rolling vol
    ret = close.pct_change().dropna()
    if len(ret) > roll_vol_win + 10:
        rv = ret.rolling(roll_vol_win).std()*np.sqrt(252)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rv.index, y=rv.values, mode="lines", name="Rolling Vol"))
        fig.update_layout(title=f"{asset} – Vol rolling {roll_vol_win}d (ann)", height=280, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

    # drawdown curve
    dd = close/close.cummax()-1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown"))
    fig.update_layout(title=f"{asset} – Drawdown (desde máximos)", height=280, margin=dict(l=20,r=20,t=50,b=20))
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    # top dd events
    ev = drawdown_events(close)
    st.markdown(f"### 🧨 Top {top_dd_n} peores drawdowns históricos")
    if ev.empty:
        st.info("No se detectaron eventos de drawdown (o data muy corta).")
    else:
        top_ev = ev.head(top_dd_n).copy()
        show_ev = top_ev[["Peak Date","Trough Date","Recovery Date","Drawdown %","Days Peak->Trough","Days Trough->Recovery","Days Total","Peak Price","Trough Price"]]
        try:
            st.dataframe(show_ev.style.format({"Drawdown %":"{:.2%}","Peak Price":"{:,.4f}","Trough Price":"{:,.4f}"}), use_container_width=True)
        except Exception:
            st.dataframe(show_ev, use_container_width=True)

        st.download_button("⬇️ Descargar todos los drawdowns (CSV)", data=csv_bytes(ev), file_name=f"drawdowns_{asset}.csv", mime="text/csv")

with tab3:
    st.subheader("Correlaciones + Rolling Corr + (opcional) Clustering")
    if rets_df.empty or rets_df.shape[1] < 2:
        st.info("No hay suficientes activos con retornos alineados.")
    else:
        corr = rets_df.corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", title="Matriz de correlación"), use_container_width=True)

        c1, c2 = st.columns(2)
        a = c1.selectbox("Activo A", options=list(rets_df.columns), index=0)
        b = c2.selectbox("Activo B", options=list(rets_df.columns), index=min(1, len(rets_df.columns)-1))
        if a != b:
            rc = rets_df[a].rolling(roll_corr_win).corr(rets_df[b])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rc.index, y=rc.values, mode="lines"))
            fig.add_hline(y=0)
            fig.update_layout(title=f"Rolling Corr ({roll_corr_win}d): {a} vs {b}", height=300, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

        if SCIPY_OK and rets_df.shape[1] >= 3:
            dist = np.sqrt(0.5*(1-corr))
            dist_cond = squareform(dist.values, checks=False)
            Z = linkage(dist_cond, method="average")

            import matplotlib.pyplot as plt
            fig2 = plt.figure(figsize=(12,4))
            dendrogram(Z, labels=corr.columns)
            plt.title("Clustering jerárquico (distancia por correlación)")
            st.pyplot(fig2, clear_figure=True)
        else:
            st.info("Clustering requiere SciPy y al menos 3 activos con data suficiente.")

with tab4:
    st.subheader("Regímenes Calm / Mid / Stress (según vol del benchmark)")
    if benchmark not in rets_df.columns:
        st.info("Benchmark no tiene retornos alineados suficientes.")
    else:
        bench = rets_df[benchmark].dropna()
        win = 30
        vol = bench.rolling(win).std().dropna()
        if len(vol) < 100:
            st.info("No hay suficiente historia para separar regímenes.")
        else:
            q_low, q_high = vol.quantile([0.40, 0.70])
            reg = pd.Series(index=vol.index, dtype="object")
            reg[vol < q_low] = "Calm"
            reg[(vol >= q_low) & (vol < q_high)] = "Mid"
            reg[vol >= q_high] = "Stress"

            st.dataframe(reg.value_counts().to_frame("días"), use_container_width=True)

            aligned = rets_df.loc[reg.index].dropna(how="any")

            def corr_reg(label):
                idx = aligned.index.intersection(reg[reg==label].index)
                if len(idx) < 40:
                    return None
                return aligned.loc[idx].corr()

            c_calm = corr_reg("Calm")
            c_stress = corr_reg("Stress")
            if c_calm is None or c_stress is None:
                st.info("No hay suficientes días Calm/Stress con retornos alineados.")
            else:
                diff = c_stress - c_calm
                st.plotly_chart(px.imshow(diff, text_auto=".2f", aspect="auto", title="Cambio de correlación: Stress - Calm"), use_container_width=True)
                st.caption("Rojo = en estrés se alinean más. Azul = en estrés se desacoplan.")

with tab5:
    st.subheader("Oro vs Plata (5 años) – precio + volumen (futuros)")
    st.caption("Para volumen usamos futuros: Oro (GC=F) y Plata (SI=F).")

    end = date.today()
    start = end - timedelta(days=int(365.25*5)+10)

    g, _ = download_first_available([GOLD_FUT], start, end)
    s, _ = download_first_available([SILV_FUT], start, end)

    if g.empty or s.empty:
        st.warning("No pude descargar GC=F o SI=F (Yahoo).")
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

st.markdown("---")
st.caption("Disclaimer: análisis estadístico, no asesoría financiera.")
