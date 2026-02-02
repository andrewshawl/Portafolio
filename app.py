# streamlit_app.py
# =========================================
# Trading Lab – Dashboard multi-activos (Streamlit)
# - Volumen / Volatilidad / Recorrido / Precio
# - Lateral vs Tendencial (ADX + R²)
# - Top N peores drawdowns históricos por activo (peak->trough)
# - Correlaciones + clustering
# - Regímenes Calm/Mid/Stress + Stress-Calm correlation shift
# - Oro vs Plata (5 años) con volumen usando futuros (GC=F / SI=F)
# =========================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

import plotly.express as px
import plotly.graph_objects as go

from datetime import date, timedelta

# SciPy para dendrograma (si no está, ocultamos esa parte)
try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Trading Lab", page_icon="📈", layout="wide")

st.title("📈 Trading Lab – Dashboard de Volatilidad, Tendencia y Drawdowns")
st.caption("Datos: Yahoo Finance (yfinance). Nota: en FX spot e índices el 'volumen' suele venir vacío o no ser volumen real.")

# -----------------------------
# Universo: lo que pidieron
# -----------------------------
ASSETS = {
    "XAGUSD (Plata spot)": {"aliases": ["XAGUSD=X", "SI=F"], "mt5": "XAGUSD"},
    "Nasdaq100":           {"aliases": ["NQ=F", "^NDX"],      "mt5": "NAS100"},
    "USDCHF":              {"aliases": ["CHF=X"],            "mt5": "USDCHF"},
    "EURGBP":              {"aliases": ["EURGBP=X"],         "mt5": "EURGBP"},
    "GBPUSD":              {"aliases": ["GBPUSD=X"],         "mt5": "GBPUSD"},
    "VIX (Índice)":        {"aliases": ["^VIX", "VX=F"],      "mt5": "VIX"},
    "USDMXN":              {"aliases": ["MXN=X"],            "mt5": "USDMXN"},
    "Ethereum":            {"aliases": ["ETH-USD"],          "mt5": "ETHUSD"},
    "DXY (Índice USD)":    {"aliases": ["DX-Y.NYB", "DX=F", "^DXY"], "mt5": "DXY"},
    "Copper (Cobre)":      {"aliases": ["HG=F"],             "mt5": "COPPER"},
    "AUDCAD":              {"aliases": ["AUDCAD=X"],         "mt5": "AUDCAD"},
    "NZDCAD":              {"aliases": ["NZDCAD=X"],         "mt5": "NZDCAD"},
    # oro (muy recomendado para el módulo):
    "XAUUSD (Oro spot)":   {"aliases": ["XAUUSD=X", "GC=F"], "mt5": "XAUUSD"},
}

GOLD_FUT = "GC=F"
SILV_FUT = "SI=F"

# -----------------------------
# Helpers
# -----------------------------
def _infer_dates(preset: str):
    today = date.today()
    if preset == "6M":
        return today - timedelta(days=183), today
    if preset == "1Y":
        return today - timedelta(days=365), today
    if preset == "2Y":
        return today - timedelta(days=365*2), today
    if preset == "5Y":
        return today - timedelta(days=int(365.25*5)), today
    if preset == "10Y":
        return today - timedelta(days=int(365.25*10)), today
    if preset == "MAX":
        return date(1990, 1, 1), today
    return None, None

@st.cache_data(ttl=60*60, show_spinner=False)
def download_first_available(aliases, start, end, interval="1d"):
    """
    Intenta descargar el primer ticker que tenga datos decentes.
    Devuelve (df, ticker_usado).
    """
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

            if len(df) >= 60:
                return df, t
        except Exception:
            continue

    empty = pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
    return empty, None

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
    return dd.min()

def drawdown_events(close: pd.Series) -> pd.DataFrame:
    """
    Eventos de drawdown 'clásicos': de un máximo (peak) al mínimo posterior (trough),
    y opcionalmente fecha de recuperación a nuevo máximo.
    """
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
            # nuevo máximo (recuperación / nuevo peak)
            if in_dd:
                dd_pct = trough_price/peak_price - 1
                events.append({
                    "Peak Date": peak_date,
                    "Trough Date": trough_date,
                    "Recovery Date": dt,
                    "Drawdown %": dd_pct,
                    "Peak Price": peak_price,
                    "Trough Price": trough_price,
                    "Recovery Price": price,
                })
                in_dd = False

            peak_price = price
            peak_date  = dt
            trough_price = price
            trough_date  = dt
        else:
            # en drawdown
            if not in_dd:
                in_dd = True
                trough_price = price
                trough_date  = dt
            if price < trough_price:
                trough_price = price
                trough_date  = dt

    # drawdown abierto al final
    if in_dd:
        dd_pct = trough_price/peak_price - 1
        events.append({
            "Peak Date": peak_date,
            "Trough Date": trough_date,
            "Recovery Date": pd.NaT,
            "Drawdown %": dd_pct,
            "Peak Price": peak_price,
            "Trough Price": trough_price,
            "Recovery Price": np.nan,
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

    ev = ev.sort_values("Drawdown %").reset_index(drop=True)  # peores primero
    return ev

def compute_metrics(df):
    if df.empty or df["Close"].dropna().empty:
        return None

    close = df["Close"].dropna()
    rets = close.pct_change().dropna()

    days = (close.index[-1] - close.index[0]).days
    years = max(days/365.25, 1e-9)

    total_ret = close.iloc[-1]/close.iloc[0] - 1
    cagr = (close.iloc[-1]/close.iloc[0])**(1/years) - 1 if close.iloc[0] > 0 else np.nan
    vol_ann = rets.std()*np.sqrt(252) if len(rets) > 10 else np.nan
    sharpe = cagr/vol_ann if (vol_ann is not None and vol_ann > 0) else np.nan
    mdd = max_drawdown(close)

    high = df["High"].fillna(close)
    low  = df["Low"].fillna(close)
    avg_range = ((high - low).abs()/close).replace([np.inf,-np.inf], np.nan).dropna().mean()

    adx, atr = compute_adx_atr(df, n=14)
    adx_last = float(adx.dropna().iloc[-1]) if not adx.dropna().empty else np.nan
    atr_last = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else np.nan
    atr_pct  = atr_last/close.iloc[-1] if (atr_last is not None and close.iloc[-1] != 0) else np.nan

    r2, slope = trend_r2(close)

    vol_s = df["Volume"].replace(0, np.nan).dropna()
    vol_mean = float(vol_s.mean()) if not vol_s.empty else np.nan
    vol_last = float(vol_s.iloc[-1]) if not vol_s.empty else np.nan

    # clasificación lateral/tendencial (regla práctica)
    label = "Mixto"
    if not np.isnan(adx_last) and not np.isnan(r2):
        if adx_last >= 25 and r2 >= 0.20:
            label = "Tendencial"
        elif adx_last <= 20 and r2 < 0.20 and abs(total_ret) < 0.08:
            label = "Lateral"
        else:
            label = "Mixto"

    return {
        "Precio": float(close.iloc[-1]),
        "Retorno total": float(total_ret),
        "CAGR": float(cagr),
        "Vol anualizada": float(vol_ann) if vol_ann is not None else np.nan,
        "Sharpe (~rf0)": float(sharpe) if sharpe is not None else np.nan,
        "MaxDD": float(mdd),
        "Avg rango diario %": float(avg_range),
        "ATR14 %": float(atr_pct),
        "ADX14": float(adx_last),
        "R2 tendencia": float(r2),
        "Volumen prom": float(vol_mean) if vol_mean is not None else np.nan,
        "Volumen último": float(vol_last) if vol_last is not None else np.nan,
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

    w_ret = (w_close.iloc[-1]/w_close.iloc[0] - 1) if len(w_close) >= 2 else np.nan
    w_range = ((w_high.max() - w_low.min())/w_close.iloc[-1]) if len(w_close) >= 2 else np.nan
    w_vol = w_close.pct_change().dropna().std()*np.sqrt(252) if len(w_close) > 3 else np.nan

    vol_s = w["Volume"].replace(0, np.nan).dropna()
    w_volsum = float(vol_s.sum()) if not vol_s.empty else np.nan

    return {
        "Semana inicio": w_close.index[0].date(),
        "Semana fin": w_close.index[-1].date(),
        "Retorno semana": float(w_ret),
        "Rango semana %": float(w_range),
        "Vol semana (aprox ann)": float(w_vol),
        "Volumen semana (suma)": float(w_volsum) if w_volsum is not None else np.nan,
    }

def to_pct(x):
    return f"{x*100:,.2f}%" if pd.notna(x) else "—"

def to_num(x, d=2):
    return f"{x:,.{d}f}" if pd.notna(x) else "—"

def df_to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8-sig")

# -----------------------------
# Sidebar UI
# -----------------------------
st.sidebar.header("⚙️ Configuración")

default_assets = [a for a in ASSETS.keys() if a != "XAUUSD (Oro spot)"]  # default sin oro
selected_assets = st.sidebar.multiselect(
    "Activos",
    options=list(ASSETS.keys()),
    default=default_assets
)

preset = st.sidebar.selectbox("Periodo", ["6M","1Y","2Y","5Y","10Y","MAX","Custom"], index=3)
if preset == "Custom":
    colA, colB = st.sidebar.columns(2)
    start_date = colA.date_input("Inicio", value=date.today()-timedelta(days=int(365.25*5)))
    end_date   = colB.date_input("Fin", value=date.today())
else:
    start_date, end_date = _infer_dates(preset)

top_dd_n = st.sidebar.selectbox("Top drawdowns por activo", [3,5,10], index=1)
roll_vol_win = st.sidebar.slider("Ventana vol rolling (días)", 10, 90, 30, step=5)
roll_corr_win= st.sidebar.slider("Ventana corr rolling (días)", 30, 180, 90, step=10)

if not selected_assets:
    st.warning("Selecciona al menos un activo en el sidebar.")
    st.stop()

benchmark = st.sidebar.selectbox(
    "Benchmark para regímenes",
    options=selected_assets,
    index=0
)

st.sidebar.markdown("---")
with st.sidebar.expander("🧩 Notas MT5 / Volumen"):
    st.write(
        "- En MT5 trabajas por símbolo (XAUUSD, NAS100, etc.). Aquí usamos tickers Yahoo como proxy.\n"
        "- FX spot: el volumen en Yahoo normalmente es N/A.\n"
        "- Para comparar volumen Oro/Plata usamos futuros GC=F y SI=F (ahí sí existe volumen)."
    )

# -----------------------------
# Descarga de datos
# -----------------------------
with st.spinner("Bajando datos y calculando métricas..."):
    data = {}
    meta = []
    for a in selected_assets:
        df, used = download_first_available(ASSETS[a]["aliases"], start_date, end_date, interval="1d")
        data[a] = df
        meta.append({
            "Activo": a,
            "MT5": ASSETS[a]["mt5"],
            "Yahoo usado": used if used else "N/A",
            "Filas": len(df) if df is not None else 0
        })

meta_df = pd.DataFrame(meta)

# Construir métricas
rows = []
week_rows = []
for a, df in data.items():
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

if not rows:
    st.error("No se pudo descargar suficiente data (Yahoo). Intenta cambiar periodo o activos.")
    st.stop()

summary = pd.DataFrame(rows).set_index("Activo")
weekdf  = pd.DataFrame(week_rows).set_index("Activo") if week_rows else pd.DataFrame()

# Retornos matrix
rets = {}
for a, df in data.items():
    c = df["Close"].dropna()
    if len(c) < 40:
        continue
    rets[a] = c.pct_change().dropna()
rets_df = pd.concat(rets, axis=1).dropna(how="any") if len(rets) else pd.DataFrame()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "✅ Resumen",
    "🔍 Detalle por activo",
    "🔗 Correlaciones",
    "🌡️ Regímenes (Calm/Stress)",
    "🥇 Oro vs Plata",
])

# =============================
# TAB 1 – Resumen
# =============================
with tab1:
    st.subheader("Mapa de símbolos (MT5 vs Yahoo)")
    st.dataframe(meta_df, use_container_width=True)

    st.subheader("Resumen por activo")
    show = summary.copy().sort_values("Vol anualizada", ascending=False)

    try:
        sty = show.style.format({
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
        })
        st.dataframe(sty, use_container_width=True)
    except Exception:
        st.dataframe(show, use_container_width=True)

    st.markdown("### Rankings rápidos")
    c1, c2, c3 = st.columns(3)

    top_vol = show["Vol anualizada"].sort_values(ascending=False).head(5)
    top_cagr = show["CAGR"].sort_values(ascending=False).head(5)
    top_sharpe = show["Sharpe (~rf0)"].sort_values(ascending=False).head(5)

    with c1:
        st.write("🔥 Más volátiles")
        st.dataframe(top_vol.to_frame("Vol anualizada"), use_container_width=True)
    with c2:
        st.write("💰 Más 'rentables' (CAGR)")
        st.dataframe(top_cagr.to_frame("CAGR"), use_container_width=True)
    with c3:
        st.write("⚖️ Mejor Sharpe (rf~0)")
        st.dataframe(top_sharpe.to_frame("Sharpe"), use_container_width=True)

    st.markdown("### Barras (rápido de ver)")
    bar_df = show[["Vol anualizada","CAGR","MaxDD"]].reset_index()
    st.plotly_chart(px.bar(bar_df, x="Activo", y="Vol anualizada", title="Volatilidad anualizada"), use_container_width=True)
    st.plotly_chart(px.bar(bar_df, x="Activo", y="CAGR", title="CAGR (ojo: en FX es cambio de tipo de cambio, no carry)"), use_container_width=True)

    st.markdown("### Lateral vs Tendencial")
    cls = show[["Clasificación","ADX14","R2 tendencia","Retorno total","Vol anualizada","Avg rango diario %","ATR14 %"]].sort_values(["Clasificación","ADX14"], ascending=[True, False])
    st.dataframe(cls, use_container_width=True)

    st.markdown("### ¿Qué pasó esta semana?")
    if not weekdf.empty:
        wk_show = weekdf.copy()
        try:
            st.dataframe(wk_show.style.format({
                "Retorno semana":"{:.2%}",
                "Rango semana %":"{:.2%}",
                "Vol semana (aprox ann)":"{:.2%}",
                "Volumen semana (suma)":"{:,.0f}",
            }), use_container_width=True)
        except Exception:
            st.dataframe(wk_show, use_container_width=True)
    else:
        st.info("No se pudo calcular la tabla semanal (datos insuficientes).")

    st.markdown("### Descargar")
    st.download_button("⬇️ Descargar resumen (CSV)", data=df_to_csv_download(show), file_name="summary.csv", mime="text/csv")
    if not weekdf.empty:
        st.download_button("⬇️ Descargar semanal (CSV)", data=df_to_csv_download(weekdf), file_name="weekly.csv", mime="text/csv")

# =============================
# TAB 2 – Detalle por activo
# =============================
with tab2:
    st.subheader("Detalle por activo")
    asset = st.selectbox("Elige un activo", options=list(show.index), index=0)

    df = data.get(asset, pd.DataFrame())
    if df.empty or df["Close"].dropna().empty:
        st.warning("Sin datos para este activo.")
    else:
        m = summary.loc[asset]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Precio", to_num(m["Precio"], 4))
        c2.metric("Retorno total", to_pct(m["Retorno total"]))
        c3.metric("Vol anual", to_pct(m["Vol anualizada"]))
        c4.metric("MaxDD", to_pct(m["MaxDD"]))
        c5.metric("ADX14", to_num(m["ADX14"], 1))
        c6.metric("Clasificación", str(m["Clasificación"]))

        st.caption(f"MT5: {ASSETS[asset]['mt5']} | Yahoo: {m['Yahoo usado']}")

        close = df["Close"].dropna()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines", name="Close"))
        fig.update_layout(title=f"{asset} – Precio", height=380, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

        rets_ = close.pct_change().dropna()
        if len(rets_) > roll_vol_win + 10:
            rv = rets_.rolling(roll_vol_win).std()*np.sqrt(252)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rv.index, y=rv.values, mode="lines", name="Rolling vol"))
            fig.update_layout(title=f"{asset} – Volatilidad rolling ({roll_vol_win}d, anualizada)", height=320, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

        dd = close/close.cummax()-1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown"))
        fig.update_layout(title=f"{asset} – Drawdown (desde máximos)", height=320, margin=dict(l=20,r=20,t=50,b=20))
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        ev = drawdown_events(close)
        st.markdown(f"### 🧨 Top {top_dd_n} peores drawdowns históricos (peak → trough)")
        if ev.empty:
            st.info("No se detectaron eventos de drawdown (o data muy corta).")
        else:
            top_ev = ev.head(top_dd_n).copy()
            top_ev_show = top_ev[["Peak Date","Trough Date","Recovery Date","Drawdown %","Days Peak->Trough","Days Trough->Recovery","Days Total","Peak Price","Trough Price"]].copy()
            try:
                st.dataframe(top_ev_show.style.format({"Drawdown %":"{:.2%}","Peak Price":"{:,.4f}","Trough Price":"{:,.4f}"}), use_container_width=True)
            except Exception:
                st.dataframe(top_ev_show, use_container_width=True)

            st.download_button(
                "⬇️ Descargar drawdowns (CSV)",
                data=df_to_csv_download(ev),
                file_name=f"drawdowns_{asset.replace(' ','_')}.csv",
                mime="text/csv"
            )

        vol_s = df["Volume"].replace(0, np.nan).dropna()
        st.markdown("### Volumen")
        if vol_s.empty:
            st.info("Volumen: N/A (normal en FX spot e índices en Yahoo).")
        else:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=vol_s.index, y=vol_s.values, name="Volume"))
            fig.update_layout(title=f"{asset} – Volumen", height=280, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

# =============================
# TAB 3 – Correlaciones
# =============================
with tab3:
    st.subheader("Correlaciones (retornos diarios)")
    if rets_df.empty or rets_df.shape[1] < 2:
        st.info("Faltan activos con data suficiente para correlación.")
    else:
        corr = rets_df.corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", title="Matriz de correlación"), use_container_width=True)

        st.markdown("### Rolling correlation (elige par)")
        colA, colB = st.columns(2)
        a1 = colA.selectbox("Activo A", options=list(rets_df.columns), index=0)
        a2 = colB.selectbox("Activo B", options=list(rets_df.columns), index=min(1, len(rets_df.columns)-1))
        if a1 != a2:
            rc = rets_df[a1].rolling(roll_corr_win).corr(rets_df[a2])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rc.index, y=rc.values, mode="lines", name="Rolling corr"))
            fig.add_hline(y=0)
            fig.update_layout(title=f"Rolling correlation {a1} vs {a2} ({roll_corr_win}d)", height=320, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

        if SCIPY_OK and rets_df.shape[1] >= 3:
            st.markdown("### Clustering (dendrograma)")
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

# =============================
# TAB 4 – Regímenes
# =============================
with tab4:
    st.subheader("Regímenes: Calm / Mid / Stress (según volatilidad del benchmark)")
    if benchmark not in rets_df.columns:
        st.info("El benchmark no tiene retornos suficientes para regímenes.")
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

            st.write("Conteo por régimen:")
            st.dataframe(reg.value_counts().to_frame("días"), use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=vol.index, y=vol.values, mode="lines", name="Vol rolling"))
            fig.add_hline(y=q_low, line_dash="dash")
            fig.add_hline(y=q_high, line_dash="dash")
            fig.update_layout(title=f"{benchmark} – Vol rolling {win}d (thresholds Calm/Mid/Stress)", height=320, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

            aligned = rets_df.loc[reg.index].dropna(how="any")

            def corr_reg(label):
                idx = aligned.index.intersection(reg[reg==label].index)
                if len(idx) < 40:
                    return None
                return aligned.loc[idx].corr()

            c_calm = corr_reg("Calm")
            c_stress = corr_reg("Stress")

            if c_calm is None or c_stress is None:
                st.info("No hay suficientes observaciones para comparar Calm vs Stress.")
            else:
                st.markdown("### Correlación en Calm vs Stress + cambio (Stress - Calm)")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(px.imshow(c_calm, text_auto=".2f", aspect="auto", title="Corr – Calm"), use_container_width=True)
                with col2:
                    st.plotly_chart(px.imshow(c_stress, text_auto=".2f", aspect="auto", title="Corr – Stress"), use_container_width=True)

                diff = c_stress - c_calm
                st.plotly_chart(px.imshow(diff, text_auto=".2f", aspect="auto", title="Cambio de correlación: Stress - Calm"), use_container_width=True)
                st.caption("Rojo = en estrés se alinean más. Azul = en estrés se desacoplan.")

# =============================
# TAB 5 – Oro vs Plata
# =============================
with tab5:
    st.subheader("Oro vs Plata (últimos 5 años) – precio + volumen (usando futuros)")
    st.caption("Para volumen usamos futuros: Oro (GC=F) y Plata (SI=F). XAU/XAG spot normalmente no trae volumen real.")

    end = date.today()
    start = end - timedelta(days=int(365.25*5)+10)

    g, _ = download_first_available([GOLD_FUT], start, end)
    s, _ = download_first_available([SILV_FUT], start, end)

    if g.empty or s.empty:
        st.warning("No pude descargar GC=F o SI=F (Yahoo).")
    else:
        df = pd.DataFrame({
            "Gold": g["Close"].dropna(),
            "Silver": s["Close"].dropna()
        }).dropna()

        norm = df / df.iloc[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=norm.index, y=norm["Gold"], mode="lines", name="Gold (norm)"))
        fig.add_trace(go.Scatter(x=norm.index, y=norm["Silver"], mode="lines", name="Silver (norm)"))
        fig.update_layout(title="Precio normalizado (5 años)", height=360, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

        ratio = df["Gold"]/df["Silver"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ratio.index, y=ratio.values, mode="lines", name="Gold/Silver"))
        fig.update_layout(title="Ratio Oro/Plata", height=300, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

        gv = g["Volume"].replace(0, np.nan).dropna()
        sv = s["Volume"].replace(0, np.nan).dropna()

        fig = go.Figure()
        if not gv.empty:
            fig.add_trace(go.Scatter(x=gv.index, y=gv.rolling(20).mean(), mode="lines", name="Gold Vol MA20"))
        if not sv.empty:
            fig.add_trace(go.Scatter(x=sv.index, y=sv.rolling(20).mean(), mode="lines", name="Silver Vol MA20"))
        fig.update_layout(title="Volumen (MA20)", height=300, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### ⚡ Momentos más volátiles del oro (GC=F)")
        top_n = st.slider("Top N picos (oro)", 5, 20, 10, step=1)
        win = st.slider("Ventana vol (oro)", 10, 90, 30, step=5)

        g_close = g["Close"].dropna()
        g_ret = g_close.pct_change().dropna()
        if len(g_ret) > win + 50:
            roll = g_ret.rolling(win).std()*np.sqrt(252)
            peaks = roll.dropna().nlargest(top_n)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines", name="Vol rolling"))
            for d in peaks.index:
                fig.add_vline(x=d, line_dash="dash", opacity=0.6)
            fig.update_layout(title=f"Oro – Vol rolling {win}d (picos marcados)", height=320, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

            table = pd.DataFrame({
                "Fecha": peaks.index.date,
                f"Vol rolling {win}d (ann)": peaks.values,
                "Retorno 1d": g_ret.reindex(peaks.index).values,
                "Retorno 5d": g_close.pct_change(5).reindex(peaks.index).values
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
            st.info("No hay suficiente historia de oro para detectar picos con esa ventana.")

st.markdown("---")
st.caption("Disclaimer: esto es análisis estadístico, no asesoría financiera.")
