# app.py
# ============================================================
# MT5 Portfolio Lab (CSV-only) — Streamlit Web App
# ============================================================
# TODO desde CSV exportados de MT5 (sin Yahoo):
# - Dashboard por activo: precio, retorno, CAGR, volatilidad, Sharpe, MaxDD
# - Volumen (tick volume), volatilidad, recorrido (rango/ATR)
# - Clasificación: Lateral vs Tendencial (ADX + R²)
# - Top N peores drawdowns por activo (peak->trough, fechas y duración)
# - Correlación normal + rolling correlation + clustering (sin matplotlib)
# - Regímenes Calm/Mid/Stress + Heatmap Stress - Calm
# - “Momentos más volátiles” para CUALQUIER activo (picos de vol rolling + tabla)
# - Comparador genérico (cualquier par): precio normalizado + ratio + volumen MA
# - “Qué pasó esta semana” por activo
#
# Requisitos:
#   streamlit, pandas, numpy, plotly, scipy
#
# Uso:
# - Exporta desde MT5 a CSV (ideal Daily). Si es H1/M15, esta app lo resamplea a diario.
# - Sube 1+ archivos. Ideal: 1 CSV por símbolo, nombre = SIMBOLO.csv (EURUSD.csv, XAUUSD.csv, BTCUSD.csv, etc.)
# ============================================================

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Clustering opcional sin matplotlib (Plotly FigureFactory)
try:
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    import plotly.figure_factory as ff
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="MT5 Portfolio Lab", page_icon="📈", layout="wide")
st.title("📈 MT5 Portfolio Lab (CSV-only)")
st.caption("Sin Yahoo, sin bloqueos. Solo CSV exportados de MetaTrader 5.")

# -----------------------------
# Parsing CSV MT5
# -----------------------------
def _norm_col(c: str) -> str:
    c = str(c).strip().lower()
    c = c.replace(" ", "_")
    c = re.sub(r"[^a-z0-9_]", "", c)
    return c

def _infer_symbol_from_filename(name: str) -> str:
    base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    base = base.rsplit(".", 1)[0].strip().upper()
    base = re.split(r"[,\s;()\-]+", base)[0].upper()
    return base

def _read_csv_flexible(file_bytes: bytes) -> pd.DataFrame:
    text = file_bytes.decode("utf-8", errors="ignore")
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    df.columns = [_norm_col(c) for c in df.columns]
    return df

def _to_numeric(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.replace(" ", "", regex=False)
    mask = x.str.contains(",") & (~x.str.contains(r"\.", regex=True))
    x.loc[mask] = x.loc[mask].str.replace(",", ".", regex=False)
    x = x.str.replace(",", "", regex=False)
    return pd.to_numeric(x, errors="coerce")

def _pick_col(cols: set, *cands):
    for c in cands:
        if c in cols:
            return c
    return None

def parse_mt5_to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    cols = set(df.columns)

    dt_col = _pick_col(cols, "time", "date", "datetime", "timestamp")
    if dt_col is None:
        for c in df.columns:
            if "time" in c or "date" in c:
                dt_col = c
                break
    if dt_col is None:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    dt = pd.to_datetime(df[dt_col], errors="coerce")
    out = df.copy()
    out["_dt"] = dt
    out = out.dropna(subset=["_dt"]).sort_values("_dt")

    c_open  = _pick_col(cols, "open")
    c_high  = _pick_col(cols, "high")
    c_low   = _pick_col(cols, "low")
    c_close = _pick_col(cols, "close", "last")

    if c_close is None:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    ohlcv = pd.DataFrame(index=out["_dt"])
    ohlcv["Close"] = _to_numeric(out[c_close])

    if c_open: ohlcv["Open"] = _to_numeric(out[c_open])
    else:      ohlcv["Open"] = ohlcv["Close"]

    if c_high: ohlcv["High"] = _to_numeric(out[c_high])
    else:      ohlcv["High"] = ohlcv[["Open","Close"]].max(axis=1)

    if c_low:  ohlcv["Low"] = _to_numeric(out[c_low])
    else:      ohlcv["Low"] = ohlcv[["Open","Close"]].min(axis=1)

    c_vol = _pick_col(cols, "tick_volume", "tickvolume", "real_volume", "volume", "vol")
    if c_vol:
        ohlcv["Volume"] = _to_numeric(out[c_vol])
    else:
        ohlcv["Volume"] = np.nan

    ohlcv = ohlcv.dropna(subset=["Close"])
    ohlcv = ohlcv[~ohlcv.index.duplicated(keep="last")]
    return ohlcv[["Open","High","Low","Close","Volume"]]

def resample_daily(ohlcv: pd.DataFrame) -> pd.DataFrame:
    if ohlcv.empty:
        return ohlcv
    d = ohlcv.copy()
    d.index = pd.to_datetime(d.index).tz_localize(None)
    daily = pd.DataFrame({
        "Open":  d["Open"].resample("1D").first(),
        "High":  d["High"].resample("1D").max(),
        "Low":   d["Low"].resample("1D").min(),
        "Close": d["Close"].resample("1D").last(),
        "Volume":d["Volume"].resample("1D").sum(min_count=1),
    })
    daily = daily.dropna(subset=["Close"])
    return daily

# -----------------------------
# Indicadores / métricas
# -----------------------------
def compute_adx_atr(df: pd.DataFrame, n=14):
    if df.empty or df["Close"].dropna().empty:
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
    minus_dm_s = pd.Series(minus_dm,index=df.index).ewm(alpha=1/n, adjust=False).mean()

    plus_di  = 100*(plus_dm_s/atr.replace(0, np.nan))
    minus_di = 100*(minus_dm_s/atr.replace(0, np.nan))

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
                events.append({
                    "Peak Date": peak_date,
                    "Trough Date": trough_date,
                    "Recovery Date": dt,
                    "Drawdown %": (trough_price/peak_price - 1),
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
        events.append({
            "Peak Date": peak_date,
            "Trough Date": trough_date,
            "Recovery Date": pd.NaT,
            "Drawdown %": (trough_price/peak_price - 1),
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
    rets  = close.pct_change().dropna()

    days = (close.index[-1] - close.index[0]).days
    years = max(days/365.25, 1e-9)

    total_ret = float(close.iloc[-1]/close.iloc[0] - 1)
    cagr = float((close.iloc[-1]/close.iloc[0])**(1/years) - 1) if close.iloc[0] > 0 else np.nan
    vol_ann = float(rets.std()*np.sqrt(252)) if len(rets) > 10 else np.nan
    sharpe  = float(cagr/vol_ann) if pd.notna(vol_ann) and vol_ann > 0 else np.nan
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
        "Sharpe(~rf0)": sharpe,
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
    close = df["Close"].dropna()
    high  = df["High"].fillna(close)
    low   = df["Low"].fillna(close)

    last_dt = close.index[-1]
    week_start = (last_dt - pd.Timedelta(days=last_dt.weekday())).normalize()
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

def rolling_vol_peaks(close: pd.Series, win: int, top_n: int):
    close = close.dropna()
    ret = close.pct_change().dropna()
    if len(ret) < win + 60:
        return None, None, None
    roll = ret.rolling(win).std()*np.sqrt(252)
    peaks = roll.dropna().nlargest(top_n)
    table = pd.DataFrame({
        "Fecha": peaks.index,
        f"Vol rolling {win}d (ann)": peaks.values,
        "Retorno 1d": ret.reindex(peaks.index).values,
        "Retorno 5d": close.pct_change(5).reindex(peaks.index).values,
    })
    return roll, peaks, table

# -----------------------------
# Sidebar: carga CSV + mapping
# -----------------------------
st.sidebar.header("📥 Cargar CSV(s) de MT5")
st.sidebar.caption("Ideal: 1 CSV por símbolo, nombre = SIMBOLO.csv. Si no, aquí lo corriges.")

uploads = st.sidebar.file_uploader("CSV(s)", type=["csv", "txt"], accept_multiple_files=True)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parámetros")
min_rows = st.sidebar.slider("Mínimo de velas por símbolo (diario)", 10, 600, 60, step=10)
top_dd_n = st.sidebar.selectbox("Top drawdowns por activo", [3,5,10], index=1)
roll_vol_win = st.sidebar.slider("Ventana vol rolling [días]", 10, 180, 30, step=5)
roll_corr_win= st.sidebar.slider("Ventana rolling corr [días]", 30, 360, 90, step=10)

if not uploads:
    st.info("👈 Sube tus CSV de MT5 y la app hace TODO el análisis.")
    with st.expander("Cómo exportar (2 líneas)", expanded=False):
        st.write("MT5 → History/Export → CSV con Time/Open/High/Low/Close (+ Tick Volume si existe).")
        st.write("Nombra: EURUSD.csv, XAUUSD.csv, BTCUSD.csv, etc.")
    st.stop()

# Overrides
with st.sidebar.expander("🧷 Asignación de símbolo por archivo", expanded=False):
    overrides = {}
    for f in uploads:
        guess = _infer_symbol_from_filename(f.name)
        sym = st.text_input(f.name, value=guess, key=f"sym_{f.name}")
        overrides[f.name] = sym.strip().upper()

# -----------------------------
# Cargar series
# -----------------------------
meta_rows = []
series = {}  # symbol -> daily ohlcv

for f in uploads:
    sym = overrides.get(f.name, _infer_symbol_from_filename(f.name))

    try:
        raw = _read_csv_flexible(f.getvalue())
        ohlcv = parse_mt5_to_ohlcv(raw)
        daily = resample_daily(ohlcv)
    except Exception:
        daily = pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    if not daily.empty:
        if sym in series:
            comb = pd.concat([series[sym], daily]).sort_index()
            comb = comb[~comb.index.duplicated(keep="last")]
            series[sym] = comb
        else:
            series[sym] = daily

    meta_rows.append({
        "Archivo": f.name,
        "Símbolo": sym,
        "Filas (diario)": int(len(daily)),
        "Desde": (daily.index.min().date().isoformat() if not daily.empty else "—"),
        "Hasta": (daily.index.max().date().isoformat() if not daily.empty else "—"),
        "Volumen": ("sí" if (not daily.empty and daily["Volume"].notna().any()) else "no/NA"),
    })

st.subheader("Estado de carga (resampleado a diario)")
st.dataframe(pd.DataFrame(meta_rows), use_container_width=True)

symbols = [s for s, df in series.items() if df is not None and len(df) >= min_rows]
if not symbols:
    st.error("No hay símbolos con suficientes velas. Baja el mínimo o sube CSV con más historia.")
    st.stop()

global_min = min(series[s].index.min() for s in symbols)
global_max = max(series[s].index.max() for s in symbols)

st.sidebar.markdown("---")
st.sidebar.header("🗓️ Rango de análisis")
start_date = st.sidebar.date_input("Inicio", value=global_min.date())
end_date   = st.sidebar.date_input("Fin", value=global_max.date())

def slice_range(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    m = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    return df.loc[m].copy()

data = {s: slice_range(series[s]) for s in symbols}
symbols = [s for s in symbols if len(data[s]) >= min_rows]
if not symbols:
    st.error("Con ese rango se quedó sin data suficiente. Ajusta fechas o baja mínimo.")
    st.stop()

benchmark = st.sidebar.selectbox("Benchmark para regímenes (estrés)", options=symbols, index=0)

# -----------------------------
# Métricas, returns
# -----------------------------
rows = []
week_rows = []
rets_cols = {}

for s in symbols:
    df = data[s]
    m = compute_metrics(df)
    if m:
        m["Símbolo"] = s
        rows.append(m)

    w = week_metrics(df)
    if w:
        w["Símbolo"] = s
        week_rows.append(w)

    close = df["Close"].dropna()
    if len(close) >= 30:
        rets_cols[s] = close.pct_change()

summary = pd.DataFrame(rows).set_index("Símbolo").sort_values("Vol anualizada", ascending=False)
weekdf  = pd.DataFrame(week_rows).set_index("Símbolo") if week_rows else pd.DataFrame()
rets_df = pd.DataFrame(rets_cols)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "✅ Resumen",
    "🔍 Detalle + Drawdowns",
    "🔗 Correlaciones",
    "🌡️ Estrés vs Calma",
    "🧨 Eventos + Comparador",
])

# ===== TAB 1 =====
with tab1:
    st.subheader("Resumen por símbolo")
    st.dataframe(
        summary.style.format({
            "Precio":"{:,.6f}",
            "Retorno total":"{:.2%}",
            "CAGR":"{:.2%}",
            "Vol anualizada":"{:.2%}",
            "Sharpe(~rf0)":"{:,.2f}",
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
        st.dataframe(summary["Sharpe(~rf0)"].sort_values(ascending=False).head(7).to_frame("Sharpe"), use_container_width=True)

    st.markdown("### Lateral vs Tendencial (ADX + R²)")
    cls = summary[["Clasificación","ADX14","R2 tendencia","Retorno total","Vol anualizada","ATR14 %","Avg rango diario %"]].sort_values(
        ["Clasificación","ADX14"], ascending=[True, False]
    )
    st.dataframe(cls, use_container_width=True)

    st.markdown("### ¿Qué pasó esta semana?")
    if not weekdf.empty:
        st.dataframe(
            weekdf.style.format({
                "Retorno semana":"{:.2%}",
                "Rango semana %":"{:.2%}",
                "Vol semana (ann)":"{:.2%}",
                "Volumen semana (suma)":"{:,.0f}",
            }),
            use_container_width=True
        )
    else:
        st.info("No hay suficiente data para tabla semanal.")

    st.download_button("⬇️ Descargar resumen (CSV)", data=csv_bytes(summary), file_name="summary.csv", mime="text/csv")

# ===== TAB 2 =====
with tab2:
    st.subheader("Detalle por símbolo + Top drawdowns")
    sym = st.selectbox("Símbolo", options=list(summary.index), index=0)

    df = data[sym]
    close = df["Close"].dropna()

    m = summary.loc[sym]
    a1,a2,a3,a4,a5,a6 = st.columns(6)
    a1.metric("Precio", f"{m['Precio']:,.6f}")
    a2.metric("Retorno total", f"{m['Retorno total']*100:,.2f}%")
    a3.metric("Vol anual", f"{m['Vol anualizada']*100:,.2f}%" if pd.notna(m["Vol anualizada"]) else "—")
    a4.metric("MaxDD", f"{m['MaxDD']*100:,.2f}%")
    a5.metric("ADX14", f"{m['ADX14']:,.1f}" if pd.notna(m["ADX14"]) else "—")
    a6.metric("Clasificación", str(m["Clasificación"]))

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
    st.markdown(f"### 🧨 Top {top_dd_n} peores drawdowns (peak → trough)")
    if ev.empty:
        st.info("No se detectaron eventos de drawdown.")
    else:
        top_ev = ev.head(top_dd_n)[["Peak Date","Trough Date","Recovery Date","Drawdown %","Days Peak->Trough","Days Trough->Recovery","Days Total","Peak Price","Trough Price"]]
        st.dataframe(
            top_ev.style.format({
                "Drawdown %":"{:.2%}",
                "Peak Price":"{:,.6f}",
                "Trough Price":"{:,.6f}",
            }),
            use_container_width=True
        )
        st.download_button("⬇️ Descargar drawdowns (CSV)", data=csv_bytes(ev), file_name=f"drawdowns_{sym}.csv", mime="text/csv")

# ===== TAB 3 =====
with tab3:
    st.subheader("Correlación normal + Rolling Corr + Clustering")
    valid_cols = [c for c in rets_df.columns if rets_df[c].dropna().shape[0] >= 60]

    if len(valid_cols) < 2:
        st.info("Para correlaciones necesitas al menos 2 símbolos con retornos suficientes.")
    else:
        r = rets_df[valid_cols].copy()
        corr = r.corr(min_periods=60)
        st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", title="Matriz de correlación (retornos diarios)"), use_container_width=True)

        st.markdown("### Rolling correlation (elige par)")
        colA, colB = st.columns(2)
        a = colA.selectbox("Símbolo A", options=valid_cols, index=0, key="ra")
        b = colB.selectbox("Símbolo B", options=valid_cols, index=1, key="rb")
        if a != b:
            ab = pd.concat([r[a], r[b]], axis=1).dropna()
            if len(ab) < roll_corr_win + 60:
                st.info("Muy pocos datos alineados para esa ventana. Baja la ventana o amplía rango.")
            else:
                rc = ab[a].rolling(roll_corr_win).corr(ab[b])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rc.index, y=rc.values, mode="lines"))
                fig.add_hline(y=0)
                fig.update_layout(title=f"Rolling Corr ({roll_corr_win}d): {a} vs {b}", height=260, margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Clustering (opcional)")
        if SCIPY_OK and len(valid_cols) >= 3:
            corr2 = corr.fillna(0)
            dist = np.sqrt(0.5*(1-corr2))
            dist_cond = squareform(dist.values, checks=False)
            Z = linkage(dist_cond, method="average")
            fig = ff.create_dendrogram(dist.values, labels=list(corr2.columns), linkagefun=lambda _: Z)
            fig.update_layout(height=420, title="Dendrograma (distancia por correlación)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Para clustering necesitas SciPy y mínimo 3 símbolos.")

# ===== TAB 4 =====
with tab4:
    st.subheader("Estrés vs Calma: correlaciones por régimen + Stress–Calm")
    st.caption("Regímenes definidos por volatilidad rolling del benchmark.")

    if benchmark not in rets_df.columns or rets_df[benchmark].dropna().shape[0] < 180:
        st.info("Benchmark sin suficiente historia. Amplía rango o elige otro benchmark.")
    else:
        win = 30
        bench = rets_df[benchmark].dropna()
        vol = bench.rolling(win).std().dropna()
        if len(vol) < 180:
            st.info("No hay suficiente historia para regímenes con esa ventana.")
        else:
            q_low, q_high = vol.quantile([0.40, 0.70])
            reg = pd.Series(index=vol.index, dtype="object")
            reg[vol < q_low] = "Calm"
            reg[(vol >= q_low) & (vol < q_high)] = "Mid"
            reg[vol >= q_high] = "Stress"

            st.dataframe(reg.value_counts().to_frame("días"), use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=vol.index, y=vol.values, mode="lines", name="Vol rolling"))
            fig.add_hline(y=q_low, line_dash="dash")
            fig.add_hline(y=q_high, line_dash="dash")
            fig.update_layout(title=f"{benchmark} – Vol rolling {win}d (umbrales)", height=240, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

            valid_cols = [c for c in rets_df.columns if rets_df[c].dropna().shape[0] >= 120]
            if len(valid_cols) < 2:
                st.info("No hay suficientes símbolos con retornos para correlación por régimen.")
            else:
                r = rets_df[valid_cols].copy()
                calm_idx = reg[reg=="Calm"].index
                stress_idx = reg[reg=="Stress"].index

                r_calm = r.loc[r.index.intersection(calm_idx)].dropna(how="any")
                r_stress = r.loc[r.index.intersection(stress_idx)].dropna(how="any")

                if len(r_calm) < 60 or len(r_stress) < 60:
                    st.info("Muy pocos días Calm/Stress para comparar. Amplía rango.")
                else:
                    c_calm = r_calm.corr(min_periods=60)
                    c_stress = r_stress.corr(min_periods=60)
                    diff = (c_stress - c_calm).fillna(0)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(px.imshow(c_calm, text_auto=".2f", aspect="auto", title="Corr – Calm"), use_container_width=True)
                    with col2:
                        st.plotly_chart(px.imshow(c_stress, text_auto=".2f", aspect="auto", title="Corr – Stress"), use_container_width=True)

                    st.plotly_chart(px.imshow(diff, text_auto=".2f", aspect="auto", title="Cambio de correlación: Stress - Calm"), use_container_width=True)
                    st.caption("Rojo = en estrés se alinean más. Azul = en estrés se desacoplan.")

# ===== TAB 5 =====
with tab5:
    st.subheader("Eventos extremos + Comparador (cualquier símbolo)")
    st.caption("Esto NO es solo oro/plata: picos de volatilidad y comparador sirven para cualquier activo.")

    subA, subB = st.tabs(["🔥 Picos de volatilidad", "🆚 Comparador (2 activos)"])

    with subA:
        sym = st.selectbox("Símbolo para picos de volatilidad", options=symbols, index=0, key="peaks_sym")
        top_n = st.slider("Top N picos", 5, 30, 10, step=1, key="peaks_topn")
        close = data[sym]["Close"].dropna()

        roll, peaks, table = rolling_vol_peaks(close, win=roll_vol_win, top_n=top_n)
        if roll is None:
            st.info("No hay suficiente historia para calcular picos con esa ventana. Amplía rango o baja ventana.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines", name="Vol rolling"))
            for d in peaks.index:
                fig.add_vline(x=d, line_dash="dash", opacity=0.5)
            fig.update_layout(title=f"{sym} – Vol rolling {roll_vol_win}d (picos marcados)", height=280, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                table.style.format({
                    f"Vol rolling {roll_vol_win}d (ann)":"{:.2%}",
                    "Retorno 1d":"{:.2%}",
                    "Retorno 5d":"{:.2%}",
                }),
                use_container_width=True
            )

    with subB:
        col1, col2 = st.columns(2)
        a = col1.selectbox("Activo A", options=symbols, index=0, key="cmp_a")
        b = col2.selectbox("Activo B", options=symbols, index=1 if len(symbols)>1 else 0, key="cmp_b")

        lookback_years = st.slider("Ventana (años) para comparar", 1, 10, 5, step=1)
        end_dt = max(data[a].index.max(), data[b].index.max())
        start_cmp = end_dt - pd.Timedelta(days=int(365.25*lookback_years))

        da = data[a][data[a].index >= start_cmp].copy()
        db = data[b][data[b].index >= start_cmp].copy()

        df = pd.DataFrame({
            "A_Close": da["Close"],
            "B_Close": db["Close"],
            "A_Vol": da["Volume"],
            "B_Vol": db["Volume"],
        }).dropna(subset=["A_Close","B_Close"])

        if df.empty or len(df) < 60:
            st.info("No hay suficiente traslape entre ambos activos en esa ventana. Amplía rango o elige otros.")
        else:
            norm_a = df["A_Close"]/df["A_Close"].iloc[0]
            norm_b = df["B_Close"]/df["B_Close"].iloc[0]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=norm_a, mode="lines", name=f"{a} (norm)"))
            fig.add_trace(go.Scatter(x=df.index, y=norm_b, mode="lines", name=f"{b} (norm)"))
            fig.update_layout(title="Precio normalizado", height=300, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

            ratio = df["A_Close"]/df["B_Close"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=ratio.values, mode="lines", name="A/B"))
            fig.update_layout(title=f"Ratio {a}/{b}", height=240, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

            fig = go.Figure()
            if df["A_Vol"].notna().any():
                fig.add_trace(go.Scatter(x=df.index, y=df["A_Vol"].rolling(20).mean(), mode="lines", name=f"{a} Vol MA20"))
            if df["B_Vol"].notna().any():
                fig.add_trace(go.Scatter(x=df.index, y=df["B_Vol"].rolling(20).mean(), mode="lines", name=f"{b} Vol MA20"))
            fig.update_layout(title="Volumen (MA20) — si el CSV lo trae", height=240, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
