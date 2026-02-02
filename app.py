import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Clustering opcional sin matplotlib
try:
    from scipy.cluster.hierarchy import linkage
    import plotly.figure_factory as ff
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

st.set_page_config(page_title="MT5 Portfolio Lab", page_icon="📈", layout="wide")
st.title("📈 MT5 Portfolio Lab (MT5 CSV only)")
st.caption("Hecho para CSV exportados de MT5 tipo: <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD> (tabs/espacios).")

# ============================================================
# Helpers de parsing (para TU formato exacto)
# ============================================================
def norm_col(c: str) -> str:
    c = str(c).strip().replace("\ufeff", "")
    c = re.sub(r"[<>]", "", c)       # <DATE> -> DATE
    c = c.lower()
    c = c.replace(" ", "_")
    c = re.sub(r"[^a-z0-9_]", "", c)
    return c

def infer_symbol_from_filename(name: str) -> str:
    base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    base = base.rsplit(".", 1)[0].strip().upper()
    base = re.split(r"[,\s;()\-]+", base)[0].upper()
    return base

def to_numeric(s: pd.Series) -> pd.Series:
    # tus números vienen con punto, pero lo dejo robusto
    x = s.astype(str).str.replace(" ", "", regex=False)
    mask = x.str.contains(",") & (~x.str.contains(r"\.", regex=True))
    x.loc[mask] = x.loc[mask].str.replace(",", ".", regex=False)
    x = x.str.replace(",", "", regex=False)
    return pd.to_numeric(x, errors="coerce")

def read_mt5_whitespace_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Lector ROBUSTO para tu caso:
    - Delimitado por tabs/espacios
    - Headers con <>
    """
    text = file_bytes.decode("utf-8", errors="ignore")
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python")
    df.columns = [norm_col(c) for c in df.columns]
    return df

def parse_mt5_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Espera columnas normalizadas:
    date, time, open, high, low, close, tickvol, vol, spread
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    cols = set(df.columns)

    # DATE + TIME (tu formato)
    if "date" in cols and "time" in cols:
        combo = df["date"].astype(str) + " " + df["time"].astype(str)
        # tu formato: 2024.01.01 23:00:00
        dt = pd.to_datetime(combo, format="%Y.%m.%d %H:%M:%S", errors="coerce")
        # fallback por si hay algo raro
        if dt.isna().mean() > 0.05:
            dt = pd.to_datetime(combo, errors="coerce")
    else:
        # fallback (por si exportaste distinto)
        for cand in ["datetime", "timestamp", "time", "date"]:
            if cand in cols:
                dt = pd.to_datetime(df[cand], errors="coerce")
                break
        else:
            return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    out = df.copy()
    out["_dt"] = dt
    out = out.dropna(subset=["_dt"]).sort_values("_dt")

    # OHLC
    for req in ["open","high","low","close"]:
        if req not in out.columns:
            return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    o = pd.DataFrame(index=pd.to_datetime(out["_dt"]).dt.tz_localize(None))
    o["Open"]  = to_numeric(out["open"])
    o["High"]  = to_numeric(out["high"])
    o["Low"]   = to_numeric(out["low"])
    o["Close"] = to_numeric(out["close"])

    # Volume: preferir TICKVOL (tu caso)
    if "tickvol" in cols:
        o["Volume"] = to_numeric(out["tickvol"])
    elif "tick_volume" in cols:
        o["Volume"] = to_numeric(out["tick_volume"])
    elif "volume" in cols:
        o["Volume"] = to_numeric(out["volume"])
    elif "vol" in cols:
        o["Volume"] = to_numeric(out["vol"])
    else:
        o["Volume"] = np.nan

    o = o.dropna(subset=["Close"])
    o = o[~o.index.duplicated(keep="last")]
    return o[["Open","High","Low","Close","Volume"]]

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d.index = pd.to_datetime(d.index).tz_localize(None)
    out = pd.DataFrame({
        "Open":  d["Open"].resample(rule).first(),
        "High":  d["High"].resample(rule).max(),
        "Low":   d["Low"].resample(rule).min(),
        "Close": d["Close"].resample(rule).last(),
        "Volume":d["Volume"].resample(rule).sum(min_count=1),
    })
    out = out.dropna(subset=["Close"])
    return out

# ============================================================
# Indicadores / métricas
# ============================================================
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
    if len(close) < 60:
        return np.nan
    y = np.log(close.values)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope*x + intercept
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return float(r2)

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
        })

    ev = pd.DataFrame(events)
    if ev.empty:
        return ev

    ev["Days Peak->Trough"] = (pd.to_datetime(ev["Trough Date"]) - pd.to_datetime(ev["Peak Date"])).dt.days
    ev = ev.sort_values("Drawdown %").reset_index(drop=True)
    return ev

def compute_metrics(df: pd.DataFrame):
    close = df["Close"].dropna()
    if len(close) < 30:
        return None

    rets = close.pct_change().dropna()
    days = (close.index[-1] - close.index[0]).days
    years = max(days/365.25, 1e-9)

    total_ret = float(close.iloc[-1]/close.iloc[0] - 1)
    cagr = float((close.iloc[-1]/close.iloc[0])**(1/years) - 1) if close.iloc[0] > 0 else np.nan
    vol_ann = float(rets.std()*np.sqrt(252))
    sharpe = float(cagr/vol_ann) if vol_ann > 0 else np.nan
    mdd = max_drawdown(close)

    high = df["High"].fillna(close)
    low  = df["Low"].fillna(close)
    avg_range = float(((high-low).abs()/close).dropna().mean())

    adx, atr = compute_adx_atr(df, n=14)
    adx_last = float(adx.dropna().iloc[-1]) if not adx.dropna().empty else np.nan
    atr_last = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else np.nan
    atr_pct  = float(atr_last/close.iloc[-1]) if close.iloc[-1] != 0 else np.nan

    r2 = trend_r2(close)

    vol_s = df["Volume"].replace(0, np.nan).dropna()
    vol_mean = float(vol_s.mean()) if not vol_s.empty else np.nan

    label = "Mixto"
    if pd.notna(adx_last) and pd.notna(r2):
        if adx_last >= 25 and r2 >= 0.20:
            label = "Tendencial"
        elif adx_last <= 20 and r2 < 0.20:
            label = "Lateral"

    return {
        "Precio": float(close.iloc[-1]),
        "Retorno total": total_ret,
        "CAGR": cagr,
        "Vol anualizada": vol_ann,
        "Sharpe(~rf0)": sharpe,
        "MaxDD": mdd,
        "Avg rango %": avg_range,
        "ATR14 %": atr_pct,
        "ADX14": adx_last,
        "R2": r2,
        "Vol prom": vol_mean,
        "Clasificación": label,
        "Barras": int(len(close)),
        "Desde": close.index.min(),
        "Hasta": close.index.max(),
    }

def rolling_vol_peaks(close: pd.Series, win: int, top_n: int):
    ret = close.pct_change().dropna()
    if len(ret) < win + 10:
        return None, None
    roll = ret.rolling(win).std()*np.sqrt(252)
    peaks = roll.dropna().nlargest(top_n)
    tab = pd.DataFrame({"Fecha": peaks.index, "Vol": peaks.values})
    return roll, tab

# =========================
# Sidebar
# =========================
st.sidebar.header("📥 CSV MT5")
uploads = st.sidebar.file_uploader("Sube CSV(s)", type=["csv", "txt"], accept_multiple_files=True)

st.sidebar.markdown("---")
freq = st.sidebar.selectbox("Frecuencia (resample)", ["1min", "5min", "15min", "1H", "1D"], index=3)
rule = {"1min":"1T", "5min":"5T", "15min":"15T", "1H":"1H", "1D":"1D"}[freq]
bars_per_day = {"1T":1440, "5T":288, "15T":96, "1H":24, "1D":1}[rule]

roll_vol_days = st.sidebar.slider("Ventana vol rolling (días)", 1, 180, 30, 1)
roll_corr_days = st.sidebar.slider("Ventana rolling corr (días)", 1, 365, 90, 1)
top_dd_n = st.sidebar.selectbox("Top drawdowns", [3,5,10], index=1)

roll_vol_win = max(2, int(roll_vol_days * bars_per_day))
roll_corr_win = max(2, int(roll_corr_days * bars_per_day))
st.sidebar.caption(f"Equivalencia: vol={roll_vol_win} barras, corr={roll_corr_win} barras")

if not uploads:
    st.info("👈 Sube tu CSV. Este código está hecho para tu formato <DATE> <TIME> ... <TICKVOL>.")
    st.stop()

# symbol override
with st.sidebar.expander("🧷 Símbolo por archivo", expanded=False):
    overrides = {}
    for f in uploads:
        guess = infer_symbol_from_filename(f.name)
        sym = st.text_input(f.name, value=guess, key=f"sym_{f.name}")
        overrides[f.name] = sym.strip().upper()

# =========================
# Load / parse / resample
# =========================
series = {}
meta = []
for f in uploads:
    sym = overrides.get(f.name, infer_symbol_from_filename(f.name))
    df = read_mt5_whitespace_csv(f.getvalue())
    ohlc = parse_mt5_ohlcv(df)
    rs = resample_ohlcv(ohlc, rule)

    if not rs.empty:
        series[sym] = rs if sym not in series else pd.concat([series[sym], rs]).sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]

    meta.append({
        "Archivo": f.name,
        "Símbolo": sym,
        "Barras": int(len(rs)),
        "Desde": (rs.index.min().strftime("%Y-%m-%d %H:%M") if not rs.empty else "—"),
        "Hasta": (rs.index.max().strftime("%Y-%m-%d %H:%M") if not rs.empty else "—"),
    })

st.subheader(f"Estado de carga (resample a {freq})")
st.dataframe(pd.DataFrame(meta), use_container_width=True)

if not series:
    st.error("No se pudo parsear nada. Revisa que el archivo tenga <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE>.")
    st.stop()

symbols = sorted(series.keys())
gmin = min(series[s].index.min() for s in symbols)
gmax = max(series[s].index.max() for s in symbols)

st.sidebar.markdown("---")
start = st.sidebar.date_input("Inicio", value=gmin.date())
end   = st.sidebar.date_input("Fin", value=gmax.date())

def slice_range(df):
    m = (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    return df.loc[m].copy()

data = {s: slice_range(series[s]) for s in symbols}
symbols = [s for s in symbols if not data[s].empty]

benchmark = st.sidebar.selectbox("Benchmark (estrés)", options=symbols, index=0)

# =========================
# Metrics / returns
# =========================
rows = []
rets_cols = {}
for s in symbols:
    m = compute_metrics(data[s])
    if m:
        m["Símbolo"] = s
        rows.append(m)
    close = data[s]["Close"].dropna()
    if len(close) >= 10:
        rets_cols[s] = close.pct_change()

summary = pd.DataFrame(rows).set_index("Símbolo") if rows else pd.DataFrame()
rets_df = pd.DataFrame(rets_cols)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["✅ Resumen","🔍 Detalle + DD","🔗 Correlaciones","🌡 Estrés","🧨 Eventos/Comparador"])

with tab1:
    if summary.empty:
        st.warning("Hay datos, pero no suficiente historia para métricas (mínimo ~30 barras). Cambia frecuencia o rango.")
    else:
        st.dataframe(summary.sort_values("Vol anualizada", ascending=False), use_container_width=True)

with tab2:
    if summary.empty:
        st.info("Sin métricas suficientes todavía.")
    else:
        sym = st.selectbox("Símbolo", options=list(summary.index), index=0)
        close = data[sym]["Close"].dropna()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines"))
        fig.update_layout(title=f"{sym} – Precio", height=320, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

        ev = drawdown_events(close)
        st.markdown(f"### Top {top_dd_n} drawdowns (peak→trough)")
        st.dataframe(ev.head(top_dd_n), use_container_width=True)

with tab3:
    cols = [c for c in rets_df.columns if rets_df[c].dropna().shape[0] >= 20]
    if len(cols) < 2:
        st.info("Necesitas 2+ activos con retornos suficientes.")
    else:
        r = rets_df[cols].dropna(how="all")
        corr = r.corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlación"), use_container_width=True)

        colA, colB = st.columns(2)
        a = colA.selectbox("A", options=cols, index=0)
        b = colB.selectbox("B", options=cols, index=1)
        if a != b:
            ab = pd.concat([r[a], r[b]], axis=1).dropna()
            if len(ab) >= roll_corr_win + 5:
                rc = ab[a].rolling(roll_corr_win).corr(ab[b])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rc.index, y=rc.values, mode="lines"))
                fig.add_hline(y=0)
                fig.update_layout(title=f"Rolling Corr ({roll_corr_win} barras): {a} vs {b}", height=260, margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Poca data para rolling corr con esa ventana.")

        if SCIPY_OK and len(cols) >= 3:
            # dendrograma en el espacio de retornos (sin matplotlib)
            X = r[cols].dropna().T.values
            fig = ff.create_dendrogram(X, labels=cols)
            fig.update_layout(height=420, title="Clustering (dendrograma)")
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    cols = [c for c in rets_df.columns if rets_df[c].dropna().shape[0] >= roll_vol_win + 20]
    if benchmark not in cols:
        st.info("Benchmark sin suficiente historia para regímenes con tu ventana.")
    else:
        bret = rets_df[benchmark].dropna()
        vol = bret.rolling(roll_vol_win).std().dropna()
        q_low, q_high = vol.quantile([0.40, 0.70])

        reg = pd.Series(index=vol.index, dtype="object")
        reg[vol < q_low] = "Calm"
        reg[(vol >= q_low) & (vol < q_high)] = "Mid"
        reg[vol >= q_high] = "Stress"

        st.dataframe(reg.value_counts().to_frame("barras"), use_container_width=True)

        r = rets_df[cols].dropna(how="any")
        calm_idx = reg[reg=="Calm"].index
        stress_idx = reg[reg=="Stress"].index

        r_calm = r.loc[r.index.intersection(calm_idx)]
        r_stress = r.loc[r.index.intersection(stress_idx)]

        if len(r_calm) > 20 and len(r_stress) > 20:
            c_calm = r_calm.corr()
            c_stress = r_stress.corr()
            diff = (c_stress - c_calm).fillna(0)

            st.plotly_chart(px.imshow(diff, text_auto=".2f", aspect="auto", title="Stress - Calm"), use_container_width=True)
        else:
            st.info("Muy pocos puntos Calm/Stress para comparar.")

with tab5:
    sym = st.selectbox("Activo para picos de vol", options=symbols, index=0, key="pvol")
    close = data[sym]["Close"].dropna()
    roll, table = rolling_vol_peaks(close, win=roll_vol_win, top_n=10)
    if roll is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines"))
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

    if len(df) >= 20:
        na = df["A_Close"]/df["A_Close"].iloc[0]
        nb = df["B_Close"]/df["B_Close"].iloc[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=na, mode="lines", name=a))
        fig.add_trace(go.Scatter(x=df.index, y=nb, mode="lines", name=b))
        fig.update_layout(title="Precio normalizado", height=280, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

st.caption("Listo. Si antes te salía Barras=0, con este lector ya no: está hecho para tu formato.")
