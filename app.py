import io
import re
import csv
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Clustering opcional sin matplotlib
try:
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    import plotly.figure_factory as ff
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# =========================
# UI setup
# =========================
st.set_page_config(page_title="MT5 Portfolio Lab", page_icon="📈", layout="wide")
st.title("📈 MT5 Portfolio Lab (MT5 CSV only)")
st.caption("Solo CSV exportados de MT5 (como <DATE> <TIME> <OPEN> ... <TICKVOL>). Sin Yahoo.")

# =========================
# CSV parsing (robusto MT5)
# =========================
def norm_col(c: str) -> str:
    # "<DATE>" -> "date"
    c = str(c).strip()
    c = c.replace("\ufeff", "")  # BOM
    c = c.lower()
    c = c.replace(" ", "_")
    c = re.sub(r"[<>]", "", c)
    c = re.sub(r"[^a-z0-9_]", "", c)
    return c

def infer_symbol_from_filename(name: str) -> str:
    base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    base = base.rsplit(".", 1)[0].strip().upper()
    base = re.split(r"[,\s;()\-]+", base)[0].upper()
    return base

def to_numeric(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.replace(" ", "", regex=False)
    # decimal coma si no hay punto
    mask = x.str.contains(",") & (~x.str.contains(r"\.", regex=True))
    x.loc[mask] = x.loc[mask].str.replace(",", ".", regex=False)
    # quitar separador de miles
    x = x.str.replace(",", "", regex=False)
    return pd.to_numeric(x, errors="coerce")

def read_mt5_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Lee CSV/tab/;/(whitespace) con headers tipo <DATE>.
    """
    text = file_bytes.decode("utf-8", errors="ignore")
    sample = text[:4096]

    # 1) Intentar sniffer con delimiters típicos
    sniffed = None
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t;,")
        sniffed = dialect.delimiter
    except Exception:
        sniffed = None

    tries = []
    if sniffed:
        tries.append(("sniff", sniffed))
    tries += [("tab", "\t"), ("semi", ";"), ("comma", ","), ("ws", r"\s+")]

    last_err = None
    for name, sep in tries:
        try:
            if sep == r"\s+":
                df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
            else:
                df = pd.read_csv(io.StringIO(text), sep=sep)
            if df is None or df.empty:
                continue
            df.columns = [norm_col(c) for c in df.columns]
            # si solo hay 1 columna, el separador no sirvió
            if df.shape[1] < 5:
                continue
            return df
        except Exception as e:
            last_err = e
            continue

    # último recurso: engine python con sep=None
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        df.columns = [norm_col(c) for c in df.columns]
        return df
    except Exception:
        raise RuntimeError(f"No pude leer CSV. Último error: {last_err}")

def pick_col(cols: set, *cands):
    for c in cands:
        if c in cols:
            return c
    return None

def parse_mt5_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Soporta tu formato:
    date time open high low close tickvol vol spread
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    cols = set(df.columns)

    date_col = pick_col(cols, "date")
    time_col = pick_col(cols, "time")
    dt_col   = pick_col(cols, "datetime", "timestamp")

    # datetime
    if dt_col is not None:
        dt = pd.to_datetime(df[dt_col], errors="coerce")
    # DATE + TIME (tu caso)
    elif date_col is not None and time_col is not None:
        combo = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
        # tu formato exacto: 2024.01.01 23:00:00
        dt = pd.to_datetime(combo, format="%Y.%m.%d %H:%M:%S", errors="coerce")
        # fallback por si algunos vienen raro
        if dt.isna().mean() > 0.2:
            dt = pd.to_datetime(combo, errors="coerce")
    elif date_col is not None:
        dt = pd.to_datetime(df[date_col], errors="coerce")
    else:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    out = df.copy()
    out["_dt"] = dt
    out = out.dropna(subset=["_dt"]).sort_values("_dt")

    c_open  = pick_col(cols, "open")
    c_high  = pick_col(cols, "high")
    c_low   = pick_col(cols, "low")
    c_close = pick_col(cols, "close", "last")

    if c_close is None:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    o = pd.DataFrame(index=pd.to_datetime(out["_dt"]).dt.tz_localize(None))
    o["Close"] = to_numeric(out[c_close])
    o["Open"]  = to_numeric(out[c_open]) if c_open else o["Close"]
    o["High"]  = to_numeric(out[c_high]) if c_high else o[["Open","Close"]].max(axis=1)
    o["Low"]   = to_numeric(out[c_low])  if c_low else o[["Open","Close"]].min(axis=1)

    # volumen: TICKVOL es lo que quieres
    c_vol = pick_col(cols, "tickvol", "tick_volume", "tickvolume", "real_volume", "volume", "vol")
    o["Volume"] = to_numeric(out[c_vol]) if c_vol else np.nan

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


# =========================
# Indicators / metrics
# =========================
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
    close = df["Close"].dropna()
    if len(close) < 3:
        return None

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
        "Avg rango %": avg_range,
        "ATR14 %": atr_pct,
        "ADX14": adx_last,
        "R2 tendencia": float(r2) if pd.notna(r2) else np.nan,
        "Vol prom": vol_mean,
        "Vol último": vol_last,
        "Clasificación": label,
        "Barras": int(len(close)),
        "Días": int((close.index.max().normalize() - close.index.min().normalize()).days) + 1
    }

def rolling_vol_peaks(close: pd.Series, win: int, top_n: int):
    close = close.dropna()
    ret = close.pct_change().dropna()
    if len(ret) < win + 5:
        return None, None, None
    roll = ret.rolling(win).std()*np.sqrt(252)
    peaks = roll.dropna().nlargest(top_n)
    table = pd.DataFrame({
        "Fecha": peaks.index,
        "Vol rolling (ann)": peaks.values,
        "Retorno 1": ret.reindex(peaks.index).values,
        "Retorno 5": close.pct_change(5).reindex(peaks.index).values,
    })
    return roll, peaks, table


# =========================
# Sidebar
# =========================
st.sidebar.header("📥 CSV MT5")
uploads = st.sidebar.file_uploader("Sube CSV(s)", type=["csv", "txt"], accept_multiple_files=True)

st.sidebar.markdown("---")
st.sidebar.header("⏱ Frecuencia de análisis")
freq = st.sidebar.selectbox("Resample", ["1min", "5min", "15min", "1H", "1D"], index=3)
rule = {"1min":"1T", "5min":"5T", "15min":"15T", "1H":"1H", "1D":"1D"}[freq]
bars_per_day = {"1T":1440, "5T":288, "15T":96, "1H":24, "1D":1}[rule]

st.sidebar.markdown("---")
st.sidebar.header("📏 Ventanas (en DÍAS)")
roll_vol_days = st.sidebar.slider("Ventana vol rolling (días)", 1, 180, 30, step=1)
roll_corr_days = st.sidebar.slider("Ventana rolling corr (días)", 1, 365, 90, step=1)
top_dd_n = st.sidebar.selectbox("Top drawdowns", [3,5,10], index=1)

# Convertimos días a barras para la frecuencia elegida
roll_vol_win = max(2, int(roll_vol_days * bars_per_day))
roll_corr_win = max(2, int(roll_corr_days * bars_per_day))
st.sidebar.caption(f"Equivalencia: vol={roll_vol_win} barras, corr={roll_corr_win} barras")

if not uploads:
    st.info("👈 Sube tu CSV MT5. Tu formato <DATE><TIME><OPEN>...<TICKVOL> ya está soportado.")
    st.stop()

# override de símbolo
with st.sidebar.expander("🧷 Símbolo por archivo", expanded=False):
    overrides = {}
    for f in uploads:
        guess = infer_symbol_from_filename(f.name)
        sym = st.text_input(f.name, value=guess, key=f"sym_{f.name}")
        overrides[f.name] = sym.strip().upper()

# =========================
# Load + parse
# =========================
series = {}
meta = []
for f in uploads:
    sym = overrides.get(f.name, infer_symbol_from_filename(f.name))
    try:
        raw = read_mt5_csv(f.getvalue())
        ohlc = parse_mt5_ohlcv(raw)
        rs = resample_ohlcv(ohlc, rule)
    except Exception as e:
        rs = pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    if not rs.empty:
        if sym in series:
            comb = pd.concat([series[sym], rs]).sort_index()
            comb = comb[~comb.index.duplicated(keep="last")]
            series[sym] = comb
        else:
            series[sym] = rs

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
    st.error("No pude parsear ningún archivo. Confirma que el CSV tenga <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE>.")
    st.stop()

symbols = sorted(series.keys())
# rango global
gmin = min(series[s].index.min() for s in symbols)
gmax = max(series[s].index.max() for s in symbols)

st.sidebar.markdown("---")
st.sidebar.header("🗓 Rango")
start = st.sidebar.date_input("Inicio", value=gmin.date())
end   = st.sidebar.date_input("Fin", value=gmax.date())

def slice_range(df):
    m = (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    return df.loc[m].copy()

data = {s: slice_range(series[s]) for s in symbols}
symbols = [s for s in symbols if not data[s].empty]

if not symbols:
    st.error("Con ese rango no quedó data.")
    st.stop()

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
    if len(close) >= 5:
        rets_cols[s] = close.pct_change()

summary = pd.DataFrame(rows).set_index("Símbolo").sort_values("Vol anualizada", ascending=False)
rets_df = pd.DataFrame(rets_cols)

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "✅ Resumen",
    "🔍 Detalle + Drawdowns",
    "🔗 Correlaciones",
    "🌡 Estrés vs Calma",
    "🧨 Eventos + Comparador",
])

with tab1:
    st.subheader("Resumen")
    st.dataframe(
        summary.style.format({
            "Precio":"{:,.6f}",
            "Retorno total":"{:.2%}",
            "CAGR":"{:.2%}",
            "Vol anualizada":"{:.2%}",
            "Sharpe(~rf0)":"{:,.2f}",
            "MaxDD":"{:.2%}",
            "Avg rango %":"{:.2%}",
            "ATR14 %":"{:.2%}",
            "ADX14":"{:,.2f}",
            "R2 tendencia":"{:,.2f}",
            "Vol prom":"{:,.0f}",
            "Vol último":"{:,.0f}",
        }),
        use_container_width=True
    )

with tab2:
    st.subheader("Detalle + drawdowns")
    sym = st.selectbox("Símbolo", options=list(summary.index), index=0)
    df = data[sym]
    close = df["Close"].dropna()

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
    st.markdown(f"### Top {top_dd_n} drawdowns (peak→trough)")
    st.dataframe(ev.head(top_dd_n), use_container_width=True)

with tab3:
    st.subheader("Correlación normal + Rolling Corr + Clustering")
    cols = [c for c in rets_df.columns if rets_df[c].dropna().shape[0] >= 20]
    if len(cols) < 2:
        st.info("Necesitas al menos 2 activos con retornos suficientes.")
    else:
        r = rets_df[cols].copy()
        corr = r.corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", title="Matriz de correlación"), use_container_width=True)

        colA, colB = st.columns(2)
        a = colA.selectbox("A", options=cols, index=0)
        b = colB.selectbox("B", options=cols, index=1)
        if a != b:
            ab = pd.concat([r[a], r[b]], axis=1).dropna()
            if len(ab) < roll_corr_win + 5:
                st.info("Poca data para rolling corr con esa ventana. Baja ventana o amplía rango.")
            else:
                rc = ab[a].rolling(roll_corr_win).corr(ab[b])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rc.index, y=rc.values, mode="lines"))
                fig.add_hline(y=0)
                fig.update_layout(title=f"Rolling Corr ({roll_corr_win} barras): {a} vs {b}", height=260, margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig, use_container_width=True)

        if SCIPY_OK and len(cols) >= 3:
            dist = np.sqrt(0.5*(1-corr.fillna(0)))
            dist_cond = squareform(dist.values, checks=False)
            Z = linkage(dist_cond, method="average")
            fig = ff.create_dendrogram(dist.values, labels=list(corr.columns), linkagefun=lambda _: Z)
            fig.update_layout(height=420, title="Dendrograma (distancia por correlación)")
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Estrés vs Calma (Calm/Mid/Stress) y Stress–Calm")
    if benchmark not in rets_df.columns or rets_df[benchmark].dropna().shape[0] < roll_vol_win + 10:
        st.info("Benchmark sin suficiente historia para regímenes con tu ventana. Baja ventana o amplía rango.")
    else:
        bench = rets_df[benchmark].dropna()
        vol = bench.rolling(roll_vol_win).std().dropna()
        if len(vol) < 30:
            st.info("No hay suficiente data para separar regímenes.")
        else:
            q_low, q_high = vol.quantile([0.40, 0.70])
            reg = pd.Series(index=vol.index, dtype="object")
            reg[vol < q_low] = "Calm"
            reg[(vol >= q_low) & (vol < q_high)] = "Mid"
            reg[vol >= q_high] = "Stress"

            st.dataframe(reg.value_counts().to_frame("barras"), use_container_width=True)

            cols = [c for c in rets_df.columns if rets_df[c].dropna().shape[0] >= 30]
            r = rets_df[cols].copy()

            calm_idx = reg[reg=="Calm"].index
            stress_idx = reg[reg=="Stress"].index

            r_calm = r.loc[r.index.intersection(calm_idx)].dropna(how="any")
            r_stress = r.loc[r.index.intersection(stress_idx)].dropna(how="any")

            if len(r_calm) < 20 or len(r_stress) < 20:
                st.info("Muy pocos puntos Calm/Stress para comparar correlaciones.")
            else:
                c_calm = r_calm.corr()
                c_stress = r_stress.corr()
                diff = (c_stress - c_calm).fillna(0)

                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(px.imshow(c_calm, text_auto=".2f", aspect="auto", title="Corr – Calm"), use_container_width=True)
                with col2:
                    st.plotly_chart(px.imshow(c_stress, text_auto=".2f", aspect="auto", title="Corr – Stress"), use_container_width=True)

                st.plotly_chart(px.imshow(diff, text_auto=".2f", aspect="auto", title="Stress - Calm"), use_container_width=True)

with tab5:
    st.subheader("Eventos extremos + Comparador")
    subA, subB = st.tabs(["🔥 Picos de volatilidad", "🆚 Comparador"])

    with subA:
        sym = st.selectbox("Símbolo", options=symbols, index=0, key="peaks_sym")
        top_n = st.slider("Top N picos", 5, 30, 10, step=1)
        close = data[sym]["Close"].dropna()

        roll, peaks, table = rolling_vol_peaks(close, win=roll_vol_win, top_n=top_n)
        if roll is None:
            st.info("Poca historia para picos con esa ventana.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines"))
            for d in peaks.index:
                fig.add_vline(x=d, line_dash="dash", opacity=0.4)
            fig.update_layout(title=f"{sym} – Vol rolling ({roll_vol_win} barras)", height=280, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(table, use_container_width=True)

    with subB:
        col1, col2 = st.columns(2)
        a = col1.selectbox("Activo A", options=symbols, index=0, key="cmp_a")
        b = col2.selectbox("Activo B", options=symbols, index=1 if len(symbols)>1 else 0, key="cmp_b")

        dfA = data[a][["Close","Volume"]].rename(columns={"Close":"A_Close","Volume":"A_Vol"})
        dfB = data[b][["Close","Volume"]].rename(columns={"Close":"B_Close","Volume":"B_Vol"})
        df = dfA.join(dfB, how="inner").dropna(subset=["A_Close","B_Close"])

        if len(df) < 20:
            st.info("No hay suficiente traslape entre ambos activos.")
        else:
            norm_a = df["A_Close"]/df["A_Close"].iloc[0]
            norm_b = df["B_Close"]/df["B_Close"].iloc[0]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=norm_a, mode="lines", name=f"{a}"))
            fig.add_trace(go.Scatter(x=df.index, y=norm_b, mode="lines", name=f"{b}"))
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
            fig.update_layout(title="Volumen (MA20) — usando TICKVOL", height=240, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Disclaimer: análisis estadístico; no es asesoría financiera.")
