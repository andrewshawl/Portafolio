import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    import plotly.figure_factory as ff
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

st.set_page_config(page_title="MT5 Portfolio Lab", page_icon="📈", layout="wide")
st.title("📈 MT5 Portfolio Lab (MT5 CSV only)")
st.caption("Sube CSV(s) de MT5. No Yahoo. No bloqueos. Todo el análisis corre con tus datos.")

# ---------- CSV parsing (MT5 style) ----------
def norm_col(c: str) -> str:
    c = str(c).strip().lower()
    c = c.replace(" ", "_")
    c = re.sub(r"[<>]", "", c)          # <DATE> -> date
    c = re.sub(r"[^a-z0-9_]", "", c)
    return c

def infer_symbol_from_filename(name: str) -> str:
    base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    base = base.rsplit(".", 1)[0].strip().upper()
    base = re.split(r"[,\s;()\-]+", base)[0].upper()
    return base

def read_csv_flexible(file_bytes: bytes) -> pd.DataFrame:
    text = file_bytes.decode("utf-8", errors="ignore")
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    df.columns = [norm_col(c) for c in df.columns]
    return df

def to_numeric(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.replace(" ", "", regex=False)
    mask = x.str.contains(",") & (~x.str.contains(r"\.", regex=True))
    x.loc[mask] = x.loc[mask].str.replace(",", ".", regex=False)
    x = x.str.replace(",", "", regex=False)
    return pd.to_numeric(x, errors="coerce")

def pick_col(cols: set, *cands):
    for c in cands:
        if c in cols:
            return c
    return None

def parse_mt5(df: pd.DataFrame) -> pd.DataFrame:
    """
    Soporta tu formato:
    <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    cols = set(df.columns)

    date_col = pick_col(cols, "date")
    time_col = pick_col(cols, "time")
    dt_col   = pick_col(cols, "datetime", "timestamp")

    if dt_col is not None:
        dt = pd.to_datetime(df[dt_col], errors="coerce")
    elif date_col is not None and time_col is not None:
        dt = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")
    elif date_col is not None:
        dt = pd.to_datetime(df[date_col], errors="coerce")
    else:
        # último recurso: busca algo que contenga date o time
        guess = None
        for c in df.columns:
            if "date" in c or "time" in c:
                guess = c
                break
        if guess is None:
            return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
        dt = pd.to_datetime(df[guess], errors="coerce")

    out = df.copy()
    out["_dt"] = dt
    out = out.dropna(subset=["_dt"]).sort_values("_dt")

    c_open  = pick_col(cols, "open")
    c_high  = pick_col(cols, "high")
    c_low   = pick_col(cols, "low")
    c_close = pick_col(cols, "close", "last")

    if c_close is None:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    o = pd.DataFrame(index=out["_dt"])
    o["Close"] = to_numeric(out[c_close])
    o["Open"]  = to_numeric(out[c_open]) if c_open else o["Close"]
    o["High"]  = to_numeric(out[c_high]) if c_high else o[["Open","Close"]].max(axis=1)
    o["Low"]   = to_numeric(out[c_low])  if c_low else o[["Open","Close"]].min(axis=1)

    # volumen: tu CSV trae tickvol
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

# ---------- Indicators ----------
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
    close = df["Close"].dropna()
    high  = df["High"].fillna(close)
    low   = df["Low"].fillna(close)

    last_dt = close.index[-1]
    week_start = (last_dt - pd.Timedelta(days=last_dt.weekday())).normalize()
    w = df[df.index >= week_start]
    if w.empty:
        w = df.tail(50)

    w_close = w["Close"].dropna()
    w_high = w["High"].fillna(w_close).dropna()
    w_low  = w["Low"].fillna(w_close).dropna()

    if len(w_close) < 2:
        return None

    w_ret = float(w_close.iloc[-1]/w_close.iloc[0] - 1)
    w_range = float((w_high.max() - w_low.min())/w_close.iloc[-1])
    w_vol = float(w_close.pct_change().dropna().std()*np.sqrt(252)) if len(w_close) > 3 else np.nan
    w_volsum = float(w["Volume"].replace(0, np.nan).dropna().sum()) if w["Volume"].notna().any() else np.nan

    return {
        "Semana inicio": w_close.index[0].date(),
        "Semana fin": w_close.index[-1].date(),
        "Retorno semana": w_ret,
        "Rango semana %": w_range,
        "Vol semana (ann)": w_vol,
        "Volumen semana (suma)": w_volsum,
    }

def rolling_vol_peaks(close: pd.Series, win: int, top_n: int):
    close = close.dropna()
    ret = close.pct_change().dropna()
    if len(ret) < win + 10:
        return None, None, None
    roll = ret.rolling(win).std()*np.sqrt(252)
    peaks = roll.dropna().nlargest(top_n)
    table = pd.DataFrame({
        "Fecha": peaks.index,
        f"Vol rolling {win} barras (ann)": peaks.values,
        "Retorno 1 barra": ret.reindex(peaks.index).values,
        "Retorno 5 barras": close.pct_change(5).reindex(peaks.index).values,
    })
    return roll, peaks, table

# ---------- UI ----------
st.sidebar.header("📥 Cargar CSV(s) de MT5")
uploads = st.sidebar.file_uploader("CSV(s)", type=["csv", "txt"], accept_multiple_files=True)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Frecuencia de análisis")
freq_label = st.sidebar.selectbox(
    "Analizar en",
    ["Diario (1D)", "1H", "15min", "5min", "1min"],
    index=0
)
freq_map = {"Diario (1D)": "1D", "1H": "1H", "15min": "15T", "5min": "5T", "1min": "1T"}
RESAMPLE_RULE = freq_map[freq_label]

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parámetros (con sentido)")
min_days = st.sidebar.slider(
    "Mínimo de historia (días) para incluir un símbolo",
    1, 365, 7, step=1,
    help="Filtro de calidad: si un símbolo trae 1-2 días, te rompe CAGR/ADX/correlaciones. Baja esto solo para probar."
)

roll_vol_days = st.sidebar.slider(
    "Ventana vol rolling (días)",
    1, 180, 30, step=1,
    help="Volatilidad móvil: últimos N días. Sirve para picos de volatilidad y para regímenes."
)

roll_corr_days = st.sidebar.slider(
    "Ventana rolling corr (días)",
    1, 365, 90, step=1,
    help="Correlación móvil: cómo cambia la correlación con el tiempo."
)

top_dd_n = st.sidebar.selectbox("Top drawdowns por activo", [3,5,10], index=1)

if not uploads:
    st.info("👈 Sube CSV(s) de MT5. Tu formato <DATE> <TIME> ... <TICKVOL> ya está soportado.")
    st.stop()

# Overrides por archivo
with st.sidebar.expander("🧷 Asignación de símbolo por archivo", expanded=False):
    overrides = {}
    for f in uploads:
        guess = infer_symbol_from_filename(f.name)
        sym = st.text_input(f.name, value=guess, key=f"sym_{f.name}")
        overrides[f.name] = sym.strip().upper()

# Cargar series por símbolo
series = {}
meta = []
for f in uploads:
    sym = overrides.get(f.name, infer_symbol_from_filename(f.name))
    raw = read_csv_flexible(f.getvalue())
    ohlcv = parse_mt5(raw)
    rs = resample_ohlcv(ohlcv, RESAMPLE_RULE)

    if not rs.empty:
        series[sym] = rs if sym not in series else (
            pd.concat([series[sym], rs]).sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]
        )

    meta.append({
        "Archivo": f.name,
        "Símbolo": sym,
        "Barras": int(len(rs)),
        "Desde": (rs.index.min().date().isoformat() if not rs.empty else "—"),
        "Hasta": (rs.index.max().date().isoformat() if not rs.empty else "—"),
        "Vol (tickvol)": ("sí" if (not rs.empty and rs["Volume"].notna().any()) else "no/NA")
    })

st.subheader(f"Estado de carga (resample a {freq_label})")
st.dataframe(pd.DataFrame(meta), use_container_width=True)

# Filtrar por historia mínima (días)
def span_days(df):
    if df.empty: return 0
    return int((df.index.max().normalize() - df.index.min().normalize()).days) + 1

symbols = [s for s, df in series.items() if span_days(df) >= min_days and len(df) >= 5]
if not symbols:
    st.error("No hay símbolos con suficiente historia. Baja 'mínimo de historia (días)' o sube más datos.")
    st.stop()

# Rango global
gmin = min(series[s].index.min() for s in symbols)
gmax = max(series[s].index.max() for s in symbols)

st.sidebar.markdown("---")
st.sidebar.header("🗓️ Rango de análisis")
start_date = st.sidebar.date_input("Inicio", value=gmin.date())
end_date   = st.sidebar.date_input("Fin", value=gmax.date())

def slice_range(df):
    m = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    return df.loc[m].copy()

data = {s: slice_range(series[s]) for s in symbols}
symbols = [s for s in symbols if not data[s].empty and span_days(data[s]) >= min_days]
if not symbols:
    st.error("Con ese rango se quedó sin data suficiente. Ajusta fechas o baja mínimo.")
    st.stop()

benchmark = st.sidebar.selectbox("Benchmark (para estrés)", options=symbols, index=0)

# Convert days -> bars for rolling ops
# Aproximación: número de barras por día según regla
bars_per_day = {"1D": 1, "1H": 24, "15T": 96, "5T": 288, "1T": 1440}[RESAMPLE_RULE]
roll_vol_win = max(2, int(roll_vol_days * bars_per_day))
roll_corr_win = max(2, int(roll_corr_days * bars_per_day))

st.sidebar.caption(f"Equivalencias: vol={roll_vol_win} barras, corr={roll_corr_win} barras (según {freq_label}).")

# Métricas y returns
rows = []
week_rows = []
rets_cols = {}

for s in symbols:
    df = data[s]
    m = compute_metrics(df)
    if m:
        m["Símbolo"] = s
        m["Hist (días)"] = span_days(df)
        rows.append(m)

    w = week_metrics(df)
    if w:
        w["Símbolo"] = s
        week_rows.append(w)

    close = df["Close"].dropna()
    if len(close) >= 10:
        rets_cols[s] = close.pct_change()

summary = pd.DataFrame(rows).set_index("Símbolo").sort_values("Vol anualizada", ascending=False)
weekdf  = pd.DataFrame(week_rows).set_index("Símbolo") if week_rows else pd.DataFrame()
rets_df = pd.DataFrame(rets_cols)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "✅ Resumen",
    "🔍 Detalle + Drawdowns",
    "🔗 Correlaciones",
    "🌡️ Estrés vs Calma",
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

with tab2:
    st.subheader("Detalle por símbolo + Top drawdowns")
    sym = st.selectbox("Símbolo", options=list(summary.index), index=0)

    df = data[sym]
    close = df["Close"].dropna()

    m = summary.loc[sym]
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Precio", f"{m['Precio']:,.6f}")
    c2.metric("Ret total", f"{m['Retorno total']*100:,.2f}%")
    c3.metric("Vol ann", f"{m['Vol anualizada']*100:,.2f}%" if pd.notna(m["Vol anualizada"]) else "—")
    c4.metric("MaxDD", f"{m['MaxDD']*100:,.2f}%")
    c5.metric("ADX14", f"{m['ADX14']:,.1f}" if pd.notna(m["ADX14"]) else "—")
    c6.metric("Hist(d)", f"{int(m['Hist (días)'])}")

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
    st.markdown(f"### Top {top_dd_n} peores drawdowns (peak → trough)")
    if ev.empty:
        st.info("No se detectaron drawdowns (o muy poca historia).")
    else:
        st.dataframe(ev.head(top_dd_n), use_container_width=True)

with tab3:
    st.subheader("Correlación normal + Rolling Corr + Clustering")
    valid_cols = [c for c in rets_df.columns if rets_df[c].dropna().shape[0] >= 10]
    if len(valid_cols) < 2:
        st.info("Necesitas al menos 2 símbolos con retornos suficientes.")
    else:
        r = rets_df[valid_cols].copy()
        corr = r.corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", title="Matriz de correlación"), use_container_width=True)

        colA, colB = st.columns(2)
        a = colA.selectbox("A", options=valid_cols, index=0, key="A")
        b = colB.selectbox("B", options=valid_cols, index=1, key="B")
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

        if SCIPY_OK and len(valid_cols) >= 3:
            st.markdown("### Clustering (dendrograma)")
            X = r.dropna().T.values  # activos como observaciones
            fig = ff.create_dendrogram(X, labels=list(r.columns))
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Estrés vs Calma (Calm/Mid/Stress) + Stress–Calm")
    if benchmark not in rets_df.columns or rets_df[benchmark].dropna().shape[0] < roll_vol_win + 10:
        st.info("Benchmark sin suficiente historia para regímenes con tu ventana. Baja ventana o amplía rango.")
    else:
        bench = rets_df[benchmark].dropna()
        vol = bench.rolling(roll_vol_win).std().dropna()
        if len(vol) < 20:
            st.info("No hay suficiente data para separar regímenes.")
        else:
            q_low, q_high = vol.quantile([0.40, 0.70])
            reg = pd.Series(index=vol.index, dtype="object")
            reg[vol < q_low] = "Calm"
            reg[(vol >= q_low) & (vol < q_high)] = "Mid"
            reg[vol >= q_high] = "Stress"

            st.dataframe(reg.value_counts().to_frame("barras"), use_container_width=True)

            valid_cols = [c for c in rets_df.columns if rets_df[c].dropna().shape[0] >= 20]
            r = rets_df[valid_cols].copy()

            calm_idx = reg[reg=="Calm"].index
            stress_idx = reg[reg=="Stress"].index

            r_calm = r.loc[r.index.intersection(calm_idx)].dropna(how="any")
            r_stress = r.loc[r.index.intersection(stress_idx)].dropna(how="any")

            if len(r_calm) < 10 or len(r_stress) < 10:
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

        if len(df) < 10:
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
            fig.update_layout(title="Volumen (MA20) si existe", height=240, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Disclaimer: análisis estadístico; no es asesoría financiera.")
