
import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.stats import norm, linregress

st.set_page_config(
    page_title="FINTECH | Valuación de Activos",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Estilo
# -----------------------------
st.markdown("""
<style>
    .stApp { background: #07111f; color: white; }
    [data-testid="stSidebar"] { background: #0b1728; }
    .block-container { padding-top: 1.2rem; }
    h1, h2, h3, h4, p, label, span, div { font-family: Arial, sans-serif; }
    h1, h2, h3, h4, p, label, span { color: white !important; }
    .metric-card {
        background: linear-gradient(135deg, #0d2138, #07111f);
        border: 1px solid #164d80;
        border-radius: 12px;
        padding: 14px;
        min-height: 110px;
    }
    .metric-title { font-size: 13px; color: #a8c7e8; }
    .metric-value { font-size: 25px; font-weight: 700; color: white; }
    .metric-help { font-size: 11px; color: #8ca8c5; }
    .stButton > button {
        background: #0b6fc4;
        color: white;
        border: 0;
        border-radius: 8px;
        font-weight: 700;
    }
    .stDownloadButton > button {
        background: #12304d;
        color: white;
        border: 1px solid #24689c;
    }
    [data-testid="stDataFrame"] { background: #0b1728; }
</style>
""", unsafe_allow_html=True)

st.title("📈 FINTECH | Valuación y desempeño de activos")
st.caption("Retorno • Volatilidad • Sharpe • Treynor • Correlación • Beta • CAPM • Alpha • VaR")

# -----------------------------
# Funciones
# -----------------------------
LOOKBACKS = {
    "5 días": 5,
    "3 meses": 63,
    "6 meses": 126,
    "YTD": None,
    "12 meses": 252,
    "1 año": 252,
    "5 años": 1260,
}

PERIODS_PER_YEAR = {
    "Diaria": 252,
    "Semanal": 52,
    "Mensual": 12,
}

def get_z(confidence: float) -> float:
    # Para VaR de pérdida: cuantil inferior 1-confidence.
    return norm.ppf(1 - confidence)

def annualized_return(prices: pd.Series, periods_per_year: int) -> float:
    prices = prices.dropna()
    if len(prices) < 2 or prices.iloc[0] <= 0:
        return np.nan
    elapsed_periods = len(prices) - 1
    years = elapsed_periods / periods_per_year
    if years <= 0:
        return np.nan
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1

def annualized_volatility(returns: pd.Series, periods_per_year: int) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    return r.std(ddof=1) * np.sqrt(periods_per_year)

def align_series(asset_returns: pd.Series, benchmark_returns: pd.Series):
    df = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    df.columns = ["asset", "market"]
    return df

def calculate_metrics(prices, benchmark_prices, rf, periods_per_year, capital, confidence, var_days):
    returns = prices.pct_change().dropna()
    market_returns = benchmark_prices.pct_change().dropna()
    aligned = align_series(returns, market_returns)

    ret_ann = annualized_return(prices, periods_per_year)
    vol_ann = annualized_volatility(returns, periods_per_year)

    # Fórmulas del documento:
    # Sharpe = (Rp - Rf) / sigma_p
    sharpe = (ret_ann - rf) / vol_ann if vol_ann and np.isfinite(vol_ann) else np.nan

    # Beta = Cov(Ri,Rm) / Var(Rm)
    beta = (
        aligned["asset"].cov(aligned["market"]) /
        aligned["market"].var(ddof=1)
        if len(aligned) > 1 and aligned["market"].var(ddof=1) != 0 else np.nan
    )

    corr = aligned["asset"].corr(aligned["market"])

    # Treynor = (Ra - Rf) / beta
    treynor = (ret_ann - rf) / beta if beta and np.isfinite(beta) else np.nan

    market_ann = annualized_return(benchmark_prices, periods_per_year)

    # CAPM = Rf + beta(Rm - Rf)
    capm = rf + beta * (market_ann - rf) if np.isfinite(beta) and np.isfinite(market_ann) else np.nan

    # Alpha = Ri - [Rf + beta(Rm-Rf)]
    alpha = ret_ann - capm if np.isfinite(capm) else np.nan

    # VaR: VaR_alpha = mu + z_alpha*sigma
    # Se reporta como pérdida positiva: -(retorno esperado adverso).
    if var_days == 1:
        horizon_factor = 1
    else:
        horizon_factor = np.sqrt(var_days)

    mu_h = returns.mean() * var_days
    sigma_h = returns.std(ddof=1) * horizon_factor
    z = get_z(confidence)
    var_return = -(mu_h + z * sigma_h)
    var_return = max(0.0, var_return)
    var_dollar = capital * var_return

    slope, intercept, r_value, p_value, std_err = (np.nan,) * 5
    if len(aligned) >= 3:
        reg = linregress(aligned["market"], aligned["asset"])
        slope, intercept, r_value, p_value, std_err = reg

    return {
        "Rentabilidad anualizada": ret_ann,
        "Volatilidad anualizada": vol_ann,
        "iSharpe": sharpe,
        "iTreynor": treynor,
        "Coef. Correlación Pearson": corr,
        "BETA": beta,
        "CAPM": capm,
        "Alpha": alpha,
        "Valor z": z,
        "VaR %": var_return,
        "VaR $$": var_dollar,
        "Retorno mercado anualizado": market_ann,
        "Regresión R²": r_value ** 2 if np.isfinite(r_value) else np.nan,
        "Observaciones": len(aligned),
    }

def select_start_date(lookback):
    today = pd.Timestamp.today().normalize()
    if lookback == "YTD":
        return pd.Timestamp(year=today.year, month=1, day=1)
    days = LOOKBACKS[lookback]
    # margen para fines de semana / feriados
    # Yahoo necesita días calendario adicionales por fines de semana,
    # feriados y sesiones sin cotización.
    return today - pd.Timedelta(days=max(30, int(days * 2.0) + 15))

def normalize_download(data, ticker):
    """
    Extrae Close/Adj Close de las distintas estructuras MultiIndex
    que puede devolver yfinance. La posición de Ticker y Campo puede
    variar según la versión de yfinance.
    """
    if data is None or data.empty:
        return pd.Series(dtype=float)

    # DataFrame simple: una sola serie.
    if not isinstance(data.columns, pd.MultiIndex):
        for field in ["Adj Close", "Close"]:
            if field in data.columns:
                return pd.to_numeric(data[field], errors="coerce").dropna()
        return pd.Series(dtype=float)

    # MultiIndex: localizar explícitamente ticker y campo, sin asumir
    # si están en el nivel 0 o en el nivel 1.
    fields = {"Adj Close", "Close"}
    candidates = []

    for col in data.columns:
        parts = [str(x) for x in col]
        if ticker in parts:
            field_parts = [x for x in parts if x in fields]
            if field_parts:
                # Preferimos Adj Close cuando exista.
                candidates.append((field_parts[0], col))

    for preferred_field in ["Adj Close", "Close"]:
        for field, col in candidates:
            if field == preferred_field:
                s = data[col]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                return pd.to_numeric(s, errors="coerce").dropna()

    # Fallback: buscar cualquier columna Close/Adj Close.
    for col in data.columns:
        parts = [str(x) for x in col]
        if any(x in fields for x in parts):
            s = data[col]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return pd.to_numeric(s, errors="coerce").dropna()

    return pd.Series(dtype=float)

def download_prices(tickers, start, end, interval):
    data = yf.download(
        tickers=tickers,
        start=start.date(),
        end=(end + pd.Timedelta(days=1)).date(),
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    result = {}
    for t in tickers:
        result[t] = normalize_download(data, t)
    return result

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Parámetros")

    n_assets = st.number_input(
        "Número de activos a valuar",
        min_value=1, max_value=20, value=3, step=1
    )

    default_tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
    tickers = []
    for i in range(int(n_assets)):
        t = st.text_input(
            f"Ticker {i+1}",
            value=default_tickers[i] if i < len(default_tickers) else "",
            key=f"ticker_{i}"
        ).strip().upper()
        if t:
            tickers.append(t)

    benchmark = st.text_input("Índice bursátil de referencia", value="^GSPC").strip().upper()

    st.divider()
    country = st.selectbox(
        "País de referencia de la tasa libre de riesgo",
        ["México", "Estados Unidos", "Canadá", "Reino Unido", "Zona Euro", "Otro"]
    )
    rf = st.number_input(
        "Tasa libre de riesgo anual (%)",
        min_value=-20.0, max_value=30.0, value=4.00, step=0.05,
        help="Captura la tasa libre de riesgo correspondiente al país y al periodo analizado."
    ) / 100

    capital = st.number_input(
        "Capital a invertir para VaR ($)",
        min_value=0.0, value=1_000_000.0, step=50_000.0
    )

    confidence_pct = st.selectbox(
        "Intervalo de confianza VaR",
        [90, 95, 99], index=1
    )
    confidence = confidence_pct / 100

    var_horizon = st.selectbox("Plazo para VaR", ["1 día", "1 mes"])
    var_days = 1 if var_horizon == "1 día" else 21

    frequency = st.selectbox(
        "Periodicidad de precios",
        ["Diaria", "Semanal", "Mensual"]
    )

    lookback = st.selectbox(
        "Plazo a calcular",
        list(LOOKBACKS.keys()),
        index=4
    )

    run = st.button("🚀 Calcular valuación", use_container_width=True)

st.info(
    "La tasa libre de riesgo es un input de control: debe corresponder al país de origen "
    "de los activos y al periodo seleccionado. Yahoo Finance se utiliza para precios históricos."
)

if not run:
    st.markdown("### Instrucciones")
    st.write("1. Captura los tickers. 2. Selecciona el índice de referencia. "
             "3. Define tasa libre de riesgo, capital, confianza, VaR, periodicidad y plazo. "
             "4. Presiona **Calcular valuación**.")
    st.stop()

if not tickers:
    st.error("Captura al menos un ticker.")
    st.stop()

if not benchmark:
    st.error("Captura el índice bursátil de referencia.")
    st.stop()

# -----------------------------
# Descarga de datos
# -----------------------------
interval_map = {"Diaria": "1d", "Semanal": "1wk", "Mensual": "1mo"}
interval = interval_map[frequency]
ppy = PERIODS_PER_YEAR[frequency]

start = select_start_date(lookback)
end = pd.Timestamp.today().normalize()

all_tickers = list(dict.fromkeys(tickers + [benchmark]))
with st.spinner("Descargando precios históricos de Yahoo Finance..."):
    series_map = download_prices(all_tickers, start, end, interval)

benchmark_prices = series_map.get(benchmark, pd.Series(dtype=float))

if benchmark_prices.empty:
    st.error(f"No fue posible obtener datos para el índice de referencia: {benchmark}")
    st.stop()

# -----------------------------
# Cálculo
# -----------------------------
rows = []
series_for_corr = {}
regression_data = {}

for ticker in tickers:
    prices = series_map.get(ticker, pd.Series(dtype=float))
    if prices.empty or len(prices) < 3:
        st.warning(f"No hay suficientes datos para {ticker}. Se omitirá.")
        continue

    # Alineación temporal del activo y benchmark.
    common = pd.concat([prices.rename("asset"), benchmark_prices.rename("market")], axis=1).dropna()
    if len(common) < 3:
        st.warning(f"No hay suficientes observaciones coincidentes para {ticker}.")
        continue

    metrics = calculate_metrics(
        common["asset"],
        common["market"],
        rf,
        ppy,
        capital,
        confidence,
        var_days,
    )
    metrics["Activo"] = ticker
    rows.append(metrics)

    returns_df = common.pct_change().dropna()
    series_for_corr[ticker] = returns_df["asset"]

    regression_data[ticker] = pd.DataFrame({
        "Mercado": common["market"].pct_change(),
        "Activo": common["asset"].pct_change()
    }).dropna()

if not rows:
    st.error("No fue posible calcular métricas con los datos descargados.")
    st.stop()

results = pd.DataFrame(rows).set_index("Activo")

# -----------------------------
# Dashboard
# -----------------------------
st.subheader("📊 Dashboard ejecutivo")
st.caption(
    f"País RF: {country} | RF: {rf:.2%} | Benchmark: {benchmark} | "
    f"Periodo: {lookback} | Frecuencia: {frequency} | VaR: {var_horizon} | Confianza: {confidence_pct}%"
)

tabs = st.tabs(["Resumen", "Correlación", "Regresión vs índice", "Datos", "Metodología"])

with tabs[0]:
    cols = st.columns(min(4, len(results)))
    for idx, ticker in enumerate(results.index):
        m = results.loc[ticker]
        with cols[idx % len(cols)]:
            st.markdown(f"#### {ticker}")
            st.metric("Rentabilidad anualizada", f"{m['Rentabilidad anualizada']:.2%}")
            st.metric("Volatilidad anualizada", f"{m['Volatilidad anualizada']:.2%}")
            st.metric("Sharpe", f"{m['iSharpe']:.2f}")
            st.metric("Treynor", f"{m['iTreynor']:.2%}")
            st.metric("Beta", f"{m['BETA']:.2f}")
            st.metric("Alpha", f"{m['Alpha']:.2%}")
            st.metric("CAPM", f"{m['CAPM']:.2%}")
            st.metric("VaR $", f"${m['VaR $$']:,.2f}")
            st.metric("VaR %", f"{m['VaR %']:.2%}")

    st.markdown("### Comparativo de indicadores")
    display = results[
        [
            "Rentabilidad anualizada", "Volatilidad anualizada", "iSharpe",
            "iTreynor", "Coef. Correlación Pearson", "BETA",
            "CAPM", "Alpha", "VaR %", "VaR $$"
        ]
    ].copy()
    pct_cols = [
        "Rentabilidad anualizada", "Volatilidad anualizada",
        "iTreynor", "CAPM", "Alpha", "VaR %"
    ]
    for c in pct_cols:
        display[c] = display[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/D")
    for c in ["iSharpe", "Coef. Correlación Pearson", "BETA"]:
        display[c] = display[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/D")
    display["VaR $$"] = display["VaR $$"].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/D")
    st.dataframe(display, use_container_width=True)

with tabs[1]:
    st.markdown("### Matriz de correlación de retornos")
    corr_df = pd.DataFrame(series_for_corr).corr()
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        title="Correlación de Pearson",
    )
    fig.update_layout(template="plotly_dark", height=520)
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    selected = st.selectbox("Activo para regresión", list(results.index))
    reg_df = regression_data[selected]

    x = reg_df["Mercado"]
    y = reg_df["Activo"]
    reg = linregress(x, y)
    y_hat = reg.intercept + reg.slope * x

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name=selected,
        opacity=0.55
    ))
    order = np.argsort(x.values)
    fig.add_trace(go.Scatter(
        x=x.iloc[order], y=y_hat.iloc[order],
        mode="lines", name="Regresión"
    ))
    fig.update_layout(
        template="plotly_dark",
        title=f"{selected} vs {benchmark}",
        xaxis_title=f"Retorno {benchmark}",
        yaxis_title=f"Retorno {selected}",
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)

    m = results.loc[selected]
    c1, c2, c3 = st.columns(3)
    c1.metric("Beta", f"{m['BETA']:.4f}")
    c2.metric("Correlación", f"{m['Coef. Correlación Pearson']:.4f}")
    c3.metric("R²", f"{m['Regresión R²']:.4f}")

with tabs[3]:
    st.markdown("### Precios históricos")
    selected_data = st.selectbox("Activo / índice", all_tickers)
    px_series = series_map.get(selected_data, pd.Series(dtype=float))
    if not px_series.empty:
        price_df = px_series.rename("Precio").to_frame()
        st.line_chart(price_df)
        st.dataframe(price_df.tail(100), use_container_width=True)

    st.markdown("### Exportar resultados")
    csv = results.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar métricas CSV",
        data=csv,
        file_name="metricas_valuacion_activos.csv",
        mime="text/csv"
    )

with tabs[4]:
    st.markdown("### Fórmulas utilizadas")
    st.latex(r"Retorno\ Anual = \left(\frac{Valor\ Final}{Valor\ Inicial}\right)^{1/n}-1")
    st.latex(r"Volatilidad\ Anual = \sigma\sqrt{n}")
    st.latex(r"Sharpe = \frac{R_p-R_f}{\sigma_p}")
    st.latex(r"Treynor = \frac{R_a-R_f}{\beta_a}")
    st.latex(r"\beta = \frac{Cov(R_i,R_m)}{\sigma_m^2}")
    st.latex(r"CAPM = R_f+\beta_i(R_m-R_f)")
    st.latex(r"\alpha = R_i-[R_f+\beta_i(R_m-R_f)]")
    st.latex(r"VaR_\alpha = \mu+z_\alpha\sigma")

    st.markdown(
        """
        **Interpretación rápida**
        - **Rentabilidad anualizada:** ganancia/pérdida anual equivalente.
        - **Volatilidad:** variabilidad anualizada de los retornos.
        - **Sharpe:** retorno excedente por unidad de volatilidad total.
        - **Treynor:** retorno excedente por unidad de riesgo sistemático.
        - **Correlación Pearson:** asociación lineal entre activo y benchmark.
        - **Beta:** sensibilidad del activo frente al mercado.
        - **CAPM:** retorno esperado según riesgo sistemático.
        - **Alpha:** diferencia entre retorno observado y retorno CAPM.
        - **VaR:** pérdida estimada bajo el nivel de confianza y horizonte elegidos.
        """
    )
    st.warning(
        "El VaR es una métrica estadística de riesgo, no una garantía de pérdida máxima. "
        "El modelo implementa el enfoque normal indicado en el documento anexo."
    )

st.caption("Fuente de precios: Yahoo Finance vía yfinance. La tasa libre de riesgo se captura como input para mantener trazabilidad por país y periodo.")
