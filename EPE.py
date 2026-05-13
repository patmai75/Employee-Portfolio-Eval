from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


APP_TITLE = "Employee Portfolio Evaluation"
TRADING_DAYS = 252
DEFAULT_TICKER = "AAPL"
DEFAULT_BENCHMARK = "SPY"
TIME_WINDOWS: dict[str, str] = {
    "1m": "1mo",
    "6m": "6mo",
    "YTD": "ytd",
    "1y": "1y",
    "3y": "3y",
    "5y": "5y",
    "10y": "10y",
    "25y": "25y",
    "Max": "max",
}
REQUIRED_OPTION_COLUMNS = ["shares", "strike_price", "grant_date", "vesting_years"]


@dataclass(frozen=True)
class MarketMetrics:
    last_price: float
    previous_price: float
    min_price: float
    max_price: float
    annual_return: float
    annual_volatility: float
    cumulative_return: float
    max_drawdown: float
    sharpe_ratio: float


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Employee Portfolio Evaluation helps employees understand stock, option, "
            "vesting, sensitivity, and Monte Carlo outcomes. Educational only."
        )
    },
)


st.markdown(
    """
    <style>
        .main .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(18, 25, 38, 0.06), rgba(65, 105, 225, 0.04));
            border: 1px solid rgba(120, 144, 180, 0.18);
            border-radius: 18px;
            padding: 1rem;
        }
        .portfolio-hero {
            padding: 1.3rem 1.5rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #101828 0%, #1d4ed8 55%, #14b8a6 100%);
            color: white;
            margin-bottom: 1rem;
        }
        .portfolio-hero h1 {margin: 0; font-size: 2.3rem;}
        .portfolio-hero p {margin: .35rem 0 0 0; opacity: .9;}
        .small-note {font-size: .85rem; opacity: .72;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_history(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    """Download adjusted OHLCV data for one or more tickers using yfinance."""
    tickers = tuple(ticker.upper().strip() for ticker in tickers if ticker.strip())
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers=list(tickers),
        period=period,
        interval="1d",
        auto_adjust=True,
        actions=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if data.empty:
        return pd.DataFrame()

    if len(tickers) == 1 and not isinstance(data.columns, pd.MultiIndex):
        data.columns = pd.MultiIndex.from_product([tickers, data.columns])

    return data.sort_index()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_profile(ticker: str) -> dict[str, Any]:
    """Fetch a lightweight company profile, preferring yfinance.fast_info when possible."""
    symbol = ticker.upper().strip()
    if not symbol:
        return {}

    profile: dict[str, Any] = {"symbol": symbol}
    stock = yf.Ticker(symbol)

    try:
        fast_info = dict(stock.fast_info or {})
        profile.update({f"fast_{key}": value for key, value in fast_info.items()})
    except Exception:
        pass

    try:
        info = stock.get_info() or {}
    except Exception:
        info = {}

    for key in [
        "longName",
        "shortName",
        "currency",
        "exchange",
        "sector",
        "industry",
        "website",
        "quoteType",
    ]:
        if info.get(key):
            profile[key] = info[key]

    return profile


def get_ticker_frame(history: pd.DataFrame, ticker: str) -> pd.DataFrame:
    symbol = ticker.upper().strip()
    if history.empty or symbol not in history.columns.get_level_values(0):
        return pd.DataFrame()
    frame = history[symbol].dropna(how="all").copy()
    frame.index = pd.to_datetime(frame.index)
    return frame


def calculate_metrics(data: pd.DataFrame, risk_free_rate: float = 0.0) -> MarketMetrics:
    close = data["Close"].dropna()
    high = data.get("High", close).dropna()
    low = data.get("Low", close).dropna()
    log_returns = np.log(close / close.shift(1)).dropna()

    last_price = float(close.iloc[-1])
    previous_price = float(close.iloc[-2]) if len(close) > 1 else last_price
    annual_return = float(log_returns.mean() * TRADING_DAYS) if not log_returns.empty else 0.0
    annual_volatility = float(log_returns.std() * np.sqrt(TRADING_DAYS)) if len(log_returns) > 1 else 0.0
    cumulative_return = float(close.iloc[-1] / close.iloc[0] - 1) if len(close) > 1 else 0.0
    running_max = close.cummax()
    max_drawdown = float((close / running_max - 1).min()) if len(close) > 1 else 0.0
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility else 0.0

    return MarketMetrics(
        last_price=last_price,
        previous_price=previous_price,
        min_price=float(low.min()),
        max_price=float(high.max()),
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        cumulative_return=cumulative_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=float(sharpe_ratio),
    )


def calculate_option_value(current_price: float, strike_price: float, shares: int) -> float:
    return max(0.0, current_price - strike_price) * shares


def normalize_options(df: pd.DataFrame | None, fallback_strike: float) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame(
            [{"shares": 0, "strike_price": fallback_strike, "grant_date": date.today(), "vesting_years": 3}]
        )
    else:
        df = df.copy()

    for column in REQUIRED_OPTION_COLUMNS:
        if column not in df.columns:
            if column == "grant_date":
                df[column] = date.today()
            elif column == "vesting_years":
                df[column] = 3
            elif column == "strike_price":
                df[column] = fallback_strike
            else:
                df[column] = 0

    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0).clip(lower=0).round().astype(int)
    df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce").fillna(fallback_strike).clip(lower=0)
    df["vesting_years"] = pd.to_numeric(df["vesting_years"], errors="coerce").fillna(3).clip(lower=0, upper=100)
    df["grant_date"] = pd.to_datetime(df["grant_date"], errors="coerce").fillna(pd.Timestamp.today()).dt.date
    return df[REQUIRED_OPTION_COLUMNS]


def enrich_options(options: pd.DataFrame, current_price: float, as_of: date) -> pd.DataFrame:
    df = normalize_options(options, current_price)
    df["vested_on"] = df.apply(
        lambda row: row["grant_date"] + timedelta(days=int(row["vesting_years"] * 365.25)), axis=1
    )
    df["is_vested"] = df["vested_on"] <= as_of
    df["intrinsic_value"] = df.apply(
        lambda row: calculate_option_value(current_price, float(row["strike_price"]), int(row["shares"])), axis=1
    )
    df["moneyness_pct"] = np.where(df["strike_price"] > 0, (current_price / df["strike_price"] - 1) * 100, np.nan)
    return df


def portfolio_values(shares: int, options: pd.DataFrame, price: float) -> dict[str, float]:
    option_values = [calculate_option_value(price, row.strike_price, int(row.shares)) for row in options.itertuples()]
    total_options = float(np.sum(option_values))
    vested_options = float(
        np.sum([value for value, row in zip(option_values, options.itertuples()) if bool(row.is_vested)])
    )
    stock_value = float(shares * price)
    return {
        "stock_value": stock_value,
        "vested_option_value": vested_options,
        "total_option_value": total_options,
        "vested_portfolio": stock_value + vested_options,
        "potential_portfolio": stock_value + total_options,
    }


def import_portfolio(contents: bytes) -> dict[str, Any]:
    df = pd.read_csv(io.BytesIO(contents))
    normalized = {column.lower().strip(): column for column in df.columns}
    df = df.rename(columns={original: clean for clean, original in normalized.items()})

    ticker = str(df.get("ticker", pd.Series([DEFAULT_TICKER])).dropna().iloc[0]).upper().strip()
    shares = int(pd.to_numeric(df.get("shares", pd.Series([0])).dropna().iloc[0], errors="coerce") or 0)
    option_rows = df[df.get("option_shares", pd.Series(index=df.index, dtype=float)).notna()].copy()

    if not option_rows.empty:
        options = pd.DataFrame(
            {
                "shares": option_rows.get("option_shares"),
                "strike_price": option_rows.get("strike_price"),
                "grant_date": option_rows.get("grant_date"),
                "vesting_years": option_rows.get("vesting_years", 3),
            }
        )
    else:
        options = pd.DataFrame(columns=REQUIRED_OPTION_COLUMNS)

    return {"ticker": ticker, "shares": shares, "options": normalize_options(options, 0.0)}


def export_portfolio(ticker: str, shares: int, options: pd.DataFrame) -> str:
    rows = [{"ticker": ticker.upper(), "shares": shares, "option_shares": "", "strike_price": "", "grant_date": "", "vesting_years": ""}]
    for row in normalize_options(options, 0.0).itertuples():
        rows.append(
            {
                "ticker": "",
                "shares": "",
                "option_shares": int(row.shares),
                "strike_price": float(row.strike_price),
                "grant_date": row.grant_date.isoformat(),
                "vesting_years": float(row.vesting_years),
            }
        )
    return pd.DataFrame(rows).to_csv(index=False)


def make_price_chart(data: pd.DataFrame, ticker: str, currency: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close", line={"width": 2.4}))
    if "Volume" in data and data["Volume"].notna().any():
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["Volume"],
                name="Volume",
                yaxis="y2",
                opacity=0.18,
                marker_color="#64748b",
            )
        )
    fig.update_layout(
        title=f"{ticker.upper()} price history",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency})",
        yaxis2={"title": "Volume", "overlaying": "y", "side": "right", "showgrid": False},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return fig


def make_comparison_chart(primary: pd.DataFrame, benchmark: pd.DataFrame, ticker: str, benchmark_ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=primary.index, y=primary["Close"] / primary["Close"].iloc[0], mode="lines", name=ticker.upper())
    )
    fig.add_trace(
        go.Scatter(
            x=benchmark.index,
            y=benchmark["Close"] / benchmark["Close"].iloc[0],
            mode="lines",
            name=benchmark_ticker.upper(),
        )
    )
    fig.update_layout(
        title="Normalized performance: employee stock vs benchmark",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return fig


def make_allocation_chart(values: dict[str, float]) -> go.Figure:
    labels = ["Shares", "Vested options", "Unvested option potential"]
    amounts = [
        values["stock_value"],
        values["vested_option_value"],
        max(0.0, values["total_option_value"] - values["vested_option_value"]),
    ]
    fig = go.Figure(data=[go.Pie(labels=labels, values=amounts, hole=0.58, sort=False)])
    fig.update_layout(title="Current portfolio composition", margin={"l": 20, "r": 20, "t": 70, "b": 20})
    return fig


def make_sensitivity_chart(shares: int, options: pd.DataFrame, current_price: float) -> go.Figure:
    adjustments = np.array([-1.0, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    prices = current_price * (1 + adjustments)
    values = []
    for price in prices:
        option_value = sum(calculate_option_value(float(price), row.strike_price, int(row.shares)) for row in options.itertuples())
        values.append(shares * float(price) + option_value)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[f"{change:+.0%}" for change in adjustments],
            y=values,
            marker_color=np.where(adjustments < 0, "#ef4444", "#2563eb"),
            name="Potential portfolio value",
        )
    )
    fig.update_layout(
        title="Sensitivity of potential portfolio value",
        xaxis_title="Stock price move",
        yaxis_title="Potential value",
        yaxis={"tickprefix": "$", "separatethousands": True},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return fig


@st.cache_data(show_spinner=False)
def run_monte_carlo(
    last_price: float,
    shares: int,
    option_shares: tuple[int, ...],
    strike_prices: tuple[float, ...],
    mu: float,
    sigma: float,
    years: int,
    simulations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    total_steps = TRADING_DAYS * years
    dt = 1 / TRADING_DAYS
    shocks = rng.standard_normal((simulations, total_steps))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    log_paths = np.empty((simulations, total_steps + 1), dtype=np.float64)
    log_paths[:, 0] = np.log(last_price)
    log_paths[:, 1:] = log_paths[:, [0]] + np.cumsum(increments, axis=1)
    price_paths = np.exp(log_paths)

    year_indices = np.arange(0, total_steps + 1, TRADING_DAYS)
    yearly_prices = price_paths[:, year_indices]
    stock_values = shares * yearly_prices

    if option_shares:
        option_share_array = np.array(option_shares, dtype=float)
        strike_array = np.array(strike_prices, dtype=float)
        option_values = np.maximum(yearly_prices[:, :, None] - strike_array[None, None, :], 0) * option_share_array
        total_option_values = option_values.sum(axis=2)
    else:
        total_option_values = np.zeros_like(stock_values)

    total_values = stock_values + total_option_values
    percentiles = {p: np.percentile(total_values, p, axis=0) for p in [5, 25, 50, 75, 95]}
    final_values = total_values[:, -1]
    return {
        "yearly_prices": yearly_prices,
        "sample_price_paths": price_paths[: min(250, simulations), :: max(1, TRADING_DAYS // 12)],
        "sample_time": np.linspace(0, years, price_paths[:1, :: max(1, TRADING_DAYS // 12)].shape[1]),
        "percentiles": percentiles,
        "mean_final": float(final_values.mean()),
        "probability_gain": float((final_values > total_values[:, 0]).mean()),
        "probability_double": float((final_values >= 2 * max(total_values[:, 0].mean(), 1)).mean()),
    }


def make_projection_chart(results: dict[str, Any], years: int) -> go.Figure:
    x = np.arange(0, years + 1)
    p = results["percentiles"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=p[95], line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=p[5],
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.14)",
            line={"width": 0},
            name="5–95% range",
        )
    )
    fig.add_trace(go.Scatter(x=x, y=p[75], line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=p[25],
            fill="tonexty",
            fillcolor="rgba(20, 184, 166, 0.18)",
            line={"width": 0},
            name="25–75% range",
        )
    )
    fig.add_trace(go.Scatter(x=x, y=p[50], mode="lines+markers", line={"color": "#1d4ed8", "width": 3}, name="Median"))
    fig.update_layout(
        title="Monte Carlo portfolio projection",
        xaxis_title="Years",
        yaxis_title="Portfolio value",
        yaxis={"tickprefix": "$", "separatethousands": True},
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return fig


def make_simulation_paths_chart(results: dict[str, Any], paths_to_show: int) -> go.Figure:
    fig = go.Figure()
    time_axis = results["sample_time"]
    for path in results["sample_price_paths"][:paths_to_show]:
        fig.add_trace(
            go.Scatter(x=time_axis, y=path, mode="lines", line={"width": 1}, opacity=0.28, showlegend=False)
        )
    fig.update_layout(
        title="Sample simulated stock price paths",
        xaxis_title="Years",
        yaxis_title="Stock price",
        yaxis={"tickprefix": "$", "separatethousands": True},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return fig


def initialize_state() -> None:
    defaults = {
        "ticker": DEFAULT_TICKER,
        "benchmark": DEFAULT_BENCHMARK,
        "shares": 0,
        "option_positions": pd.DataFrame(
            [{"shares": 0, "strike_price": 100.0, "grant_date": date.today(), "vesting_years": 3}]
        ),
        "uploader_key": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Guía / Guide")
        language = st.segmented_control("Idioma / Language", ["Español", "English"], default="Español")
        if language == "Español":
            st.markdown(
                """
                **Qué hace esta app**
                1. Importa o captura acciones y opciones de empleado.
                2. Descarga precios ajustados con `yfinance` y cachea datos para evitar llamadas innecesarias.
                3. Compara tu acción contra un benchmark como `SPY`.
                4. Calcula valor intrínseco, vesting, sensibilidad y escenarios Monte Carlo.

                **Privacidad:** los datos viven en tu sesión de Streamlit; no se guardan en una base de datos.

                **Nota:** herramienta educativa, no asesoría financiera, fiscal ni legal.
                """
            )
        else:
            st.markdown(
                """
                **What this app does**
                1. Imports or captures employee shares and option grants.
                2. Downloads adjusted prices with `yfinance` and caches data to reduce network calls.
                3. Compares employer stock against a benchmark such as `SPY`.
                4. Calculates intrinsic value, vesting, sensitivity, and Monte Carlo scenarios.

                **Privacy:** data stays in your Streamlit session; it is not stored in a database.

                **Note:** educational only; not financial, tax, or legal advice.
                """
            )
        st.divider()
        st.caption("Built with modern Streamlit patterns: tabs, cached data, data editor, column configs, and responsive Plotly charts.")


def main() -> None:
    initialize_state()
    render_sidebar()

    st.markdown(
        f"""
        <div class="portfolio-hero">
            <h1>💼 {APP_TITLE}</h1>
            <p>Stock + options + vesting + risk + scenarios, redesigned while preserving the original employee-portfolio workflow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls = st.container(border=True)
    with controls:
        left, middle, right, actions = st.columns([1.2, 1.2, 1.2, 1.4], vertical_alignment="bottom")
        with left:
            st.session_state.ticker = st.text_input("Company ticker", st.session_state.ticker).upper().strip()
        with middle:
            st.session_state.benchmark = st.text_input("Benchmark ETF/stock", st.session_state.benchmark).upper().strip()
        with right:
            time_window = st.selectbox("Analysis window", list(TIME_WINDOWS), index=4)
        with actions:
            uploaded_file = st.file_uploader(
                "Import CSV", type="csv", key=f"file_csv_{st.session_state.uploader_key}", label_visibility="collapsed"
            )
            if uploaded_file is not None:
                imported = import_portfolio(uploaded_file.getvalue())
                st.session_state.ticker = imported["ticker"]
                st.session_state.shares = imported["shares"]
                st.session_state.option_positions = imported["options"]
                st.session_state.uploader_key += 1
                st.toast("Portfolio imported", icon="✅")

        clear_col, download_col = st.columns([1, 3], vertical_alignment="center")
        with clear_col:
            if st.button("Reset portfolio", icon=":material/delete:"):
                for key in ["ticker", "benchmark", "shares", "option_positions"]:
                    st.session_state.pop(key, None)
                initialize_state()
                st.rerun()
        with download_col:
            st.download_button(
                "Download portfolio CSV",
                data=export_portfolio(
                    st.session_state.ticker,
                    int(st.session_state.shares),
                    normalize_options(st.session_state.option_positions, 0.0),
                ),
                file_name=f"{st.session_state.ticker or 'portfolio'}_employee_portfolio.csv",
                mime="text/csv",
                icon=":material/download:",
                type="primary",
            )

    ticker = st.session_state.ticker or DEFAULT_TICKER
    benchmark = st.session_state.benchmark or DEFAULT_BENCHMARK
    period = TIME_WINDOWS[time_window]

    with st.spinner("Loading market data from Yahoo Finance..."):
        history = fetch_history((ticker, benchmark), period)
        company_data = get_ticker_frame(history, ticker)
        benchmark_data = get_ticker_frame(history, benchmark)
        profile = fetch_profile(ticker)

    if company_data.empty:
        st.error(f"No market data found for `{ticker}` in the selected window. Check the ticker and try again.")
        st.stop()

    currency = str(profile.get("currency") or profile.get("fast_currency") or "USD")
    company_name = profile.get("longName") or profile.get("shortName") or ticker.upper()
    metrics = calculate_metrics(company_data)
    price_delta = metrics.last_price - metrics.previous_price
    price_delta_pct = price_delta / metrics.previous_price if metrics.previous_price else 0.0

    st.subheader(f"{company_name} ({ticker.upper()})")
    meta = " · ".join(str(value) for value in [profile.get("exchange"), profile.get("sector"), profile.get("industry")] if value)
    if meta:
        st.caption(meta)

    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Last price", f"{currency} {metrics.last_price:,.2f}", f"{price_delta_pct:+.2%}")
    kpi_cols[1].metric("Window return", f"{metrics.cumulative_return:+.2%}")
    kpi_cols[2].metric("Annual return", f"{metrics.annual_return:+.2%}")
    kpi_cols[3].metric("Annual volatility", f"{metrics.annual_volatility:.2%}")
    kpi_cols[4].metric("Max drawdown", f"{metrics.max_drawdown:.2%}")
    kpi_cols[5].metric("Sharpe", f"{metrics.sharpe_ratio:.2f}")

    overview_tab, portfolio_tab, scenarios_tab, projection_tab, data_tab = st.tabs(
        ["📈 Market", "💼 Portfolio", "🎚️ Sensitivity", "🔮 Projection", "🧾 Data"]
    )

    with overview_tab:
        chart_col, benchmark_col = st.columns([1.25, 1], gap="large")
        with chart_col:
            st.plotly_chart(make_price_chart(company_data, ticker, currency), width="stretch", theme="streamlit")
        with benchmark_col:
            if benchmark_data.empty:
                st.warning(f"No comparison data found for `{benchmark}`.")
            else:
                st.plotly_chart(make_comparison_chart(company_data, benchmark_data, ticker, benchmark), width="stretch", theme="streamlit")
        st.dataframe(
            company_data.tail(10).sort_index(ascending=False),
            width="stretch",
            column_config={
                "Open": st.column_config.NumberColumn(format=f"{currency} %.2f"),
                "High": st.column_config.NumberColumn(format=f"{currency} %.2f"),
                "Low": st.column_config.NumberColumn(format=f"{currency} %.2f"),
                "Close": st.column_config.NumberColumn(format=f"{currency} %.2f"),
                "Volume": st.column_config.NumberColumn(format="%d"),
            },
        )

    with portfolio_tab:
        st.session_state.shares = st.number_input(
            "Number of shares", min_value=0, value=int(st.session_state.shares), step=1
        )
        st.markdown("#### Option grants")
        edited_options = st.data_editor(
            normalize_options(st.session_state.option_positions, metrics.last_price),
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "shares": st.column_config.NumberColumn("Granted quantity", min_value=0, step=1, format="%d"),
                "strike_price": st.column_config.NumberColumn("Exercise price", min_value=0.0, format=f"{currency} %.2f"),
                "grant_date": st.column_config.DateColumn("Grant date"),
                "vesting_years": st.column_config.NumberColumn("Years to full vesting", min_value=0.0, max_value=100.0, step=0.25),
            },
            key="options_editor",
        )
        st.session_state.option_positions = normalize_options(edited_options, metrics.last_price)
        enriched_options = enrich_options(st.session_state.option_positions, metrics.last_price, date.today())
        values = portfolio_values(int(st.session_state.shares), enriched_options, metrics.last_price)

        value_cols = st.columns(5)
        value_cols[0].metric("Stock value", f"{currency} {values['stock_value']:,.0f}")
        value_cols[1].metric("Vested options", f"{currency} {values['vested_option_value']:,.0f}")
        value_cols[2].metric("Total options", f"{currency} {values['total_option_value']:,.0f}")
        value_cols[3].metric("Vested portfolio", f"{currency} {values['vested_portfolio']:,.0f}")
        value_cols[4].metric("Potential portfolio", f"{currency} {values['potential_portfolio']:,.0f}")

        table_col, pie_col = st.columns([1.4, 1], gap="large")
        with table_col:
            st.dataframe(
                enriched_options,
                width="stretch",
                hide_index=True,
                column_config={
                    "strike_price": st.column_config.NumberColumn(format=f"{currency} %.2f"),
                    "intrinsic_value": st.column_config.NumberColumn(format=f"{currency} %.0f"),
                    "moneyness_pct": st.column_config.NumberColumn("Moneyness", format="%.1f%%"),
                    "is_vested": st.column_config.CheckboxColumn("Vested"),
                },
            )
        with pie_col:
            st.plotly_chart(make_allocation_chart(values), width="stretch", theme="streamlit")

    enriched_options = enrich_options(st.session_state.option_positions, metrics.last_price, date.today())
    has_portfolio = int(st.session_state.shares) > 0 or bool((enriched_options["shares"] > 0).any())

    with scenarios_tab:
        if not has_portfolio:
            st.info("Add shares or option grants to unlock sensitivity analysis.")
        else:
            st.plotly_chart(
                make_sensitivity_chart(int(st.session_state.shares), enriched_options, metrics.last_price),
                width="stretch",
                theme="streamlit",
            )
            st.caption("Options are valued by current intrinsic value only: max(stock price - strike, 0) × granted quantity.")

    with projection_tab:
        if not has_portfolio:
            st.info("Add shares or option grants to run a Monte Carlo projection.")
        else:
            sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
            projection_years = sim_col1.slider("Projection years", 1, 10, 5)
            simulations = sim_col2.slider("Simulations", 1_000, 50_000, 10_000, step=1_000)
            use_historical = sim_col3.toggle("Use historical inputs", value=True)
            seed = sim_col4.number_input("Random seed", min_value=1, value=42, step=1)

            if use_historical:
                mu = metrics.annual_return
                sigma = max(metrics.annual_volatility, 0.0001)
                st.caption(f"Using {time_window} annualized return {mu:.2%} and volatility {sigma:.2%}.")
            else:
                input_col1, input_col2 = st.columns(2)
                mu = input_col1.number_input("Expected annual return (%)", value=10.0) / 100
                sigma = max(input_col2.number_input("Expected annual volatility (%)", value=15.0, min_value=0.0) / 100, 0.0001)

            active_options = enriched_options[enriched_options["shares"] > 0]
            results = run_monte_carlo(
                metrics.last_price,
                int(st.session_state.shares),
                tuple(active_options["shares"].astype(int)),
                tuple(active_options["strike_price"].astype(float)),
                float(mu),
                float(sigma),
                int(projection_years),
                int(simulations),
                int(seed),
            )
            summary_cols = st.columns(4)
            summary_cols[0].metric("Median final", f"{currency} {results['percentiles'][50][-1]:,.0f}")
            summary_cols[1].metric("5–95% range", f"{currency} {results['percentiles'][5][-1]:,.0f} – {results['percentiles'][95][-1]:,.0f}")
            summary_cols[2].metric("Mean final", f"{currency} {results['mean_final']:,.0f}")
            summary_cols[3].metric("P(gain)", f"{results['probability_gain']:.1%}")

            st.plotly_chart(make_projection_chart(results, int(projection_years)), width="stretch", theme="streamlit")
            with st.expander("Show simulated stock-price paths"):
                paths_to_show = st.slider("Paths to display", 10, min(250, int(simulations)), 80, step=10)
                st.plotly_chart(make_simulation_paths_chart(results, paths_to_show), width="stretch", theme="streamlit")

    with data_tab:
        st.markdown("#### Cached data inputs")
        st.write("Market data is cached for 15 minutes; profile metadata is cached for 60 minutes.")
        st.json({"ticker": ticker.upper(), "benchmark": benchmark.upper(), "period": period, "profile": profile}, expanded=False)
        st.download_button(
            "Download adjusted price history",
            data=company_data.to_csv().encode("utf-8"),
            file_name=f"{ticker.upper()}_{period}_adjusted_history.csv",
            mime="text/csv",
            icon=":material/table:",
        )

    st.caption(
        "Educational model only. Yahoo Finance data can be delayed, revised, or unavailable; verify against official plan documents and professional advice."
    )


if __name__ == "__main__":
    main()
