from __future__ import annotations

import io
import importlib.util
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
DEFAULT_TICKER = "PG"
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
        html, body, [class*="css"] {font-size: 16px;}
        .main .block-container {padding-top: 1.15rem; padding-bottom: 3rem; max-width: 1480px;}
        h1, h2, h3 {letter-spacing: -0.032em;}
        h2 {font-size: clamp(1.38rem, 1.8vw, 2rem) !important;}
        h3 {font-size: clamp(1.12rem, 1.35vw, 1.45rem) !important;}
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.055), rgba(37, 99, 235, 0.035));
            border: 1px solid rgba(120, 144, 180, 0.20);
            border-radius: 18px;
            padding: .78rem .82rem;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.045);
            min-height: 104px;
            overflow: visible;
        }
        div[data-testid="stMetricLabel"] p {
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .01em;
            color: rgba(51, 65, 85, .92);
            white-space: normal;
        }
        div[data-testid="stMetricValue"] {
            font-size: clamp(1.02rem, 1.22vw, 1.38rem);
            font-weight: 850;
            line-height: 1.08;
            white-space: normal;
            overflow-wrap: anywhere;
        }
        div[data-testid="stMetricDelta"] div {font-size: .78rem;}
        .portfolio-hero {
            padding: clamp(1.15rem, 3vw, 2.4rem);
            border-radius: 30px;
            background: radial-gradient(circle at 20% 20%, rgba(20,184,166,.55), transparent 24%), linear-gradient(135deg, #0f172a 0%, #1d4ed8 52%, #14b8a6 100%);
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 24px 55px rgba(29, 78, 216, 0.23);
        }
        .portfolio-hero h1 {margin: 0; font-size: clamp(1.9rem, 4.2vw, 4rem); line-height: .98;}
        .portfolio-hero p {margin: .75rem 0 0 0; opacity: .93; font-size: clamp(.98rem, 1.65vw, 1.22rem); max-width: 1040px;}
        .section-card {
            border: 1px solid rgba(148, 163, 184, .25);
            border-radius: 24px;
            padding: 1.1rem 1.25rem;
            background: rgba(248, 250, 252, .62);
        }
        .small-note {font-size: .95rem; opacity: .75;}
        @media (max-width: 760px) {
            html, body, [class*="css"] {font-size: 15px;}
            .main .block-container {padding-left: .85rem; padding-right: .85rem;}
            div[data-testid="column"] {width: 100% !important; flex: 1 1 100% !important;}
            div[data-testid="stMetric"] {padding: .75rem; border-radius: 16px; min-height: auto;}
        }
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

    download_kwargs: dict[str, Any] = {
        "tickers": list(tickers),
        "interval": "1d",
        "auto_adjust": True,
        "actions": True,
        "progress": False,
        "group_by": "ticker",
        "threads": True,
    }
    if period == "max":
        download_kwargs["start"] = "1900-01-01"
    else:
        download_kwargs["period"] = period

    data = yf.download(**download_kwargs)

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
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=amounts,
                hole=0.58,
                sort=False,
                marker={"colors": ["#2563eb", "#14b8a6", "#f59e0b"], "line": {"color": "#ffffff", "width": 2}},
                textfont={"color": "#0f172a", "size": 14},
                insidetextfont={"color": "#ffffff", "size": 14},
                outsidetextfont={"color": "#0f172a", "size": 13},
            )
        ]
    )
    fig.update_layout(
        title="Current portfolio composition",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#0f172a"},
        legend={"font": {"color": "#0f172a"}},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return fig


def make_sensitivity_chart(shares: int, options: pd.DataFrame, current_price: float) -> go.Figure:
    adjustments = np.array([-1.0, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1.0])
    prices = current_price * (1 + adjustments)
    values = []
    for price in prices:
        option_value = sum(calculate_option_value(float(price), row.strike_price, int(row.shares)) for row in options.itertuples())
        values.append(shares * float(price) + option_value)

    def compact_money(value: float) -> str:
        if abs(value) >= 1_000_000:
            return f"${value / 1_000_000:.1f}M"
        if abs(value) >= 1_000:
            return f"${value / 1_000:.0f}K"
        return f"${value:,.0f}"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[f"{change:+.0%}" for change in adjustments],
            y=values,
            text=[compact_money(value) for value in values],
            textposition="outside",
            textfont={"size": 13, "color": "#0f172a"},
            cliponaxis=False,
            marker_color=np.where(adjustments < 0, "#ef4444", "#2563eb"),
            name="Potential portfolio value",
        )
    )
    max_value = max(values) if values else 0
    fig.update_layout(
        title="Sensitivity of potential portfolio value",
        xaxis_title="Stock price move (%)",
        yaxis_title="Potential value",
        yaxis={"tickprefix": "$", "separatethousands": True, "range": [0, max_value * 1.18 if max_value else 1]},
        uniformtext={"mode": "show", "minsize": 10},
        margin={"l": 20, "r": 20, "t": 78, "b": 20},
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




def history_window(data: pd.DataFrame, years: int | None) -> pd.DataFrame:
    if data.empty or years is None:
        return data.copy()
    cutoff = data.index.max() - pd.DateOffset(years=years)
    return data[data.index >= cutoff].copy()


def portfolio_summary(shares: int, options: pd.DataFrame, price: float) -> dict[str, float]:
    option_shares = int(options["shares"].sum()) if not options.empty else 0
    vested_shares = int(options.loc[options["is_vested"], "shares"].sum()) if not options.empty else 0
    unvested_shares = option_shares - vested_shares
    itm_options = options[options["strike_price"] < price] if not options.empty else options
    itm_shares = int(itm_options["shares"].sum()) if not itm_options.empty else 0
    weighted_strike = float(np.average(options["strike_price"], weights=options["shares"])) if option_shares else 0.0
    weighted_vested_strike = (
        float(np.average(options.loc[options["is_vested"], "strike_price"], weights=options.loc[options["is_vested"], "shares"]))
        if vested_shares
        else 0.0
    )
    total_intrinsic = float(options["intrinsic_value"].sum()) if "intrinsic_value" in options else 0.0
    return {
        "common_shares": float(shares),
        "option_shares": float(option_shares),
        "vested_option_shares": float(vested_shares),
        "unvested_option_shares": float(unvested_shares),
        "in_the_money_option_shares": float(itm_shares),
        "equivalent_share_exposure": float(shares + option_shares),
        "weighted_avg_strike": weighted_strike,
        "weighted_avg_vested_strike": weighted_vested_strike,
        "avg_intrinsic_per_option": total_intrinsic / option_shares if option_shares else 0.0,
        "option_intrinsic_value": total_intrinsic,
        "options_itm_pct": itm_shares / option_shares if option_shares else 0.0,
    }


def make_waterfall_chart(values: dict[str, float], currency: str) -> go.Figure:
    unvested_option_value = max(0.0, values["total_option_value"] - values["vested_option_value"])
    fig = go.Figure(
        go.Waterfall(
            name="Portfolio build-up",
            orientation="v",
            measure=["relative", "relative", "total", "relative", "total"],
            x=["Stock value", "Vested options", "Current value", "Unvested options", "Potential value"],
            y=[
                values["stock_value"],
                values["vested_option_value"],
                values["vested_portfolio"],
                unvested_option_value,
                values["potential_portfolio"],
            ],
            text=[
                f"{currency} {values['stock_value']:,.0f}",
                f"{currency} {values['vested_option_value']:,.0f}",
                f"{currency} {values['vested_portfolio']:,.0f}",
                f"{currency} {unvested_option_value:,.0f}",
                f"{currency} {values['potential_portfolio']:,.0f}",
            ],
            textposition="outside",
            connector={"line": {"color": "rgba(100,116,139,.45)"}},
            increasing={"marker": {"color": "#2563eb"}},
            totals={"marker": {"color": "#14b8a6"}},
            hovertemplate="%{x}<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Current value vs total potential portfolio value",
        yaxis={"tickprefix": f"{currency} ", "separatethousands": True},
        margin={"l": 20, "r": 20, "t": 80, "b": 40},
    )
    return fig


def make_vesting_schedule_chart(options: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if options.empty or not (options["shares"] > 0).any():
        fig.update_layout(title="Option vesting schedule", annotations=[{"text": "Add option grants to see vesting", "showarrow": False}])
        return fig

    schedule = options[options["shares"] > 0].copy()
    schedule["status"] = np.where(schedule["is_vested"], "Vested", "Unvested")
    grouped = (
        schedule.groupby(["vested_on", "status"], as_index=False)
        .agg(
            shares=("shares", "sum"),
            intrinsic_value=("intrinsic_value", "sum"),
            avg_strike=("strike_price", "mean"),
            grants=("shares", "size"),
        )
        .sort_values("vested_on")
    )
    colors = {"Vested": "#14b8a6", "Unvested": "#f59e0b"}
    # Plotly uses milliseconds for bar widths on date axes. A fixed 18-day width
    # keeps isolated vesting dates from rendering as oversized columns.
    bar_width_ms = 18 * 24 * 60 * 60 * 1000
    for status, frame in grouped.groupby("status", sort=False):
        fig.add_trace(
            go.Bar(
                x=frame["vested_on"],
                y=frame["shares"],
                name=status,
                width=bar_width_ms,
                marker={"color": colors[status], "line": {"color": "rgba(15,23,42,.16)", "width": 1}},
                text=frame["shares"].map(lambda value: f"{value:,.0f}"),
                textposition="outside",
                customdata=np.stack([frame["avg_strike"], frame["intrinsic_value"], frame["grants"]], axis=1),
                hovertemplate=(
                    "Vests %{x|%b %d, %Y}<br>"
                    "Option shares %{y:,.0f}<br>"
                    "Avg strike $%{customdata[0]:,.2f}<br>"
                    "Intrinsic value $%{customdata[1]:,.0f}<br>"
                    "Grants %{customdata[2]:.0f}<extra></extra>"
                ),
            )
        )
    today = pd.Timestamp.today().normalize()
    fig.add_shape(
        type="line",
        x0=today,
        x1=today,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line={"color": "#ef4444", "width": 2, "dash": "dash"},
    )
    fig.add_annotation(x=today, y=1.04, xref="x", yref="paper", text="Today", showarrow=False, font={"color": "#ef4444"})
    fig.update_layout(
        title="Option vesting schedule by date",
        xaxis_title="Vesting date",
        yaxis_title="Option shares vesting",
        barmode="stack",
        bargap=0.62,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 20, "r": 20, "t": 80, "b": 30},
    )
    return fig


def make_cumulative_vesting_chart(options: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if options.empty or not (options["shares"] > 0).any():
        fig.update_layout(title="Cumulative vested option shares")
        return fig

    schedule = options[options["shares"] > 0].sort_values("vested_on").copy()
    schedule["cumulative_shares"] = schedule["shares"].cumsum()
    fig.add_trace(
        go.Scatter(
            x=schedule["vested_on"],
            y=schedule["cumulative_shares"],
            mode="lines+markers",
            fill="tozeroy",
            line={"color": "#1d4ed8", "width": 3},
            marker={"size": 9, "color": "#1d4ed8"},
            name="Cumulative vested shares",
            hovertemplate="%{x|%b %d, %Y}<br>Cumulative shares %{y:,.0f}<extra></extra>",
        )
    )
    today = pd.Timestamp.today().normalize()
    vested_today = int(schedule.loc[pd.to_datetime(schedule["vested_on"]) <= today, "shares"].sum())
    fig.add_shape(
        type="line",
        x0=today,
        x1=today,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line={"color": "#ef4444", "width": 2, "dash": "dash"},
    )
    fig.add_annotation(
        x=today,
        y=1.04,
        xref="x",
        yref="paper",
        text=f"Today · {vested_today:,.0f} vested",
        showarrow=False,
        font={"color": "#ef4444"},
    )
    fig.update_layout(
        title="Cumulative vesting curve",
        xaxis_title="Vesting date",
        yaxis_title="Cumulative option shares",
        margin={"l": 20, "r": 20, "t": 80, "b": 30},
    )
    return fig


def make_drawdown_chart(data: pd.DataFrame) -> go.Figure:
    close = data["Close"].dropna()
    drawdown = close / close.cummax() - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy", line={"color": "#ef4444"}, name="Drawdown"))
    fig.update_layout(
        title="Drawdown from previous highs",
        xaxis_title="Date",
        yaxis={"tickformat": ".0%"},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return fig


def make_return_distribution_chart(data: pd.DataFrame) -> go.Figure:
    returns = data["Close"].pct_change().dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns, nbinsx=70, marker_color="#2563eb", opacity=.84, name="Daily returns"))
    fig.update_layout(
        title="Distribution of daily returns",
        xaxis={"title": "Daily return", "tickformat": ".1%"},
        yaxis_title="Observations",
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return fig


def make_rolling_volatility_chart(data: pd.DataFrame) -> go.Figure:
    returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()
    rolling_vol = returns.rolling(63).std() * np.sqrt(TRADING_DAYS)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, mode="lines", line={"color": "#7c3aed"}, name="63D annualized vol"))
    fig.update_layout(
        title="Rolling 3-month annualized volatility",
        xaxis_title="Date",
        yaxis={"tickformat": ".0%"},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return fig


def scenario_metrics_from_history(full_data: pd.DataFrame, selected_data: pd.DataFrame, custom_mu: float, custom_sigma: float) -> pd.DataFrame:
    scenarios: list[dict[str, Any]] = []
    windows = {
        "Full history": full_data,
        "10Y history": history_window(full_data, 10),
        "5Y history": history_window(full_data, 5),
        "3Y history": history_window(full_data, 3),
        "Custom / selected": selected_data,
    }
    for name, frame in windows.items():
        if frame.empty or len(frame.dropna(subset=["Close"])) < 2:
            continue
        if name == "Custom / selected":
            mu, sigma = custom_mu, custom_sigma
            window_start = "Not applicable"
        else:
            metrics = calculate_metrics(frame)
            mu, sigma = metrics.annual_return, max(metrics.annual_volatility, 0.0001)
            window_start = frame.index.min().date()
        scenarios.append(
            {
                "scenario": name,
                "annual_return": float(mu),
                "annual_volatility": float(max(sigma, 0.0001)),
                "start_date": str(window_start),
                "observations": (
                    np.nan if name == "Custom / selected" else int(frame["Close"].dropna().shape[0])
                ),
            }
        )
    return pd.DataFrame(scenarios)


def make_scenario_scatter(scenarios: pd.DataFrame) -> go.Figure:
    """Institutional return/volatility comparison without bubble sizing."""
    fig = go.Figure()
    if scenarios.empty:
        return fig

    fig.add_trace(
        go.Bar(
            x=scenarios["scenario"],
            y=scenarios["annual_return"],
            name="Annual return",
            marker_color="#1d4ed8",
            hovertemplate="%{x}<br>Return %{y:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=scenarios["scenario"],
            y=scenarios["annual_volatility"],
            name="Annual volatility",
            marker_color="#14b8a6",
            hovertemplate="%{x}<br>Volatility %{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Scenario assumptions: annual return vs volatility",
        xaxis_title="",
        yaxis={"title": "Annualized rate", "tickformat": ".0%"},
        barmode="group",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 20, "r": 20, "t": 75, "b": 20},
    )
    return fig


def make_multi_projection_chart(results_by_scenario: dict[str, dict[str, Any]], years: int) -> go.Figure:
    fig = go.Figure()
    palette = ["#1d4ed8", "#14b8a6", "#f97316", "#7c3aed", "#ef4444"]
    x = np.arange(0, years + 1)
    for index, (scenario, results) in enumerate(results_by_scenario.items()):
        color = palette[index % len(palette)]
        p = results["percentiles"]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=p[50],
                mode="lines+markers",
                name=scenario,
                line={"color": color, "width": 3},
                marker={"size": 8},
                hovertemplate="%{fullData.name}<br>Year %{x}<br>Median value $%{y:,.0f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Median portfolio projections across scenarios",
        xaxis_title="Years from today",
        yaxis={"title": "Projected median portfolio value", "tickprefix": "$", "separatethousands": True},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.20,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255,255,255,0)",
        },
        margin={"l": 20, "r": 20, "t": 85, "b": 110},
    )
    return fig


def projection_summary_table(
    results_by_scenario: dict[str, dict[str, Any]],
    currency: str,
    scenario_assumptions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    assumptions = {}
    if scenario_assumptions is not None and not scenario_assumptions.empty:
        assumptions = scenario_assumptions.set_index("scenario").to_dict("index")

    rows = []
    for scenario, results in results_by_scenario.items():
        assumption = assumptions.get(scenario, {})
        rows.append(
            {
                "Scenario": scenario,
                "Ann. return": assumption.get("annual_return", np.nan),
                "Ann. vol": assumption.get("annual_volatility", np.nan),
                "Data starts": assumption.get("start_date", "—"),
                "Observations": assumption.get("observations", np.nan),
                "P5 final": results["percentiles"][5][-1],
                "Median final": results["percentiles"][50][-1],
                "P95 final": results["percentiles"][95][-1],
                "Mean final": results["mean_final"],
                "P(gain)": results["probability_gain"],
                "P(double)": results["probability_double"],
            }
        )
    return pd.DataFrame(rows)



def pdf_dependencies_available() -> bool:
    return importlib.util.find_spec("reportlab") is not None


def figure_to_png_bytes(fig: go.Figure, width: int = 720, height: int = 420) -> bytes | None:
    """Render a Plotly figure to PNG bytes when Kaleido is installed and usable."""
    if importlib.util.find_spec("kaleido") is None:
        return None
    try:
        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception:
        return None


def build_one_pager_pdf(
    *,
    ticker: str,
    company_name: str,
    currency: str,
    time_window: str,
    benchmark: str,
    metrics: MarketMetrics,
    benchmark_metrics: MarketMetrics | None,
    values: dict[str, float],
    summary: dict[str, float],
    options: pd.DataFrame,
    waterfall_fig: go.Figure,
    allocation_fig: go.Figure,
    vesting_fig: go.Figure,
    cumulative_vesting_fig: go.Figure,
    sensitivity_fig: go.Figure,
    comparison_fig: go.Figure | None,
    projection_fig: go.Figure | None,
    projection_summary: pd.DataFrame | None,
    custom_mu: float,
    custom_sigma: float,
) -> bytes:
    """Build a compact institutional one-page PDF report."""
    if not pdf_dependencies_available():
        raise RuntimeError("PDF export requires reportlab. Install dependencies from requirements.txt.")

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A3, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    generated_label = f"Generated {pd.Timestamp.today().strftime('%b %d, %Y')}"

    def draw_footer(canvas: Canvas, doc: SimpleDocTemplate) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.14 * inch, generated_label)
        canvas.restoreState()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A3),
        rightMargin=0.28 * inch,
        leftMargin=0.28 * inch,
        topMargin=0.22 * inch,
        bottomMargin=0.30 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=1,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=10,
        textColor=colors.HexColor("#475569"),
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=11,
        textColor=colors.HexColor("#1d4ed8"),
        spaceBefore=3,
        spaceAfter=3,
    )
    small_style = ParagraphStyle("Small", parent=styles["Normal"], fontSize=7.0, leading=8.0)

    def money(value: float) -> str:
        return f"{currency} {value:,.0f}"

    def pct(value: float) -> str:
        return f"{value:.1%}"

    def styled_table(data: list[list[Any]], *, header: bool = True, font_size: float = 6.8) -> Table:
        table = Table(data, repeatRows=1 if header else 0, hAlign="CENTER")
        commands = [
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]
        if header:
            commands.extend(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0f2fe")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ]
            )
        table.setStyle(TableStyle(commands))
        return table

    report_width = doc.width
    chart_col_width = report_width / 3
    chart_inner_width = chart_col_width - 0.24 * inch
    chart_image_height = 2.36 * inch

    def pdf_chart_content(
        fig: go.Figure | None,
        *,
        width: float,
        height: float,
        render_width: int = 900,
        render_height: int = 430,
    ) -> Image | Paragraph:
        pdf_fig = None
        if fig is not None:
            pdf_fig = go.Figure(fig)
            pdf_fig.update_layout(
                font={"size": 12, "color": "#0f172a"},
                title={"font": {"size": 15, "color": "#0f172a"}},
                legend={"font": {"size": 11, "color": "#0f172a"}},
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                margin={"l": 48, "r": 24, "t": 52, "b": 54},
            )
            pdf_fig.update_xaxes(tickfont={"size": 10, "color": "#0f172a"}, title_font={"size": 11}, automargin=True)
            pdf_fig.update_yaxes(tickfont={"size": 10, "color": "#0f172a"}, title_font={"size": 11}, automargin=True)
        image_bytes = figure_to_png_bytes(pdf_fig, width=render_width, height=render_height) if pdf_fig is not None else None
        if image_bytes:
            return Image(io.BytesIO(image_bytes), width=width, height=height)
        return Paragraph("Chart renderer unavailable. Install kaleido to embed this chart.", small_style)

    story: list[Any] = []
    story.append(Paragraph(f"{company_name} ({ticker.upper()}) · Employee Portfolio One-Pager", title_style))
    story.append(Paragraph("Institutional snapshot of market context, portfolio value, option grants, vesting, sensitivity and projections.", subtitle_style))
    story.append(Spacer(1, 4))

    benchmark_return = pct(benchmark_metrics.annual_return) if benchmark_metrics is not None else "—"
    benchmark_volatility = pct(benchmark_metrics.annual_volatility) if benchmark_metrics is not None else "—"
    benchmark_window_return = pct(benchmark_metrics.cumulative_return) if benchmark_metrics is not None else "—"
    context_data = [
        ["Section", "Metric", "Value", "Section", "Metric", "Value"],
        ["Market", "Ticker", ticker.upper(), "Benchmark", "Ticker", benchmark.upper()],
        ["Market", "Window", time_window, "Benchmark", "Window return", benchmark_window_return],
        ["Market", "Last price", f"{currency} {metrics.last_price:,.2f}", "Benchmark", "Ann. return", benchmark_return],
        ["Risk", "Ann. return", pct(metrics.annual_return), "Benchmark", "Ann. vol", benchmark_volatility],
        ["Risk", "Ann. volatility", pct(metrics.annual_volatility), "Portfolio", "Current value", money(values["vested_portfolio"])],
        ["Portfolio", "Stock value", money(values["stock_value"]), "Portfolio", "Potential value", money(values["potential_portfolio"])],
    ]
    context = styled_table(context_data, font_size=6.8)
    context._argW = [0.62 * inch, 0.90 * inch, 1.08 * inch] * 2
    context_block = Table(
        [[Paragraph("Market & portfolio context", section_style)], [context]],
        colWidths=[report_width * 0.56],
        hAlign="CENTER",
    )
    comparison_content = pdf_chart_content(
        comparison_fig,
        width=report_width * 0.39,
        height=1.58 * inch,
        render_width=820,
        render_height=300,
    )
    comparison_block = Table(
        [[Paragraph("Benchmark comparison", section_style)], [comparison_content]],
        colWidths=[report_width * 0.40],
        hAlign="CENTER",
    )
    top_grid = Table(
        [[context_block, comparison_block]],
        colWidths=[report_width * 0.58, report_width * 0.42],
        hAlign="CENTER",
    )
    top_grid.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ]
        )
    )
    story.append(top_grid)
    story.append(Spacer(1, 4))

    executive_data = [
        ["Metric", "Value", "Metric", "Value", "Metric", "Value", "Metric", "Value"],
        ["Vested options", money(values["vested_option_value"]), "Total options", money(values["total_option_value"]), "Weighted avg strike", f"{currency} {summary['weighted_avg_strike']:,.2f}", "Vested avg strike", f"{currency} {summary['weighted_avg_vested_strike']:,.2f}"],
        ["ITM option shares", f"{summary['in_the_money_option_shares']:,.0f}", "ITM %", pct(summary["options_itm_pct"]), "Unvested options", f"{summary['unvested_option_shares']:,.0f}", "Equivalent exposure", f"{summary['equivalent_share_exposure']:,.0f}"],
        ["Avg intrinsic / option", f"{currency} {summary['avg_intrinsic_per_option']:,.2f}", "Option intrinsic", money(summary["option_intrinsic_value"]), "Custom return", pct(custom_mu), "Custom vol", pct(custom_sigma)],
    ]
    executive = styled_table(executive_data, font_size=6.8)
    executive._argW = [1.0 * inch, 1.0 * inch] * 4
    story.append(Paragraph("Executive position summary", section_style))
    story.append(executive)
    story.append(Spacer(1, 4))

    def chart_block(title: str, fig: go.Figure | None) -> Table:
        content = pdf_chart_content(fig, width=chart_inner_width, height=chart_image_height)
        block = Table(
            [[Paragraph(title, section_style)], [content]],
            colWidths=[chart_inner_width],
        )
        block.setStyle(
            TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        return block

    chart_grid = Table(
        [
            [chart_block("Portfolio value bridge", waterfall_fig), chart_block("Portfolio allocation", allocation_fig), chart_block("Sensitivity analysis", sensitivity_fig)],
            [chart_block("Option vesting schedule", vesting_fig), chart_block("Cumulative vesting curve", cumulative_vesting_fig), chart_block("Median projection scenarios", projection_fig)],
        ],
        colWidths=[chart_col_width, chart_col_width, chart_col_width],
        rowHeights=[2.78 * inch, 2.78 * inch],
        hAlign="CENTER",
    )
    chart_grid.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.append(chart_grid)
    story.append(Spacer(1, 3))

    option_rows = [["Grant date", "Vests on", "Shares", "Strike", "Status", "Intrinsic", "Moneyness"]]
    option_view = options[options["shares"] > 0].sort_values("vested_on").head(8)
    for row in option_view.itertuples():
        option_rows.append(
            [
                row.grant_date.isoformat(),
                row.vested_on.isoformat(),
                f"{int(row.shares):,}",
                f"{currency} {float(row.strike_price):,.2f}",
                "Vested" if bool(row.is_vested) else "Unvested",
                money(float(row.intrinsic_value)),
                f"{float(row.moneyness_pct):,.1f}%" if pd.notna(row.moneyness_pct) else "—",
            ]
        )
    if len(option_rows) == 1:
        option_rows.append(["—", "—", "0", "—", "—", "—", "—"])
    option_table = styled_table(option_rows, font_size=6.3)

    projection_rows = [["Scenario", "Ann. return", "Ann. vol", "Data starts", "P5", "Median", "P95", "Mean", "P(gain)", "P(double)"]]
    if projection_summary is not None and not projection_summary.empty:
        for _, row in projection_summary.head(6).iterrows():
            projection_rows.append(
                [
                    str(row["Scenario"]),
                    pct(float(row["Ann. return"])) if pd.notna(row["Ann. return"]) else "—",
                    pct(float(row["Ann. vol"])) if pd.notna(row["Ann. vol"]) else "—",
                    str(row["Data starts"]),
                    money(float(row["P5 final"])),
                    money(float(row["Median final"])),
                    money(float(row["P95 final"])),
                    money(float(row["Mean final"])),
                    f"{float(row['P(gain)']):.1%}",
                    f"{float(row['P(double)']):.1%}",
                ]
            )
    else:
        projection_rows.append(["Run projections", "—", "—", "—", "—", "—", "—", "—", "—", "—"])
    projection_table = styled_table(projection_rows, font_size=5.8)

    option_block = Table(
        [[Paragraph("Option grant detail", section_style)], [option_table]],
        colWidths=[report_width * 0.47],
        hAlign="CENTER",
    )
    projection_block = Table(
        [[Paragraph("Projection scenario summary", section_style)], [projection_table]],
        colWidths=[report_width * 0.50],
        hAlign="CENTER",
    )
    for block in (option_block, projection_block):
        block.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                ]
            )
        )
    tables = Table(
        [[option_block, projection_block]],
        colWidths=[report_width * 0.48, report_width * 0.52],
        hAlign="CENTER",
    )
    tables.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(tables)

    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return buffer.getvalue()

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
            <p>Convierte tu compensación accionaria en una vista institucional: valor actual, opciones, vesting, riesgo y escenarios futuros en un solo lugar.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls = st.container(border=True)
    with controls:
        left, middle, right, actions = st.columns([1.1, 1.1, 1.05, 1.55], vertical_alignment="bottom")
        with left:
            st.session_state.ticker = st.text_input("Company ticker", st.session_state.ticker, help="Default: PG (Procter & Gamble)").upper().strip()
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
        full_history = fetch_history((ticker,), "max")
        company_data = get_ticker_frame(history, ticker)
        full_company_data = get_ticker_frame(full_history, ticker)
        benchmark_data = get_ticker_frame(history, benchmark)
        profile = fetch_profile(ticker)

    if full_company_data.empty:
        full_company_data = company_data

    if company_data.empty:
        st.error(f"No market data found for `{ticker}` in the selected window. Check the ticker and try again.")
        st.stop()

    currency = str(profile.get("currency") or profile.get("fast_currency") or "USD")
    company_name = profile.get("longName") or profile.get("shortName") or ticker.upper()
    metrics = calculate_metrics(company_data)
    benchmark_metrics = calculate_metrics(benchmark_data) if not benchmark_data.empty else None
    price_delta = metrics.last_price - metrics.previous_price
    price_delta_pct = price_delta / metrics.previous_price if metrics.previous_price else 0.0

    st.subheader(f"{company_name} ({ticker.upper()})")
    meta = " · ".join(str(value) for value in [profile.get("exchange"), profile.get("sector"), profile.get("industry")] if value)
    if meta:
        st.caption(meta)

    kpi_top = st.columns(3)
    kpi_top[0].metric("Last price", f"{currency} {metrics.last_price:,.2f}", f"{price_delta_pct:+.2%}")
    kpi_top[1].metric("Window return", f"{metrics.cumulative_return:+.2%}")
    kpi_top[2].metric("Annual return", f"{metrics.annual_return:+.2%}")
    kpi_bottom = st.columns(3)
    kpi_bottom[0].metric("Annual volatility", f"{metrics.annual_volatility:.2%}")
    kpi_bottom[1].metric("Max drawdown", f"{metrics.max_drawdown:.2%}")
    kpi_bottom[2].metric("Sharpe ratio", f"{metrics.sharpe_ratio:.2f}")

    overview_tab, portfolio_tab, vesting_tab, scenarios_tab, projection_tab, data_tab = st.tabs(
        ["📈 Market", "💼 Portfolio Summary", "🗓️ Vesting", "🎚️ Sensitivity", "🔮 Projections", "🧾 Data"]
    )

    with overview_tab:
        st.markdown("### Market analytics")
        st.caption("Price action, benchmark comparison and risk views for the selected analysis window.")
        chart_col, benchmark_col = st.columns([1.25, 1], gap="large")
        with chart_col:
            st.plotly_chart(make_price_chart(company_data, ticker, currency), width="stretch", theme="streamlit")
        with benchmark_col:
            if benchmark_data.empty:
                st.warning(f"No comparison data found for `{benchmark}`.")
            else:
                st.plotly_chart(make_comparison_chart(company_data, benchmark_data, ticker, benchmark), width="stretch", theme="streamlit")
                if benchmark_metrics is not None:
                    bench_cols = st.columns(3)
                    bench_cols[0].metric(f"{benchmark.upper()} window return", f"{benchmark_metrics.cumulative_return:+.2%}")
                    bench_cols[1].metric(f"{benchmark.upper()} annual return", f"{benchmark_metrics.annual_return:+.2%}")
                    bench_cols[2].metric(f"{benchmark.upper()} annual volatility", f"{benchmark_metrics.annual_volatility:.2%}")

        risk_col1, risk_col2, risk_col3 = st.columns(3, gap="large")
        with risk_col1:
            st.plotly_chart(make_drawdown_chart(company_data), width="stretch", theme="streamlit")
        with risk_col2:
            st.plotly_chart(make_return_distribution_chart(company_data), width="stretch", theme="streamlit")
        with risk_col3:
            st.plotly_chart(make_rolling_volatility_chart(company_data), width="stretch", theme="streamlit")


    with portfolio_tab:
        st.session_state.shares = st.number_input(
            "Number of shares", min_value=0, value=int(st.session_state.shares), step=1
        )
        st.markdown("#### Option grants (editable)")
        st.caption("Edit grants directly in the table: add rows, change quantities, strikes, grant dates, and vesting terms.")
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

        summary = portfolio_summary(int(st.session_state.shares), enriched_options, metrics.last_price)

        st.markdown("### Executive position summary")
        value_top = st.columns(3)
        value_top[0].metric("Stock value", f"{currency} {values['stock_value']:,.0f}", f"{int(st.session_state.shares):,} shares")
        value_top[1].metric("Vested options", f"{currency} {values['vested_option_value']:,.0f}", f"{summary['vested_option_shares']:,.0f} option shares")
        value_top[2].metric("Total options", f"{currency} {values['total_option_value']:,.0f}", f"{summary['option_shares']:,.0f} option shares")
        value_bottom = st.columns(2)
        value_bottom[0].metric("Vested portfolio", f"{currency} {values['vested_portfolio']:,.0f}")
        value_bottom[1].metric("Potential portfolio", f"{currency} {values['potential_portfolio']:,.0f}")

        detail_top = st.columns(4)
        detail_top[0].metric("Weighted avg strike", f"{currency} {summary['weighted_avg_strike']:,.2f}")
        detail_top[1].metric("Vested avg strike", f"{currency} {summary['weighted_avg_vested_strike']:,.2f}")
        detail_top[2].metric("ITM option shares", f"{summary['in_the_money_option_shares']:,.0f}", f"{summary['options_itm_pct']:.0%} of options")
        detail_top[3].metric("Avg intrinsic / option", f"{currency} {summary['avg_intrinsic_per_option']:,.2f}")
        detail_bottom = st.columns(3)
        detail_bottom[0].metric("Equivalent exposure", f"{summary['equivalent_share_exposure']:,.0f}", "shares + options")
        detail_bottom[1].metric("Unvested options", f"{summary['unvested_option_shares']:,.0f}")
        detail_bottom[2].metric("Option intrinsic value", f"{currency} {summary['option_intrinsic_value']:,.0f}")

        chart_a, chart_b = st.columns([1.05, 1], gap="large")
        with chart_a:
            st.plotly_chart(make_waterfall_chart(values, currency), width="stretch", theme="streamlit")
        with chart_b:
            st.plotly_chart(make_allocation_chart(values), width="stretch", theme="streamlit")

        st.markdown("#### Option grant detail")
        st.dataframe(
            enriched_options,
            width="stretch",
            hide_index=True,
            column_config={
                "shares": st.column_config.NumberColumn("Option shares", format="%d"),
                "strike_price": st.column_config.NumberColumn("Strike", format=f"{currency} %.2f"),
                "intrinsic_value": st.column_config.NumberColumn("Intrinsic value", format=f"{currency} %.0f"),
                "moneyness_pct": st.column_config.NumberColumn("Moneyness", format="%.1f%%"),
                "is_vested": st.column_config.CheckboxColumn("Vested"),
                "vested_on": st.column_config.DateColumn("Fully vested on"),
            },
        )

    enriched_options = enrich_options(st.session_state.option_positions, metrics.last_price, date.today())
    has_portfolio = int(st.session_state.shares) > 0 or bool((enriched_options["shares"] > 0).any())
    projection_summary_for_pdf: pd.DataFrame | None = None
    projection_fig_for_pdf: go.Figure | None = None
    custom_mu_for_pdf = 0.10
    custom_sigma_for_pdf = 0.15

    with vesting_tab:
        st.markdown("### Vesting schedule")
        st.caption("Visualize when option grants become fully vested and how total vested option exposure accumulates over time.")
        vest_a, vest_b = st.columns(2, gap="large")
        with vest_a:
            st.plotly_chart(make_vesting_schedule_chart(enriched_options), width="stretch", theme="streamlit")
        with vest_b:
            st.plotly_chart(make_cumulative_vesting_chart(enriched_options), width="stretch", theme="streamlit")

        if (enriched_options["shares"] > 0).any():
            schedule_view = enriched_options[enriched_options["shares"] > 0].sort_values("vested_on").copy()
            schedule_view["days_to_vest"] = (pd.to_datetime(schedule_view["vested_on"]) - pd.Timestamp.today().normalize()).dt.days
            st.dataframe(
                schedule_view[["grant_date", "vested_on", "days_to_vest", "shares", "strike_price", "is_vested", "intrinsic_value"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "strike_price": st.column_config.NumberColumn("Strike", format=f"{currency} %.2f"),
                    "intrinsic_value": st.column_config.NumberColumn("Intrinsic", format=f"{currency} %.0f"),
                    "is_vested": st.column_config.CheckboxColumn("Vested"),
                },
            )
        else:
            st.info("Add at least one option grant to see the vesting schedule.")

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
            st.info("Add shares or option grants to run Monte Carlo projections.")
        else:
            st.markdown("### Multi-scenario Monte Carlo projection")
            st.caption(
                "Compare several historical regimes at once. Randomness is handled internally with a fixed reproducible seed, so you do not need to manage it."
            )
            sim_col1, sim_col2 = st.columns(2)
            projection_years = sim_col1.slider("Projection horizon", 1, 10, 5, help="Years projected from today.")
            simulations = sim_col2.slider("Simulation paths per scenario", 1_000, 50_000, 12_000, step=1_000)

            custom_col1, custom_col2 = st.columns(2)
            custom_mu = custom_col1.number_input(
                "Custom expected annual return (%)",
                value=10.0,
                help="Institutional base-case default. Adjust if you want a custom forward-looking return.",
            ) / 100
            custom_sigma = max(
                custom_col2.number_input(
                    "Custom expected annual volatility (%)",
                    value=15.0,
                    min_value=0.0,
                    help="Institutional base-case default. Adjust if you want a custom forward-looking volatility.",
                ) / 100,
                0.0001,
            )
            custom_mu_for_pdf = float(custom_mu)
            custom_sigma_for_pdf = float(custom_sigma)

            scenarios = scenario_metrics_from_history(full_company_data, company_data, custom_mu, custom_sigma)
            all_scenarios = scenarios["scenario"].tolist()
            selected_scenarios = st.multiselect(
                "Projection scenarios",
                options=all_scenarios,
                default=all_scenarios,
                help="All available regimes are selected by default, including 3Y and Custom / selected.",
            )
            scenario_subset = scenarios[scenarios["scenario"].isin(selected_scenarios)]
            scenario_display = scenario_subset.assign(
                annual_return_pct=scenario_subset["annual_return"] * 100,
                annual_volatility_pct=scenario_subset["annual_volatility"] * 100,
            )[["scenario", "annual_return_pct", "annual_volatility_pct", "start_date"]]

            scen_col1, scen_col2 = st.columns([1.1, 1], gap="large")
            with scen_col1:
                st.dataframe(
                    scenario_display,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "scenario": st.column_config.TextColumn("Scenario"),
                        "annual_return_pct": st.column_config.NumberColumn("Ann. return", format="%.2f%%"),
                        "annual_volatility_pct": st.column_config.NumberColumn("Ann. vol", format="%.2f%%"),
                        "start_date": st.column_config.TextColumn("Data starts"),
                    },
                )
            with scen_col2:
                st.plotly_chart(make_scenario_scatter(scenario_subset), width="stretch", theme="streamlit")

            active_options = enriched_options[enriched_options["shares"] > 0]
            results_by_scenario: dict[str, dict[str, Any]] = {}
            for index, row in enumerate(scenario_subset.itertuples(index=False)):
                results_by_scenario[row.scenario] = run_monte_carlo(
                    metrics.last_price,
                    int(st.session_state.shares),
                    tuple(active_options["shares"].astype(int)),
                    tuple(active_options["strike_price"].astype(float)),
                    float(row.annual_return),
                    float(row.annual_volatility),
                    int(projection_years),
                    int(simulations),
                    20_260_513 + index,
                )

            if not results_by_scenario:
                st.warning("Select at least one scenario to project.")
            else:
                projection_fig_for_pdf = make_multi_projection_chart(results_by_scenario, int(projection_years))
                st.plotly_chart(projection_fig_for_pdf, width="stretch", theme="streamlit")
                summary = projection_summary_table(results_by_scenario, currency, scenario_subset)
                projection_summary_for_pdf = summary
                best = summary.loc[summary["Median final"].idxmax()]
                worst = summary.loc[summary["Median final"].idxmin()]
                quick_cols = st.columns(4)
                quick_cols[0].metric("Best median scenario", str(best["Scenario"]), f"{currency} {best['Median final']:,.0f}")
                quick_cols[1].metric("Most conservative median", str(worst["Scenario"]), f"{currency} {worst['Median final']:,.0f}")
                quick_cols[2].metric("Highest P(gain)", str(summary.loc[summary["P(gain)"].idxmax(), "Scenario"]))
                quick_cols[3].metric("Scenarios compared", f"{len(results_by_scenario)}")
                summary_display = summary.copy()
                summary_display["Ann. return"] = summary_display["Ann. return"] * 100
                summary_display["Ann. vol"] = summary_display["Ann. vol"] * 100
                summary_display["P(gain)"] = summary_display["P(gain)"] * 100
                summary_display["P(double)"] = summary_display["P(double)"] * 100
                st.dataframe(
                    summary_display,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Ann. return": st.column_config.NumberColumn("Ann. return", format="%.2f%%"),
                        "Ann. vol": st.column_config.NumberColumn("Ann. vol", format="%.2f%%"),
                        "Observations": st.column_config.NumberColumn("Observations", format="%d"),
                        "P5 final": st.column_config.NumberColumn("P5 final", format=f"{currency} %.0f"),
                        "Median final": st.column_config.NumberColumn("Median final", format=f"{currency} %.0f"),
                        "P95 final": st.column_config.NumberColumn("P95 final", format=f"{currency} %.0f"),
                        "Mean final": st.column_config.NumberColumn("Mean final", format=f"{currency} %.0f"),
                        "P(gain)": st.column_config.NumberColumn("P(gain)", format="%.1f%%"),
                        "P(double)": st.column_config.NumberColumn("P(double)", format="%.1f%%"),
                    },
                )

                st.markdown("#### Custom scenario simulated stock-price paths")
                st.caption(
                    f"These paths use the Custom inputs above: {custom_mu:.1%} expected annual return and "
                    f"{custom_sigma:.1%} annual volatility. They are shown separately from the selected historical regimes."
                )
                custom_path_results = run_monte_carlo(
                    metrics.last_price,
                    int(st.session_state.shares),
                    tuple(active_options["shares"].astype(int)),
                    tuple(active_options["strike_price"].astype(float)),
                    float(custom_mu),
                    float(custom_sigma),
                    int(projection_years),
                    int(simulations),
                    20_260_599,
                )
                paths_to_show = st.slider("Paths to display", 10, min(250, int(simulations)), 80, step=10)
                st.plotly_chart(make_simulation_paths_chart(custom_path_results, paths_to_show), width="stretch", theme="streamlit")

    with data_tab:
        st.markdown("#### Exports")
        st.write("Market data is cached for 15 minutes; profile metadata is cached for 60 minutes.")
        st.json({"ticker": ticker.upper(), "benchmark": benchmark.upper(), "period": period, "profile": profile}, expanded=False)

        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                "Download adjusted price history",
                data=company_data.to_csv().encode("utf-8"),
                file_name=f"{ticker.upper()}_{period}_adjusted_history.csv",
                mime="text/csv",
                icon=":material/table:",
            )
        with export_col2:
            if not has_portfolio:
                st.info("Add shares or option grants to enable the one-page PDF report.")
            elif not pdf_dependencies_available():
                st.info("PDF export is ready in code. Install `reportlab` and `kaleido` from requirements.txt to enable the one-page report download.")
            else:
                report_values = portfolio_values(int(st.session_state.shares), enriched_options, metrics.last_price)
                report_summary = portfolio_summary(int(st.session_state.shares), enriched_options, metrics.last_price)
                try:
                    pdf_bytes = build_one_pager_pdf(
                        ticker=ticker,
                        company_name=company_name,
                        currency=currency,
                        time_window=time_window,
                        benchmark=benchmark,
                        metrics=metrics,
                        benchmark_metrics=benchmark_metrics,
                        values=report_values,
                        summary=report_summary,
                        options=enriched_options,
                        waterfall_fig=make_waterfall_chart(report_values, currency),
                        allocation_fig=make_allocation_chart(report_values),
                        vesting_fig=make_vesting_schedule_chart(enriched_options),
                        cumulative_vesting_fig=make_cumulative_vesting_chart(enriched_options),
                        sensitivity_fig=make_sensitivity_chart(int(st.session_state.shares), enriched_options, metrics.last_price),
                        comparison_fig=(
                            make_comparison_chart(company_data, benchmark_data, ticker, benchmark)
                            if not benchmark_data.empty
                            else None
                        ),
                        projection_fig=projection_fig_for_pdf,
                        projection_summary=projection_summary_for_pdf,
                        custom_mu=custom_mu_for_pdf,
                        custom_sigma=custom_sigma_for_pdf,
                    )
                    st.download_button(
                        "Download one-page PDF report",
                        data=pdf_bytes,
                        file_name=f"{ticker.upper()}_employee_portfolio_one_pager.pdf",
                        mime="application/pdf",
                        icon=":material/picture_as_pdf:",
                        type="primary",
                    )
                except Exception as exc:
                    st.warning(f"PDF report could not be generated: {exc}")

    st.caption(
        "Educational model only. Yahoo Finance data can be delayed, revised, or unavailable; verify against official plan documents and professional advice."
    )


if __name__ == "__main__":
    main()
