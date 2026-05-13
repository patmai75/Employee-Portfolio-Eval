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
        xaxis_title="Stock price move (%)",
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
    for status, frame in grouped.groupby("status", sort=False):
        fig.add_trace(
            go.Bar(
                x=frame["vested_on"],
                y=frame["shares"],
                name=status,
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
        bargap=0.22,
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
            window_start = frame.index.min().date()
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
                "observations": int(frame["Close"].dropna().shape[0]),
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
                name=f"{scenario} median",
                line={"color": color, "width": 3},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=p[95],
                mode="lines",
                line={"color": color, "width": 1, "dash": "dot"},
                opacity=.42,
                name=f"{scenario} p95",
                showlegend=index < 2,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=p[5],
                mode="lines",
                line={"color": color, "width": 1, "dash": "dot"},
                opacity=.42,
                name=f"{scenario} p5",
                showlegend=index < 2,
            )
        )
    fig.update_layout(
        title="Portfolio projections across historical regimes",
        xaxis_title="Years from today",
        yaxis={"title": "Projected portfolio value", "tickprefix": "$", "separatethousands": True},
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


def projection_summary_table(results_by_scenario: dict[str, dict[str, Any]], currency: str) -> pd.DataFrame:
    rows = []
    for scenario, results in results_by_scenario.items():
        rows.append(
            {
                "Scenario": scenario,
                "P5 final": results["percentiles"][5][-1],
                "Median final": results["percentiles"][50][-1],
                "P95 final": results["percentiles"][95][-1],
                "Mean final": results["mean_final"],
                "P(gain)": results["probability_gain"],
                "P(double)": results["probability_double"],
            }
        )
    return pd.DataFrame(rows)

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
                st.plotly_chart(make_multi_projection_chart(results_by_scenario, int(projection_years)), width="stretch", theme="streamlit")
                summary = projection_summary_table(results_by_scenario, currency)
                best = summary.loc[summary["Median final"].idxmax()]
                worst = summary.loc[summary["Median final"].idxmin()]
                quick_cols = st.columns(4)
                quick_cols[0].metric("Best median scenario", str(best["Scenario"]), f"{currency} {best['Median final']:,.0f}")
                quick_cols[1].metric("Most conservative median", str(worst["Scenario"]), f"{currency} {worst['Median final']:,.0f}")
                quick_cols[2].metric("Highest P(gain)", str(summary.loc[summary["P(gain)"].idxmax(), "Scenario"]))
                quick_cols[3].metric("Scenarios compared", f"{len(results_by_scenario)}")
                st.dataframe(
                    summary,
                    width="stretch",
                    hide_index=True,
                    column_config={
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
