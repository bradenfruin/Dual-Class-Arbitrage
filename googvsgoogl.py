from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class Params:
    lookback: int = 60              # length of backtest
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5
    max_hold_days: int = 30
    initial_capital: float = 100_000.0       # how much bread you got
    gross_leverage: float = 1.0   


def compute_zscore(premium: pd.Series, lookback: int) -> pd.DataFrame:
    mu = premium.rolling(lookback).mean()
    sd = premium.rolling(lookback).std(ddof=0).replace(0, np.nan)
    z = (premium - mu) / sd
    return pd.DataFrame({"premium": premium, "mu": mu, "sd": sd, "z": z})


def backtest_dual_class_raw_premium_same_close(
    prices_a: pd.Series,
    prices_b: pd.Series,
    params: Params,
) -> pd.DataFrame:

    df = pd.DataFrame({"A": prices_a, "B": prices_b}).dropna()
    if df.empty:
        raise ValueError("No overlapping price data after alignment/dropna.")

    df["premium"] = df["A"] / df["B"] - 1.0
    zdf = compute_zscore(df["premium"], params.lookback)
    df = df.join(zdf[["mu", "sd", "z"]])
    df["retA_fwd"] = df["A"].shift(-1) / df["A"] - 1.0
    df["retB_fwd"] = df["B"].shift(-1) / df["B"] - 1.0
    pos = 0
    hold_days = 0
    equity = params.initial_capital
    wA = 0.0
    wB = 0.0
    trades = []
    curve = []
    def weights_for(p: int) -> tuple[float, float]:
        if p == 0:
            return 0.0, 0.0
        if p == +1:
            return +0.5 * params.gross_leverage, -0.5 * params.gross_leverage
        if p == -1:
            return -0.5 * params.gross_leverage, +0.5 * params.gross_leverage
        raise ValueError("pos must be -1, 0, +1")
    for t in range(len(df)):
        date = df.index[t]
        z = df["z"].iloc[t]

        curve.append((date, equity, pos, float(z) if np.isfinite(z) else np.nan))

        if not np.isfinite(z) or t == len(df) - 1:
            continue

        z = float(z)
        new_pos = pos

        # exits / stops at the close
        if pos != 0:
            if abs(z) >= params.stop_z:
                new_pos = 0
                trades.append((date, "EXIT", pos, z, equity, f"STOP |z|>={params.stop_z}"))
            elif hold_days >= params.max_hold_days:
                new_pos = 0
                trades.append((date, "EXIT", pos, z, equity, f"TIME {hold_days}d>={params.max_hold_days}"))
            elif abs(z) <= params.exit_z:
                new_pos = 0
                trades.append((date, "EXIT", pos, z, equity, f"EXIT |z|<={params.exit_z}"))

        # Entries
        if pos == 0:
            hold_days = 0
            if z >= params.entry_z:
                new_pos = -1
                trades.append((date, "ENTER", new_pos, z, equity, f"ENTER z>={params.entry_z}"))
            elif z <= -params.entry_z:
                new_pos = +1
                trades.append((date, "ENTER", new_pos, z, equity, f"ENTER z<={-params.entry_z}"))

        # Weigths
        if new_pos != pos:
            wA, wB = weights_for(new_pos)
            pos = new_pos

        # Next day P/L
        rA = float(df["retA_fwd"].iloc[t])
        rB = float(df["retB_fwd"].iloc[t])
        pnl = equity * (wA * rA + wB * rB)
        equity += pnl

        if pos != 0:
            hold_days += 1
        else:
            hold_days = 0

    out = pd.DataFrame(curve, columns=["date", "equity", "pos", "z"]).set_index("date")
    out["returns"] = out["equity"].pct_change().fillna(0.0)

    trades_df = pd.DataFrame(trades, columns=["date", "type", "pos", "z", "equity", "reason"])
    trades_df = trades_df.set_index("date") if not trades_df.empty else trades_df
    out.attrs["trades"] = trades_df
    return out


# THE MAIN CODE
if __name__ == "__main__":
    

    tickers = ["GOOGL", "GOOG"]
    px = yf.download(tickers, start="2015-01-01", auto_adjust=False, progress=False)["Close"]
    prices_a = px["GOOGL"]
    prices_b = px["GOOG"]

    params = Params(
        lookback=60,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.5,
        max_hold_days=30,
        initial_capital=100_000,
        gross_leverage=1.0,
    )

    result = backtest_dual_class_raw_premium_same_close(prices_a, prices_b, params)
    trades = result.attrs["trades"]

    total_return = result["equity"].iloc[-1] / params.initial_capital - 1
    ann_return = (1 + total_return) ** (252 / max(1, len(result) - 1)) - 1
    ann_vol = result["returns"].std(ddof=0) * math.sqrt(252)
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else np.nan
    max_dd = (result["equity"] / result["equity"].cummax() - 1).min()

    print("Final equity:", round(result["equity"].iloc[-1], 2))
    print("Total return:", round(total_return * 100, 2), "%")
    print("Ann return:", round(ann_return * 100, 2), "%")
    print("Ann vol:", round(ann_vol * 100, 2), "%")
    print("Sharpe (naive):", round(float(sharpe), 2))
    print("Max drawdown:", round(max_dd * 100, 2), "%")


    import matplotlib.pyplot as plt

    result["equity"].plot(title="Equity Curve")
    plt.show()
