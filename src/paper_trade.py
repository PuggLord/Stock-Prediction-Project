from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils import PROJECT_ROOT, read_json, write_json

DEFAULT_LEDGER_PATH = PROJECT_ROOT / "signals" / "paper_ledger.json"


def _init_ledger(ticker: str, initial_cash: float) -> dict[str, Any]:
    return {
        "version": 1,
        "ticker": ticker,
        "cash": float(initial_cash),
        "shares": 0,
        "last_price": None,
        "equity": float(initial_cash),
        "history": [],
    }


def _load_ledger(path: Path, ticker: str, initial_cash: float) -> dict[str, Any]:
    if path.exists():
        ledger = read_json(path)
        if ledger.get("ticker") != ticker:
            raise ValueError(
                f"Ledger ticker mismatch: existing={ledger.get('ticker')} requested={ticker}"
            )
        return ledger
    return _init_ledger(ticker=ticker, initial_cash=initial_cash)


def apply_signal_to_ledger(
    ledger: dict[str, Any],
    signal_payload: dict[str, Any],
    max_history: int = 300,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply 0/1 signal to a paper ledger with target position sizing."""
    price = float(signal_payload["price"])
    signal = int(signal_payload["signal"])
    size_pct = float(signal_payload["size_pct"])

    cash = float(ledger["cash"])
    shares = int(ledger["shares"])

    equity_before = cash + (shares * price)
    target_shares = int((equity_before * size_pct) // price) if signal == 1 else 0

    max_affordable_shares = shares + int(cash // price)
    target_shares = min(target_shares, max_affordable_shares)

    delta_shares = target_shares - shares
    action = "HOLD"

    if delta_shares > 0:
        cost = delta_shares * price
        cash -= cost
        shares += delta_shares
        action = "BUY"
    elif delta_shares < 0:
        qty = -delta_shares
        proceeds = qty * price
        cash += proceeds
        shares -= qty
        action = "SELL"

    equity_after = cash + (shares * price)

    event = {
        "asof": signal_payload["asof"],
        "action": action,
        "shares_delta": int(delta_shares),
        "shares_total": int(shares),
        "price": float(price),
        "cash": round(float(cash), 4),
        "equity": round(float(equity_after), 4),
        "signal": int(signal),
        "p_up_3d": float(signal_payload["p_up_3d"]),
        "threshold": float(signal_payload["threshold"]),
    }

    history = list(ledger.get("history", []))
    history.append(event)
    if len(history) > max_history:
        history = history[-max_history:]

    ledger["cash"] = float(cash)
    ledger["shares"] = int(shares)
    ledger["last_price"] = float(price)
    ledger["equity"] = float(equity_after)
    ledger["history"] = history

    return ledger, event


def run_paper_trade_step(
    signal_payload: dict[str, Any],
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    initial_cash: float = 10_000.0,
) -> dict[str, Any]:
    """Load ledger, apply signal, persist ledger, and return trade event."""
    ticker = str(signal_payload["ticker"])
    ledger = _load_ledger(ledger_path, ticker=ticker, initial_cash=initial_cash)
    ledger, event = apply_signal_to_ledger(ledger, signal_payload)
    write_json(ledger, ledger_path)

    return {
        "ledger_path": str(ledger_path),
        "event": event,
    }
