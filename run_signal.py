from __future__ import annotations

import argparse
from pathlib import Path

from src.data import download_nvda_data
from src.features import build_feature_dataset
from src.paper_trade import run_paper_trade_step
from src.signal import generate_latest_signal, save_signal_payload
from src.train import LR_MODEL_PATH, train_models
from src.utils import PROJECT_ROOT, SEED, write_json

SIGNALS_DIR = PROJECT_ROOT / "signals"
DEFAULT_SIGNAL_PATH = SIGNALS_DIR / "latest_signal.json"
DEFAULT_LEDGER_PATH = SIGNALS_DIR / "paper_ledger.json"
DEFAULT_PAPER_EVENT_PATH = SIGNALS_DIR / "paper_trade_last.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a compact JSON signal for OpenClaw/local bots and "
            "optionally execute one paper-trade step."
        )
    )
    parser.add_argument("--ticker", type=str, default="NVDA")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--size-pct", type=float, default=0.10)
    parser.add_argument("--horizon-days", type=int, default=3)
    parser.add_argument("--asof", type=str, default=None, help="Optional YYYY-MM-DD override")

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SIGNAL_PATH,
        help="Path for compact signal JSON",
    )

    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force fresh Yahoo download before generating signal",
    )
    parser.add_argument(
        "--retrain-model",
        action="store_true",
        help="Retrain model before generating signal",
    )

    parser.add_argument(
        "--paper-trade",
        action="store_true",
        help="Apply the signal to a local paper-trade ledger",
    )
    parser.add_argument(
        "--paper-ledger",
        type=Path,
        default=DEFAULT_LEDGER_PATH,
        help="Ledger JSON path used for paper trading",
    )
    parser.add_argument(
        "--paper-initial-cash",
        type=float,
        default=10_000.0,
        help="Initial cash if paper ledger does not exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.ticker.upper() != "NVDA":
        raise SystemExit(
            "This model is currently trained for NVDA. "
            "Use --ticker NVDA or retrain a per-symbol model first."
        )

    raw_data = download_nvda_data(
        symbol=args.ticker,
        force_download=args.refresh_data,
    )

    # Keep processed feature cache up to date for observability/debugging.
    feature_data = build_feature_dataset(raw_data=raw_data)

    if args.retrain_model or not LR_MODEL_PATH.exists():
        train_models(feature_data=feature_data, seed=SEED)

    signal_payload = generate_latest_signal(
        raw_data=raw_data,
        ticker=args.ticker,
        threshold=args.threshold,
        size_pct=args.size_pct,
        horizon_days=args.horizon_days,
        as_of=args.asof,
        force_download=False,
        model_path=LR_MODEL_PATH,
    )

    save_signal_payload(signal_payload, args.output)

    print(
        "Signal generated: "
        f"{signal_payload['ticker']} {signal_payload['asof']} "
        f"p_up_3d={signal_payload['p_up_3d']:.3f} "
        f"signal={signal_payload['signal']} "
        f"output={args.output}"
    )

    if args.paper_trade:
        trade_result = run_paper_trade_step(
            signal_payload=signal_payload,
            ledger_path=args.paper_ledger,
            initial_cash=args.paper_initial_cash,
        )
        write_json(trade_result["event"], DEFAULT_PAPER_EVENT_PATH)
        print(
            "Paper trade: "
            f"action={trade_result['event']['action']} "
            f"shares_delta={trade_result['event']['shares_delta']} "
            f"equity={trade_result['event']['equity']:.2f}"
        )
        print(f"Ledger: {trade_result['ledger_path']}")
        print(f"Last trade event: {DEFAULT_PAPER_EVENT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"run_signal failed: {exc}") from exc
