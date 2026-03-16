from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.data import download_nvda_data
from src.features import build_feature_dataset
from src.paper_trade import run_paper_trade_step
from src.remote import DEFAULT_HOST, DEFAULT_PORT, post_webhook, serve, set_latest_signal
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

    # ------------------------------------------------------------------
    # Remote control spawn options
    # ------------------------------------------------------------------
    remote_group = parser.add_argument_group(
        "remote control",
        "Spawn an HTTP server for remote triggering or POST signals to a webhook.",
    )
    remote_group.add_argument(
        "--serve",
        action="store_true",
        help=(
            "Start an HTTP server that accepts remote trigger requests. "
            "Exposes: GET /health  GET /signal/latest  POST /signal"
        ),
    )
    remote_group.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        metavar="HOST",
        help=f"Bind address for --serve (default: {DEFAULT_HOST})",
    )
    remote_group.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        metavar="PORT",
        help=f"TCP port for --serve (default: {DEFAULT_PORT})",
    )
    remote_group.add_argument(
        "--webhook-url",
        type=str,
        default=None,
        metavar="URL",
        help="POST the generated signal payload as JSON to this URL.",
    )
    remote_group.add_argument(
        "--webhook-timeout",
        type=int,
        default=10,
        metavar="SECONDS",
        help="HTTP timeout in seconds for --webhook-url (default: 10)",
    )

    return parser.parse_args()


def _run_signal_once(
    ticker: str,
    threshold: float,
    size_pct: float,
    horizon_days: int,
    asof: str | None,
    refresh_data: bool,
    retrain_model: bool,
    output: Path,
    paper_trade: bool,
    paper_ledger: Path,
    paper_initial_cash: float,
    webhook_url: str | None,
    webhook_timeout: int,
) -> dict[str, Any]:
    """Generate one signal, optionally paper-trade and POST to webhook."""
    if ticker.upper() != "NVDA":
        raise ValueError(
            "This model is currently trained for NVDA. "
            "Use --ticker NVDA or retrain a per-symbol model first."
        )

    raw_data = download_nvda_data(
        symbol=ticker,
        force_download=refresh_data,
    )

    # Keep processed feature cache up to date for observability/debugging.
    feature_data = build_feature_dataset(raw_data=raw_data)

    if retrain_model or not LR_MODEL_PATH.exists():
        train_models(feature_data=feature_data, seed=SEED)

    signal_payload = generate_latest_signal(
        raw_data=raw_data,
        ticker=ticker,
        threshold=threshold,
        size_pct=size_pct,
        horizon_days=horizon_days,
        as_of=asof,
        force_download=False,
        model_path=LR_MODEL_PATH,
    )

    save_signal_payload(signal_payload, output)

    print(
        "Signal generated: "
        f"{signal_payload['ticker']} {signal_payload['asof']} "
        f"p_up_3d={signal_payload['p_up_3d']:.3f} "
        f"signal={signal_payload['signal']} "
        f"output={output}"
    )

    if paper_trade:
        trade_result = run_paper_trade_step(
            signal_payload=signal_payload,
            ledger_path=paper_ledger,
            initial_cash=paper_initial_cash,
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

    if webhook_url:
        print(f"Posting signal to webhook: {webhook_url}")
        try:
            resp = post_webhook(signal_payload, webhook_url, timeout=webhook_timeout)
            print(f"Webhook response: {resp}")
        except RuntimeError as exc:
            print(f"WARNING: Webhook delivery failed: {exc}")

    return signal_payload


def main() -> None:
    args = parse_args()

    if args.serve:
        # --serve: spawn an HTTP server that triggers signal generation on demand.
        def _trigger(options: dict[str, Any]) -> dict[str, Any]:
            payload = _run_signal_once(
                ticker=options.get("ticker", args.ticker),
                threshold=float(options.get("threshold", args.threshold)),
                size_pct=float(options.get("size_pct", args.size_pct)),
                horizon_days=int(options.get("horizon_days", args.horizon_days)),
                asof=options.get("asof", args.asof),
                refresh_data=bool(options.get("refresh_data", args.refresh_data)),
                retrain_model=bool(options.get("retrain_model", args.retrain_model)),
                output=args.output,
                paper_trade=bool(options.get("paper_trade", args.paper_trade)),
                paper_ledger=args.paper_ledger,
                paper_initial_cash=float(
                    options.get("paper_initial_cash", args.paper_initial_cash)
                ),
                webhook_url=options.get("webhook_url", args.webhook_url),
                webhook_timeout=int(options.get("webhook_timeout", args.webhook_timeout)),
            )
            set_latest_signal(payload)
            return payload

        serve(
            trigger_callback=_trigger,
            host=args.host,
            port=args.port,
            output_path=args.output,
        )
        return

    # Default: one-shot signal generation.
    _run_signal_once(
        ticker=args.ticker,
        threshold=args.threshold,
        size_pct=args.size_pct,
        horizon_days=args.horizon_days,
        asof=args.asof,
        refresh_data=args.refresh_data,
        retrain_model=args.retrain_model,
        output=args.output,
        paper_trade=args.paper_trade,
        paper_ledger=args.paper_ledger,
        paper_initial_cash=args.paper_initial_cash,
        webhook_url=args.webhook_url,
        webhook_timeout=args.webhook_timeout,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"run_signal failed: {exc}") from exc
