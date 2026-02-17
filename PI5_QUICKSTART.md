# Pi5 Quickstart (OpenClaw Bot)

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Generate compact signal JSON

```bash
python run_signal.py --output signals/latest_signal.json
```

Output example:

```json
{"ticker":"NVDA","asof":"2026-02-17","p_up_3d":0.61,"signal":1,"size_pct":0.1}
```

## 3) Optional paper-trade step

```bash
python run_signal.py --output signals/latest_signal.json --paper-trade
```

This updates:

- `signals/paper_ledger.json`
- `signals/paper_trade_last.json`

## 4) Build downloadable Pi5 bundle

```bash
python bundle_pi5.py
```

Zip output:

- `downloads/pi5_package.zip`
