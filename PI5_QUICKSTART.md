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

## 4) Remote control (spawn options)

### 4a) HTTP server mode (`--serve`)

Spawn the signal runner as a persistent HTTP server so a remote machine can
trigger signal generation on demand without SSH access:

```bash
python run_signal.py --serve --host 0.0.0.0 --port 8080
```

Available routes:

| Method | Route             | Description                              |
|--------|-------------------|------------------------------------------|
| GET    | `/health`         | Liveness probe – returns `{"status":"ok"}` |
| GET    | `/signal/latest`  | Return the last generated signal JSON    |
| POST   | `/signal`         | Trigger signal generation; optional JSON body overrides any default flag (e.g. `{"threshold":0.6,"paper_trade":true}`) |

**Trigger from another machine:**

```bash
# One-shot trigger with default settings
curl -s -X POST http://<pi5-ip>:8080/signal | python -m json.tool

# Override threshold and enable paper trading for this run
curl -s -X POST http://<pi5-ip>:8080/signal \
  -H 'Content-Type: application/json' \
  -d '{"threshold":0.6,"paper_trade":true}' | python -m json.tool

# Fetch the most recently generated signal without triggering a new one
curl -s http://<pi5-ip>:8080/signal/latest | python -m json.tool
```

Combine with other flags for the initial spawn:

```bash
# Serve with paper trading enabled for every remote trigger
python run_signal.py --serve --port 8080 --paper-trade

# Serve and also POST each signal to a webhook on every remote trigger
python run_signal.py --serve --port 8080 --webhook-url http://dashboard.local/ingest
```

### 4b) Webhook delivery (`--webhook-url`)

After generating a signal (one-shot or via server), POST the payload to a
remote URL automatically:

```bash
python run_signal.py --webhook-url http://dashboard.local/ingest
```

The full signal JSON is sent as `application/json`. Use `--webhook-timeout` to
adjust the HTTP timeout (default 10 s):

```bash
python run_signal.py \
  --webhook-url http://dashboard.local/ingest \
  --webhook-timeout 30
```

## 5) Build downloadable Pi5 bundle

```bash
python bundle_pi5.py
```

Zip output:

- `downloads/pi5_package.zip`
