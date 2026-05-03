## Windows VPS Startup

Start order:
1. Launch MT5 and log into the trading account.
2. Wait for charts/EAs to load.
3. Start the bot with:

```powershell
cd "C:\path\to\apex_bot"
python scripts\start_bridge_prod.py
```

Recommended Task Scheduler action:
- Program: `python`
- Arguments: `scripts\start_bridge_prod.py`
- Start in: `C:\path\to\apex_bot`

Recommended environment variables:
- `APEX_RUNTIME_ROOT`
- `APEX_CONFIG_DIR`
- `APEX_DATA_DIR`
- `APEX_CACHE_DIR`
- `APEX_LOGS_DIR`
- `APEX_STATE_DIR`
- `APEX_MODELS_DIR`
- `APEX_TEMP_DIR`
- `OPENAI_API_KEY`
- `BRIDGE_AUTH_TOKEN`

Readiness command:

```powershell
cd "C:\path\to\apex_bot"
python scripts\preflight_prod.py
```

Live health checks:

```powershell
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/stats
curl "http://127.0.0.1:8000/debug/symbol?symbol=BTCUSD&account=Main&magic=7777&timeframe=M15"
curl "http://127.0.0.1:8000/debug/xau_grid?account=Main&magic=7777&timeframe=M15"
```
