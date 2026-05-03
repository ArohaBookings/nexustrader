# VPS Operations

## Scope
- Python bridge stays on `127.0.0.1:8000`
- only these public paths are intended to be proxied:
  - `/dashboard`
  - `/dashboard/login`
  - `/dashboard/logout`
  - `/dashboard/data`
  - optional `/health`
- do **not** expose:
  - `/v1/*`
  - `/stats`
  - `/debug/*`

## Reverse proxy
Config file:
- [ops/vps/Caddyfile](/Users/leobons/Library/Mobile%20Documents/com~apple~CloudDocs/Kimi_Agent_Set%20%26%20Forget%20Trade%20Bot/apex_bot/ops/vps/Caddyfile)

This Caddy config only proxies `/dashboard*` and `/health` to `127.0.0.1:8000` and returns `404` for everything else.

## Windows VPS startup model
Scripts:
- [ops/vps/bot_runner.ps1](/Users/leobons/Library/Mobile%20Documents/com~apple~CloudDocs/Kimi_Agent_Set%20%26%20Forget%20Trade%20Bot/apex_bot/ops/vps/bot_runner.ps1)
- [ops/vps/proxy_runner.ps1](/Users/leobons/Library/Mobile%20Documents/com~apple~CloudDocs/Kimi_Agent_Set%20%26%20Forget%20Trade%20Bot/apex_bot/ops/vps/proxy_runner.ps1)
- [ops/vps/mt5_runner.ps1](/Users/leobons/Library/Mobile%20Documents/com~apple~CloudDocs/Kimi_Agent_Set%20%26%20Forget%20Trade%20Bot/apex_bot/ops/vps/mt5_runner.ps1)
- [ops/vps/manage_apex.ps1](/Users/leobons/Library/Mobile%20Documents/com~apple~CloudDocs/Kimi_Agent_Set%20%26%20Forget%20Trade%20Bot/apex_bot/ops/vps/manage_apex.ps1)
- [ops/vps/deploy_vps.ps1](/Users/leobons/Library/Mobile%20Documents/com~apple~CloudDocs/Kimi_Agent_Set%20%26%20Forget%20Trade%20Bot/apex_bot/ops/vps/deploy_vps.ps1)

Tasks installed by `manage_apex.ps1 -Action install`:
- `ApexProxy`
- `ApexBot`
- `ApexMT5`

Notes:
- `ApexProxy` is registered for startup/logon.
- `ApexBot` is registered for logon and runs the existing bridge supervisor script.
- `ApexMT5` is registered for logon and keeps MT5 alive if `system.mt5_terminal_path` or `APEX_MT5_TERMINAL_PATH` is set.
- For true reboot recovery on a Windows MT5 VPS, configure Windows auto-logon for the VPS trading user. MT5 is a GUI app and is most reliable in an interactive session.

## Required environment variables
- `APEX_DASHBOARD_PASSWORD`
- `APEX_DASHBOARD_SESSION_SECRET`
- optional:
  - `APEX_PUBLIC_BASE_URL`
  - `APEX_CADDY_EXE`
  - `APEX_MT5_TERMINAL_PATH`
  - `APEX_PYTHON_EXE`

## Commands
Install tasks:
```powershell
powershell -ExecutionPolicy Bypass -File ops\vps\manage_apex.ps1 -Action install
```

Start:
```powershell
powershell -ExecutionPolicy Bypass -File ops\vps\manage_apex.ps1 -Action start -Target all
```

Stop:
```powershell
powershell -ExecutionPolicy Bypass -File ops\vps\manage_apex.ps1 -Action stop -Target all
```

Restart:
```powershell
powershell -ExecutionPolicy Bypass -File ops\vps\manage_apex.ps1 -Action restart -Target all
```

Status:
```powershell
powershell -ExecutionPolicy Bypass -File ops\vps\manage_apex.ps1 -Action status
```

Tail bot logs:
```powershell
powershell -ExecutionPolicy Bypass -File ops\vps\manage_apex.ps1 -Action logs -LogTarget bot
```

Tail proxy logs:
```powershell
powershell -ExecutionPolicy Bypass -File ops\vps\manage_apex.ps1 -Action logs -LogTarget proxy
```

Deploy update:
```powershell
powershell -ExecutionPolicy Bypass -File ops\vps\deploy_vps.ps1
```

## Public URLs
Once the reverse proxy is running on the VPS IP:
- `http://31.44.5.163/dashboard`
- `http://31.44.5.163/health`

Local-only URLs on the VPS:
- `http://127.0.0.1:8000/dashboard`
- `http://127.0.0.1:8000/health`

## Security
- The dashboard app still does its own password auth.
- The reverse proxy does not expose trade endpoints.
- HTTPS is not added in-app. Use TLS at the proxy/load-balancer layer if you want encrypted public access.
