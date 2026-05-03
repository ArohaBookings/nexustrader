param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$dataDir = Join-Path $projectRoot "data"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
$logPath = Join-Path $dataDir "caddy-runner.log"

$caddyExe = if ($env:APEX_CADDY_EXE) {
    $env:APEX_CADDY_EXE
} elseif (Test-Path "C:\\Tools\\caddy\\caddy.exe") {
    "C:\\Tools\\caddy\\caddy.exe"
} elseif (Test-Path "C:\\Program Files\\Caddy\\caddy.exe") {
    "C:\\Program Files\\Caddy\\caddy.exe"
} else {
    "caddy.exe"
}

$caddyFile = if ($env:APEX_CADDYFILE) {
    $env:APEX_CADDYFILE
} else {
    Join-Path $PSScriptRoot "Caddyfile"
}

$restartSleep = 10
if ($env:APEX_PROXY_RESTART_SLEEP_SECONDS) {
    $restartSleep = [Math]::Max(1, [int]$env:APEX_PROXY_RESTART_SLEEP_SECONDS)
}

while ($true) {
    "[{0}] [APEX] starting caddy reverse proxy" -f (Get-Date -Format o) | Out-File -FilePath $logPath -Append -Encoding utf8
    & $caddyExe run --config $caddyFile 2>&1 | Tee-Object -FilePath $logPath -Append
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) {
        "[{0}] [APEX] caddy exited cleanly" -f (Get-Date -Format o) | Out-File -FilePath $logPath -Append -Encoding utf8
        break
    }
    "[{0}] [APEX] caddy exited with code {1}; restarting in {2}s" -f (Get-Date -Format o), $exitCode, $restartSleep | Out-File -FilePath $logPath -Append -Encoding utf8
    Start-Sleep -Seconds $restartSleep
}
