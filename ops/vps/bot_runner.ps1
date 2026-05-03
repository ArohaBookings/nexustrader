param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$dataDir = Join-Path $projectRoot "data"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
$logPath = Join-Path $dataDir "apex-supervisor.log"

$pythonExe = if ($env:APEX_PYTHON_EXE) {
    $env:APEX_PYTHON_EXE
} else {
    Join-Path $projectRoot ".venv\\Scripts\\python.exe"
}

if (-not $env:APEX_RUNTIME_HEARTBEAT_FILE) {
    $env:APEX_RUNTIME_HEARTBEAT_FILE = Join-Path $dataDir "apex-runtime-heartbeat.json"
}

$launcher = Join-Path $projectRoot "scripts\\start_bridge_prod.py"
$restartSleep = 10
if ($env:APEX_BRIDGE_RESTART_SLEEP_SECONDS) {
    $restartSleep = [Math]::Max(1, [int]$env:APEX_BRIDGE_RESTART_SLEEP_SECONDS)
}

Push-Location $projectRoot
try {
    while ($true) {
        "[{0}] [APEX] starting bridge supervisor" -f (Get-Date -Format o) | Out-File -FilePath $logPath -Append -Encoding utf8
        $env:PYTHONPATH = "."
        & $pythonExe $launcher 2>&1 | Tee-Object -FilePath $logPath -Append
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0) {
            "[{0}] [APEX] bridge supervisor exited cleanly" -f (Get-Date -Format o) | Out-File -FilePath $logPath -Append -Encoding utf8
            break
        }
        "[{0}] [APEX] bridge supervisor exited with code {1}; restarting in {2}s" -f (Get-Date -Format o), $exitCode, $restartSleep | Out-File -FilePath $logPath -Append -Encoding utf8
        Start-Sleep -Seconds $restartSleep
    }
}
finally {
    Pop-Location
}
