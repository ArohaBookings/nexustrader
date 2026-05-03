param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$configPath = Join-Path $projectRoot "config\\settings.yaml"
$dataDir = Join-Path $projectRoot "data"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
$logPath = Join-Path $dataDir "mt5-supervisor.log"

function Get-YamlScalar {
    param(
        [string]$Path,
        [string]$Section,
        [string]$Key
    )
    if (-not (Test-Path $Path)) {
        return ""
    }
    $lines = Get-Content -Path $Path
    $inSection = $false
    foreach ($line in $lines) {
        if ($line -match "^[A-Za-z0-9_]+:\\s*$") {
            $inSection = ($line.Trim().TrimEnd(":") -eq $Section)
            continue
        }
        if ($inSection -and $line -match ("^\\s{2}" + [regex]::Escape($Key) + ":\\s*(.*)$")) {
            return $Matches[1].Trim().Trim("'").Trim('"')
        }
    }
    return ""
}

$mt5Exe = if ($env:APEX_MT5_TERMINAL_PATH) {
    $env:APEX_MT5_TERMINAL_PATH
} else {
    Get-YamlScalar -Path $configPath -Section "system" -Key "mt5_terminal_path"
}

$checkSeconds = 30
if ($env:APEX_MT5_WATCHDOG_SECONDS) {
    $checkSeconds = [Math]::Max(5, [int]$env:APEX_MT5_WATCHDOG_SECONDS)
}

while ($true) {
    if (-not $mt5Exe -or -not (Test-Path $mt5Exe)) {
        "[{0}] [APEX] MT5 path missing: {1}" -f (Get-Date -Format o), $mt5Exe | Out-File -FilePath $logPath -Append -Encoding utf8
        Start-Sleep -Seconds $checkSeconds
        continue
    }

    $running = Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessName -in @("terminal64", "terminal")
    }
    if (-not $running) {
        "[{0}] [APEX] starting MT5 terminal: {1}" -f (Get-Date -Format o), $mt5Exe | Out-File -FilePath $logPath -Append -Encoding utf8
        Start-Process -FilePath $mt5Exe | Out-Null
        Start-Sleep -Seconds 10
    }
    Start-Sleep -Seconds $checkSeconds
}
