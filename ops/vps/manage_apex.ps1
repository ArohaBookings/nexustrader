param(
    [ValidateSet("install", "start", "stop", "restart", "status", "logs", "reload-proxy")]
    [string]$Action = "status",
    [ValidateSet("all", "bot", "proxy", "mt5")]
    [string]$Target = "all",
    [string]$PublicBaseUrl = "",
    [string]$LogTarget = "bot"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$dataDir = Join-Path $projectRoot "data"
$configPath = Join-Path $projectRoot "config\\settings.yaml"
$botRunner = Join-Path $PSScriptRoot "bot_runner.ps1"
$proxyRunner = Join-Path $PSScriptRoot "proxy_runner.ps1"
$mt5Runner = Join-Path $PSScriptRoot "mt5_runner.ps1"
$caddyFile = Join-Path $PSScriptRoot "Caddyfile"

$botTask = "ApexBot"
$proxyTask = "ApexProxy"
$mt5Task = "ApexMT5"

if (-not $PublicBaseUrl) {
    if ($env:APEX_PUBLIC_BASE_URL) {
        $PublicBaseUrl = $env:APEX_PUBLIC_BASE_URL
    } else {
        $PublicBaseUrl = "http://31.44.5.163"
    }
}

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

function Get-DashboardPassword {
    if ($env:APEX_DASHBOARD_PASSWORD) { return $env:APEX_DASHBOARD_PASSWORD }
    if ($env:DASHBOARD_PASSWORD) { return $env:DASHBOARD_PASSWORD }
    return (Get-YamlScalar -Path $configPath -Section "dashboard" -Key "password")
}

function Resolve-CaddyExe {
    if ($env:APEX_CADDY_EXE -and (Test-Path $env:APEX_CADDY_EXE)) { return $env:APEX_CADDY_EXE }
    if (Test-Path "C:\\Tools\\caddy\\caddy.exe") { return "C:\\Tools\\caddy\\caddy.exe" }
    if (Test-Path "C:\\Program Files\\Caddy\\caddy.exe") { return "C:\\Program Files\\Caddy\\caddy.exe" }
    if (Get-Command caddy.exe -ErrorAction SilentlyContinue) { return "caddy.exe" }
    return ""
}

function Ensure-CaddyInstalled {
    $caddyExe = Resolve-CaddyExe
    if ($caddyExe) { return $caddyExe }
    if (Get-Command winget.exe -ErrorAction SilentlyContinue) {
        winget install --id CaddyServer.Caddy --accept-source-agreements --accept-package-agreements
        Start-Sleep -Seconds 2
        $caddyExe = Resolve-CaddyExe
    }
    if (-not $caddyExe) {
        throw "Caddy is not installed. Install it or set APEX_CADDY_EXE."
    }
    return $caddyExe
}

function Get-TaskStateValue {
    param([string]$TaskName)
    try {
        return (Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop).State.ToString()
    } catch {
        return "MISSING"
    }
}

function Stop-RunnerProcess {
    param([string]$ScriptPath)
    $escaped = [regex]::Escape($ScriptPath)
    $procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -match $escaped
    }
    foreach ($proc in $procs) {
        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
        } catch {
        }
    }
}

function Register-ApexTask {
    param(
        [string]$TaskName,
        [string]$ScriptPath,
        [bool]$AtStartup,
        [bool]$AtLogon,
        [string]$PrincipalType
    )
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`""
    $triggers = @()
    if ($AtStartup) { $triggers += New-ScheduledTaskTrigger -AtStartup }
    if ($AtLogon) { $triggers += New-ScheduledTaskTrigger -AtLogOn }
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -MultipleInstances IgnoreNew
    if ($PrincipalType -eq "SYSTEM") {
        $principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -RunLevel Highest -LogonType ServiceAccount
    } else {
        $userId = if ($env:USERDOMAIN) { "$($env:USERDOMAIN)\\$($env:USERNAME)" } else { $env:USERNAME }
        $principal = New-ScheduledTaskPrincipal -UserId $userId -RunLevel Highest -LogonType Interactive
    }
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers -Settings $settings -Principal $principal -Force | Out-Null
}

function Install-ApexTasks {
    Ensure-CaddyInstalled | Out-Null
    Register-ApexTask -TaskName $proxyTask -ScriptPath $proxyRunner -AtStartup $true -AtLogon $true -PrincipalType "SYSTEM"
    Register-ApexTask -TaskName $botTask -ScriptPath $botRunner -AtStartup $false -AtLogon $true -PrincipalType "INTERACTIVE"
    Register-ApexTask -TaskName $mt5Task -ScriptPath $mt5Runner -AtStartup $false -AtLogon $true -PrincipalType "INTERACTIVE"
}

function Start-ApexTarget {
    param([string]$Name)
    if (Get-TaskStateValue -TaskName $Name -ne "MISSING") {
        Start-ScheduledTask -TaskName $Name | Out-Null
    }
}

function Stop-ApexTarget {
    param([string]$Name, [string]$ScriptPath)
    try {
        Stop-ScheduledTask -TaskName $Name | Out-Null
    } catch {
    }
    Stop-RunnerProcess -ScriptPath $ScriptPath
}

function Invoke-DashboardCheck {
    param([string]$BaseUrl)
    $password = Get-DashboardPassword
    if (-not $password) {
        return @{ reachable = $false; auth = $false; reason = "dashboard_password_missing" }
    }
    try {
        $session = New-Object Microsoft.PowerShell.Commands.WebRequestSession
        Invoke-WebRequest -Uri "$BaseUrl/dashboard/login" -Method Post -Body @{ password = $password } -WebSession $session -MaximumRedirection 5 -ErrorAction Stop | Out-Null
        $data = Invoke-RestMethod -Uri "$BaseUrl/dashboard/data" -WebSession $session -ErrorAction Stop
        return @{
            reachable = $true
            auth = [bool]($data.summary)
            symbol_count = [int]($data.symbols.Count)
        }
    } catch {
        return @{ reachable = $false; auth = $false; reason = $_.Exception.Message }
    }
}

function Show-Status {
    $health = $null
    $stats = $null
    try { $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -ErrorAction Stop } catch {}
    try { $stats = Invoke-RestMethod -Uri "http://127.0.0.1:8000/stats" -ErrorAction Stop } catch {}

    $localDash = Invoke-DashboardCheck -BaseUrl "http://127.0.0.1:8000"
    $publicDash = Invoke-DashboardCheck -BaseUrl $PublicBaseUrl
    $proxyRunning = [bool](Get-Process -Name caddy -ErrorAction SilentlyContinue)
    $mt5Running = [bool](Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -in @("terminal64", "terminal") })

    $status = [ordered]@{
        bridge_running = [bool]($health -and $health.ok)
        reverse_proxy_running = $proxyRunning
        dashboard_auth_working = [bool]$localDash.auth
        dashboard_public_working = [bool]$publicDash.auth
        mt5_connected = [bool]($health -and $health.broker_connectivity -and $health.broker_connectivity.terminal_connected)
        daily_state = if ($health) { $health.current_daily_state } else { "" }
        daily_state_reason = if ($health) { $health.current_daily_state_reason } else { "" }
        open_positions = if ($stats) { $stats.open_positions } else { $null }
        queued_actions = if ($stats) { $stats.queued_actions_total } else { $null }
        delivered_actions_acked_total = if ($stats -and $stats.action_status_counts) { $stats.action_status_counts.ACKED } else { $null }
        watchdog_state = if ($health) { $health.watchdog_state } else { "" }
        last_ai_call = if ($health) { $health.last_successful_ai_call } else { "" }
        last_price_update_age_seconds = if ($health) { $health.last_price_update_age_seconds } else { $null }
        task_bot = Get-TaskStateValue -TaskName $botTask
        task_proxy = Get-TaskStateValue -TaskName $proxyTask
        task_mt5 = Get-TaskStateValue -TaskName $mt5Task
        public_dashboard_url = "$PublicBaseUrl/dashboard"
    }
    $status | ConvertTo-Json -Depth 6
}

switch ($Action) {
    "install" {
        Install-ApexTasks
        Show-Status
    }
    "start" {
        if ($Target -in @("all", "proxy")) { Start-ApexTarget -Name $proxyTask }
        if ($Target -in @("all", "mt5")) { Start-ApexTarget -Name $mt5Task }
        if ($Target -in @("all", "bot")) { Start-ApexTarget -Name $botTask }
        Start-Sleep -Seconds 2
        Show-Status
    }
    "stop" {
        if ($Target -in @("all", "bot")) { Stop-ApexTarget -Name $botTask -ScriptPath $botRunner }
        if ($Target -in @("all", "proxy")) { Stop-ApexTarget -Name $proxyTask -ScriptPath $proxyRunner }
        if ($Target -in @("all", "mt5")) { Stop-ApexTarget -Name $mt5Task -ScriptPath $mt5Runner }
        Show-Status
    }
    "restart" {
        & $PSCommandPath -Action stop -Target $Target -PublicBaseUrl $PublicBaseUrl | Out-Null
        Start-Sleep -Seconds 2
        & $PSCommandPath -Action start -Target $Target -PublicBaseUrl $PublicBaseUrl
    }
    "reload-proxy" {
        $caddyExe = Ensure-CaddyInstalled
        & $caddyExe reload --config $caddyFile
    }
    "logs" {
        $path = switch ($LogTarget) {
            "proxy" { Join-Path $dataDir "caddy-runner.log" }
            "mt5" { Join-Path $dataDir "mt5-supervisor.log" }
            default { Join-Path $dataDir "apex.log" }
        }
        if (-not (Test-Path $path)) {
            throw "Log file not found: $path"
        }
        Get-Content -Path $path -Tail 100 -Wait
    }
    "status" {
        Show-Status
    }
}
