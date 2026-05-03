param(
    [switch]$SkipCompile = $false,
    [switch]$SkipRestart = $false,
    [string]$PublicBaseUrl = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$pythonExe = if ($env:APEX_PYTHON_EXE) {
    $env:APEX_PYTHON_EXE
} else {
    Join-Path $projectRoot ".venv\\Scripts\\python.exe"
}
$manageScript = Join-Path $PSScriptRoot "manage_apex.ps1"

Push-Location $projectRoot
try {
    git pull --ff-only
    if (-not $SkipCompile) {
        $env:PYTHONPATH = "."
        & $pythonExe -m compileall src tests scripts
        if ($LASTEXITCODE -ne 0) {
            throw "compileall failed"
        }
    }
    if (-not $SkipRestart) {
        & $manageScript -Action restart -Target all -PublicBaseUrl $PublicBaseUrl | Out-Null
    }
    & $manageScript -Action status -PublicBaseUrl $PublicBaseUrl
}
finally {
    Pop-Location
}
