$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimeRoot = Join-Path $Root '.cloudheal'

foreach ($name in @('frontend', 'backend')) {
    $pidFile = Join-Path $RuntimeRoot "$name.pid"
    if (-not (Test-Path $pidFile)) { continue }

    $id = 0
    [int]::TryParse((Get-Content $pidFile -Raw).Trim(), [ref]$id) | Out-Null
    if ($id -gt 0) {
        $process = Get-Process -Id $id -ErrorAction SilentlyContinue
        if ($null -ne $process) {
            Write-Host "Stopping CloudHeal $name (PID $id)..."
            Stop-Process -Id $id -Force
        }
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

Write-Host 'CloudHeal processes started by the launcher have been stopped.'
