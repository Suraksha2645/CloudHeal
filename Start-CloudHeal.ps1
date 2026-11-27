$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$FrontendRoot = Join-Path $Root 'frontend'
$LogRoot = Join-Path $Root 'logs'
$RuntimeRoot = Join-Path $Root '.cloudheal'
$BackendUrl = 'http://localhost:8000'
$FrontendUrl = 'http://localhost:5173'
$Distro = 'Ubuntu-22.04'

New-Item -ItemType Directory -Force -Path $LogRoot, $RuntimeRoot | Out-Null

function Test-HttpEndpoint {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [string]$Contains
    )

    try {
        $body = (& curl.exe -fsS --max-time 3 $Url 2>$null | Out-String)
        if ($LASTEXITCODE -ne 0) { return $false }
        return [string]::IsNullOrEmpty($Contains) -or $body.Contains($Contains)
    } catch {
        return $false
    }
}

function Wait-ForEndpoint {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$TimeoutSeconds = 60,
        [string]$Contains
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        if (Test-HttpEndpoint -Url $Url -Contains $Contains) { return $true }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)
    return $false
}

function Save-ProcessId {
    param([string]$Name, [int]$Id)
    Set-Content -Path (Join-Path $RuntimeRoot "$Name.pid") -Value $Id -Encoding ascii
}

function Start-CloudHealProcess {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    $stdout = Join-Path $LogRoot "$Name.log"
    $stderr = Join-Path $LogRoot "$Name.error.log"
    $process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -WindowStyle Normal -PassThru
    Save-ProcessId -Name $Name -Id $process.Id
    return $process
}

Write-Host '================================='
Write-Host '       STARTING CLOUDHEAL'
Write-Host '================================='

Write-Host 'Checking WSL Ubuntu-22.04...'
& wsl.exe -d $Distro -- bash -lc 'true' 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "WSL distribution '$Distro' is unavailable. Run 'wsl --install -d $Distro' or check its name with 'wsl -l -q'."
}

if (Test-HttpEndpoint -Url "$BackendUrl/api/health" -Contains '"status":"healthy"') {
    Write-Host 'Backend is already healthy.'
} else {
    Write-Host 'Starting FastAPI backend...'
    $backendCommand = "cd /mnt/c/CloudHeal && source .venv-linux/bin/activate && exec python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
    Start-CloudHealProcess -Name 'backend' -FilePath 'wsl.exe' `
        -ArgumentList @('-d', $Distro, '--', 'bash', '-lc', $backendCommand) `
        -WorkingDirectory $Root | Out-Null

    if (-not (Wait-ForEndpoint -Url "$BackendUrl/api/health" -TimeoutSeconds 90 -Contains '"status":"healthy"')) {
        throw "Backend did not become healthy. Review $LogRoot\backend.log and $LogRoot\backend.error.log."
    }
    Write-Host 'Backend is healthy.'
}

$telemetry = (& curl.exe -fsS --max-time 5 "$BackendUrl/api/telemetry/status" 2>$null | Out-String)
if ($LASTEXITCODE -ne 0 -or -not $telemetry.Contains('"running":true')) {
    throw "Backend is reachable but telemetry is not running. Review the backend logs in $LogRoot\."
}
Write-Host 'Telemetry collector is running.'

if (Test-HttpEndpoint -Url "$FrontendUrl/" -Contains '<div id="root"></div>') {
    Write-Host 'Frontend is already ready.'
} else {
    Write-Host 'Starting Vite dashboard on port 5173...'
    Start-CloudHealProcess -Name 'frontend' -FilePath 'npm.cmd' `
        -ArgumentList @('run', 'dev', '--', '--host', '127.0.0.1', '--port', '5173', '--strictPort') `
        -WorkingDirectory $FrontendRoot | Out-Null

    if (-not (Wait-ForEndpoint -Url "$FrontendUrl/" -TimeoutSeconds 60 -Contains '<div id="root"></div>')) {
        throw "Frontend did not become ready. Review $LogRoot\frontend.log and $LogRoot\frontend.error.log."
    }
    Write-Host 'Frontend is ready.'
}

if (-not (Test-HttpEndpoint -Url "$FrontendUrl/api/health" -Contains '"status":"healthy"')) {
    throw 'Frontend could not reach the backend through the Vite /api proxy.'
}

Write-Host ''
Write-Host 'CloudHeal is ready:'
Write-Host "  Dashboard: $FrontendUrl/"
Write-Host "  Backend:   $BackendUrl/api/health"
Write-Host "  Logs:      $LogRoot\"
Start-Process $FrontendUrl
