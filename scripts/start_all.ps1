# =============================================================================
# APEX Stack - Start All Services
# Starts Nexus Assistant and RAGNAROK v7 APEX servers
# =============================================================================

param(
    [switch]$Background,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

# Configuration
$NexusDir = "C:\Users\gary\nexus_assistant_unified\backend"
$RagnarokDir = "C:\Users\gary\python-commercial-video-agents\ragnarok_v6_legendary"
$NexusPort = 8000
$RagnarokPort = 8001

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  APEX Stack - Starting Services" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Function to check if port is in use
function Test-PortInUse {
    param([int]$Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    return $null -ne $connection
}

# Function to start a Python server
function Start-PythonServer {
    param(
        [string]$Name,
        [string]$Directory,
        [string]$Module,
        [int]$Port,
        [switch]$Background
    )

    Write-Host "`n[$Name] Starting on port $Port..." -ForegroundColor Yellow

    # Check if port is already in use
    if (Test-PortInUse -Port $Port) {
        Write-Host "[$Name] Port $Port already in use - server may be running" -ForegroundColor Green
        return $true
    }

    # Activate virtual environment if exists
    $venvPath = Join-Path $Directory "venv\Scripts\Activate.ps1"
    $venvPathAlt = Join-Path $Directory ".venv\Scripts\Activate.ps1"

    Push-Location $Directory
    try {
        if (Test-Path $venvPath) {
            Write-Host "[$Name] Activating venv..." -ForegroundColor Gray
            & $venvPath
        } elseif (Test-Path $venvPathAlt) {
            Write-Host "[$Name] Activating .venv..." -ForegroundColor Gray
            & $venvPathAlt
        }

        if ($Background) {
            # Start as background job
            $job = Start-Job -ScriptBlock {
                param($dir, $mod, $port)
                Set-Location $dir
                python -m uvicorn $mod --host 0.0.0.0 --port $port
            } -ArgumentList $Directory, $Module, $Port

            Write-Host "[$Name] Started as background job (ID: $($job.Id))" -ForegroundColor Green
        } else {
            # Start in new window
            $process = Start-Process -FilePath "python" `
                -ArgumentList "-m", "uvicorn", $Module, "--host", "0.0.0.0", "--port", $Port `
                -WorkingDirectory $Directory `
                -PassThru `
                -WindowStyle Normal

            Write-Host "[$Name] Started with PID: $($process.Id)" -ForegroundColor Green
        }

        # Wait a moment and verify
        Start-Sleep -Seconds 3
        if (Test-PortInUse -Port $Port) {
            Write-Host "[$Name] Verified running on port $Port" -ForegroundColor Green
            return $true
        } else {
            Write-Host "[$Name] Warning: Port $Port not yet active" -ForegroundColor Yellow
            return $false
        }
    }
    finally {
        Pop-Location
    }
}

# Start RAGNAROK first (dependency)
Write-Host "`n[Phase 1] Starting RAGNAROK v7 APEX..." -ForegroundColor Magenta
$ragnarokStarted = Start-PythonServer `
    -Name "RAGNAROK" `
    -Directory $RagnarokDir `
    -Module "server_commercial_bridge:app" `
    -Port $RagnarokPort `
    -Background:$Background

# Wait for RAGNAROK to be ready
if ($ragnarokStarted) {
    Write-Host "`n[Phase 2] Waiting for RAGNAROK health check..." -ForegroundColor Magenta
    $maxAttempts = 10
    $attempt = 0
    $healthy = $false

    while ($attempt -lt $maxAttempts -and -not $healthy) {
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:$RagnarokPort/health" -TimeoutSec 5
            if ($response.status -eq "healthy") {
                $healthy = $true
                Write-Host "[RAGNAROK] Health check passed!" -ForegroundColor Green
            }
        } catch {
            $attempt++
            Write-Host "[RAGNAROK] Waiting... ($attempt/$maxAttempts)" -ForegroundColor Gray
            Start-Sleep -Seconds 2
        }
    }
}

# Start Nexus Assistant
Write-Host "`n[Phase 3] Starting Nexus Assistant..." -ForegroundColor Magenta
$nexusStarted = Start-PythonServer `
    -Name "Nexus" `
    -Directory $NexusDir `
    -Module "app.main:app" `
    -Port $NexusPort `
    -Background:$Background

# Final status
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  APEX Stack Status" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if (Test-PortInUse -Port $RagnarokPort) {
    Write-Host "  RAGNAROK v7 APEX:  http://localhost:$RagnarokPort  [RUNNING]" -ForegroundColor Green
} else {
    Write-Host "  RAGNAROK v7 APEX:  http://localhost:$RagnarokPort  [NOT RUNNING]" -ForegroundColor Red
}

if (Test-PortInUse -Port $NexusPort) {
    Write-Host "  Nexus Assistant:   http://localhost:$NexusPort  [RUNNING]" -ForegroundColor Green
    Write-Host "  API Docs:          http://localhost:$NexusPort/docs" -ForegroundColor Gray
} else {
    Write-Host "  Nexus Assistant:   http://localhost:$NexusPort  [NOT RUNNING]" -ForegroundColor Red
}

Write-Host "============================================================`n" -ForegroundColor Cyan
