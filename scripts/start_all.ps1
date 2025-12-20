# =============================================================================
# APEX STACK v3.0 - START ALL SERVICES
# =============================================================================
# Starts: Nexus :8000, RAGNAROK :8001, Trinity :8002
# =============================================================================

param(
    [switch]$SkipHealthCheck,
    [switch]$SkipTrinity
)

$ErrorActionPreference = "Continue"

# Paths
$NexusPath = "C:\Users\gary\nexus_assistant_unified\backend"
$RagnarokPath = "C:\Users\gary\python-commercial-video-agents\ragnarok_v6_legendary"
$LogDir = "C:\Users\gary\nexus_assistant_unified\logs"

# Create log directory if not exists
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "           APEX STACK v3.0 - Trinity Enhanced               " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Services:"
Write-Host "    - Nexus Backend    :8000"
Write-Host "    - RAGNAROK v7.0    :8001"
Write-Host "    - Trinity Intel    :8002"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# =============================================================================
# PORT CHECK & CLEANUP
# =============================================================================

function Test-Port {
    param($Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    return $null -ne $conn
}

function Stop-PortProcess {
    param($Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    if ($conn) {
        $procId = $conn.OwningProcess | Select-Object -First 1
        Write-Host "  Killing process $procId on port $Port..." -ForegroundColor Yellow
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
}

Write-Host "[0/4] Checking ports..." -ForegroundColor Gray

$portsToCheck = @(8000, 8001, 8002)
foreach ($port in $portsToCheck) {
    if (Test-Port $port) {
        Write-Host "  Port $port in use, cleaning up..." -ForegroundColor Yellow
        Stop-PortProcess $port
    }
}

Write-Host "  All ports available" -ForegroundColor Green
Write-Host ""

# =============================================================================
# START SERVICES
# =============================================================================

# --- Nexus Backend :8000 ---
Write-Host "[1/4] Starting Nexus Backend on :8000..." -ForegroundColor Blue

$nexusLog = "$LogDir\nexus_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$nexusProc = Start-Process -FilePath "python" `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload" `
    -WorkingDirectory $NexusPath `
    -RedirectStandardOutput $nexusLog `
    -RedirectStandardError "$LogDir\nexus_error.log" `
    -WindowStyle Hidden `
    -PassThru

Write-Host "  Nexus started (PID: $($nexusProc.Id))" -ForegroundColor Green

# --- RAGNAROK :8001 ---
Write-Host "[2/4] Starting RAGNAROK v7.0 on :8001..." -ForegroundColor Blue

$ragnarokLog = "$LogDir\ragnarok_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$ragnarokProc = Start-Process -FilePath "python" `
    -ArgumentList "-m", "uvicorn", "server_commercial_bridge:app", "--host", "0.0.0.0", "--port", "8001", "--reload" `
    -WorkingDirectory $RagnarokPath `
    -RedirectStandardOutput $ragnarokLog `
    -RedirectStandardError "$LogDir\ragnarok_error.log" `
    -WindowStyle Hidden `
    -PassThru

Write-Host "  RAGNAROK started (PID: $($ragnarokProc.Id))" -ForegroundColor Green

# --- Trinity :8002 ---
if (-not $SkipTrinity) {
    Write-Host "[3/4] Starting Trinity Market Intelligence on :8002..." -ForegroundColor Blue

    $trinityLog = "$LogDir\trinity_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
    $trinityProc = Start-Process -FilePath "python" `
        -ArgumentList "-m", "uvicorn", "trinity_server:app", "--host", "0.0.0.0", "--port", "8002", "--reload" `
        -WorkingDirectory $RagnarokPath `
        -RedirectStandardOutput $trinityLog `
        -RedirectStandardError "$LogDir\trinity_error.log" `
        -WindowStyle Hidden `
        -PassThru

    Write-Host "  Trinity started (PID: $($trinityProc.Id))" -ForegroundColor Green
} else {
    Write-Host "[3/4] Trinity skipped (-SkipTrinity flag)" -ForegroundColor Yellow
}

Write-Host ""

# =============================================================================
# WAIT FOR STARTUP
# =============================================================================

Write-Host "[4/4] Waiting for services to initialize..." -ForegroundColor Blue
Start-Sleep -Seconds 5

# =============================================================================
# HEALTH CHECKS
# =============================================================================

if (-not $SkipHealthCheck) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "                     HEALTH CHECKS                          " -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    $nexusOk = $false
    $ragnarokOk = $false
    $trinityOk = $true

    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/api/nexus/health" -Method Get -TimeoutSec 10
        Write-Host "  Nexus Backend    (:8000) - ONLINE" -ForegroundColor Green
        $nexusOk = $true
    } catch {
        Write-Host "  Nexus Backend    (:8000) - OFFLINE" -ForegroundColor Red
    }

    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8001/health" -Method Get -TimeoutSec 10
        Write-Host "  RAGNAROK v7.0    (:8001) - ONLINE" -ForegroundColor Green
        $ragnarokOk = $true
    } catch {
        Write-Host "  RAGNAROK v7.0    (:8001) - OFFLINE" -ForegroundColor Red
    }

    if (-not $SkipTrinity) {
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:8002/health" -Method Get -TimeoutSec 10
            Write-Host "  Trinity Intel    (:8002) - ONLINE" -ForegroundColor Green
            $trinityOk = $true
        } catch {
            Write-Host "  Trinity Intel    (:8002) - OFFLINE" -ForegroundColor Red
            $trinityOk = $false
        }
    }

    Write-Host ""

    if ($nexusOk -and $ragnarokOk -and $trinityOk) {
        Write-Host "============================================================" -ForegroundColor Green
        Write-Host "              ALL SERVICES OPERATIONAL                      " -ForegroundColor Green
        Write-Host "============================================================" -ForegroundColor Green
    } else {
        Write-Host "============================================================" -ForegroundColor Yellow
        Write-Host "              SOME SERVICES FAILED TO START                 " -ForegroundColor Yellow
        Write-Host "============================================================" -ForegroundColor Yellow
    }
}

# =============================================================================
# SUMMARY
# =============================================================================

Write-Host ""
Write-Host "Service URLs:" -ForegroundColor Cyan
Write-Host "  - Nexus API:      http://localhost:8000/docs"
Write-Host "  - RAGNAROK API:   http://localhost:8001/docs"
if (-not $SkipTrinity) {
    Write-Host "  - Trinity API:    http://localhost:8002/docs"
}
Write-Host ""
Write-Host "Logs: $LogDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "To stop all services: .\stop_all.ps1" -ForegroundColor Gray
