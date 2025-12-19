# =============================================================================
# APEX Stack - Stop All Services
# Stops Nexus Assistant and RAGNAROK v7 APEX servers
# =============================================================================

param(
    [switch]$Force,
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"

# Configuration
$NexusPort = 8000
$RagnarokPort = 8001

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  APEX Stack - Stopping Services" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Function to stop process by port
function Stop-ProcessByPort {
    param(
        [string]$Name,
        [int]$Port,
        [switch]$Force
    )

    Write-Host "`n[$Name] Stopping processes on port $Port..." -ForegroundColor Yellow

    # Find processes using the port
    $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue

    if (-not $connections) {
        Write-Host "[$Name] No process found on port $Port" -ForegroundColor Gray
        return $true
    }

    $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    $stopped = 0

    foreach ($pid in $pids) {
        try {
            $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
            if ($process) {
                $processName = $process.ProcessName

                if ($Force) {
                    Stop-Process -Id $pid -Force
                    Write-Host "[$Name] Force killed PID $pid ($processName)" -ForegroundColor Yellow
                } else {
                    Stop-Process -Id $pid
                    Write-Host "[$Name] Stopped PID $pid ($processName)" -ForegroundColor Green
                }
                $stopped++
            }
        } catch {
            Write-Host "[$Name] Failed to stop PID $pid`: $_" -ForegroundColor Red
        }
    }

    # Verify port is free
    Start-Sleep -Seconds 1
    $stillInUse = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue

    if ($stillInUse) {
        Write-Host "[$Name] Warning: Port $Port still in use" -ForegroundColor Yellow
        return $false
    } else {
        Write-Host "[$Name] Port $Port is now free" -ForegroundColor Green
        return $true
    }
}

# Also stop any background jobs
Write-Host "`n[Phase 1] Stopping PowerShell background jobs..." -ForegroundColor Magenta
$jobs = Get-Job | Where-Object { $_.State -eq "Running" }
if ($jobs) {
    foreach ($job in $jobs) {
        Stop-Job -Job $job
        Remove-Job -Job $job
        Write-Host "Stopped job: $($job.Id) - $($job.Name)" -ForegroundColor Gray
    }
} else {
    Write-Host "No background jobs found" -ForegroundColor Gray
}

# Stop Nexus first (it depends on RAGNAROK)
Write-Host "`n[Phase 2] Stopping Nexus Assistant..." -ForegroundColor Magenta
$nexusStopped = Stop-ProcessByPort -Name "Nexus" -Port $NexusPort -Force:$Force

# Stop RAGNAROK
Write-Host "`n[Phase 3] Stopping RAGNAROK v7 APEX..." -ForegroundColor Magenta
$ragnarokStopped = Stop-ProcessByPort -Name "RAGNAROK" -Port $RagnarokPort -Force:$Force

# Final status
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  APEX Stack - Shutdown Complete" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$connection8000 = Get-NetTCPConnection -LocalPort $NexusPort -ErrorAction SilentlyContinue
$connection8001 = Get-NetTCPConnection -LocalPort $RagnarokPort -ErrorAction SilentlyContinue

if (-not $connection8000) {
    Write-Host "  Nexus (port $NexusPort):     [STOPPED]" -ForegroundColor Green
} else {
    Write-Host "  Nexus (port $NexusPort):     [STILL RUNNING]" -ForegroundColor Red
}

if (-not $connection8001) {
    Write-Host "  RAGNAROK (port $RagnarokPort):  [STOPPED]" -ForegroundColor Green
} else {
    Write-Host "  RAGNAROK (port $RagnarokPort):  [STILL RUNNING]" -ForegroundColor Red
}

Write-Host "============================================================`n" -ForegroundColor Cyan

# Tip for force stop
if ($connection8000 -or $connection8001) {
    Write-Host "Tip: Use -Force flag to forcefully terminate stubborn processes" -ForegroundColor Yellow
    Write-Host "  .\stop_all.ps1 -Force`n" -ForegroundColor Gray
}
