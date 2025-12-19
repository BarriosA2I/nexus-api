# =============================================================================
# APEX Stack - Health Check
# Checks health status of Nexus Assistant and RAGNAROK v7 APEX servers
# =============================================================================

param(
    [switch]$Detailed,
    [switch]$Json,
    [int]$Timeout = 5
)

$ErrorActionPreference = "Continue"

# Configuration
$NexusPort = 8000
$RagnarokPort = 8001
$NexusHealthUrl = "http://localhost:$NexusPort/api/nexus/health"
$RagnarokHealthUrl = "http://localhost:$RagnarokPort/health"

# Results object
$healthStatus = @{
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    overall = "unknown"
    services = @{
        nexus = @{
            status = "unknown"
            port = $NexusPort
            url = $NexusHealthUrl
            response_ms = $null
            details = $null
            error = $null
        }
        ragnarok = @{
            status = "unknown"
            port = $RagnarokPort
            url = $RagnarokHealthUrl
            response_ms = $null
            details = $null
            error = $null
        }
    }
}

function Test-ServiceHealth {
    param(
        [string]$Name,
        [string]$Url,
        [int]$Port,
        [int]$Timeout
    )

    $result = @{
        status = "unknown"
        response_ms = $null
        details = $null
        error = $null
    }

    # First check if port is listening
    $portCheck = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    if (-not $portCheck) {
        $result.status = "offline"
        $result.error = "Port $Port not listening"
        return $result
    }

    # Try HTTP health check
    try {
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $response = Invoke-RestMethod -Uri $Url -TimeoutSec $Timeout -ErrorAction Stop
        $stopwatch.Stop()

        $result.response_ms = $stopwatch.ElapsedMilliseconds
        $result.details = $response

        if ($response.status -eq "healthy" -or $response.status -eq "ok") {
            $result.status = "healthy"
        } else {
            $result.status = "degraded"
        }
    } catch {
        $result.status = "unhealthy"
        $result.error = $_.Exception.Message
    }

    return $result
}

if (-not $Json) {
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  APEX Stack - Health Check" -ForegroundColor Cyan
    Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host "============================================================" -ForegroundColor Cyan
}

# Check RAGNAROK
if (-not $Json) { Write-Host "`n[RAGNAROK v7 APEX]" -ForegroundColor Magenta }
$ragnarokHealth = Test-ServiceHealth -Name "RAGNAROK" -Url $RagnarokHealthUrl -Port $RagnarokPort -Timeout $Timeout
$healthStatus.services.ragnarok.status = $ragnarokHealth.status
$healthStatus.services.ragnarok.response_ms = $ragnarokHealth.response_ms
$healthStatus.services.ragnarok.details = $ragnarokHealth.details
$healthStatus.services.ragnarok.error = $ragnarokHealth.error

if (-not $Json) {
    switch ($ragnarokHealth.status) {
        "healthy" {
            Write-Host "  Status: HEALTHY" -ForegroundColor Green
            Write-Host "  Response: $($ragnarokHealth.response_ms)ms" -ForegroundColor Gray
        }
        "degraded" {
            Write-Host "  Status: DEGRADED" -ForegroundColor Yellow
        }
        "unhealthy" {
            Write-Host "  Status: UNHEALTHY" -ForegroundColor Red
            Write-Host "  Error: $($ragnarokHealth.error)" -ForegroundColor Red
        }
        "offline" {
            Write-Host "  Status: OFFLINE" -ForegroundColor Red
            Write-Host "  Error: $($ragnarokHealth.error)" -ForegroundColor Red
        }
    }

    if ($Detailed -and $ragnarokHealth.details) {
        Write-Host "  Details:" -ForegroundColor Gray
        $ragnarokHealth.details | ConvertTo-Json -Depth 3 | Write-Host -ForegroundColor DarkGray
    }
}

# Check Nexus
if (-not $Json) { Write-Host "`n[Nexus Assistant]" -ForegroundColor Magenta }
$nexusHealth = Test-ServiceHealth -Name "Nexus" -Url $NexusHealthUrl -Port $NexusPort -Timeout $Timeout
$healthStatus.services.nexus.status = $nexusHealth.status
$healthStatus.services.nexus.response_ms = $nexusHealth.response_ms
$healthStatus.services.nexus.details = $nexusHealth.details
$healthStatus.services.nexus.error = $nexusHealth.error

if (-not $Json) {
    switch ($nexusHealth.status) {
        "healthy" {
            Write-Host "  Status: HEALTHY" -ForegroundColor Green
            Write-Host "  Response: $($nexusHealth.response_ms)ms" -ForegroundColor Gray
        }
        "degraded" {
            Write-Host "  Status: DEGRADED" -ForegroundColor Yellow
        }
        "unhealthy" {
            Write-Host "  Status: UNHEALTHY" -ForegroundColor Red
            Write-Host "  Error: $($nexusHealth.error)" -ForegroundColor Red
        }
        "offline" {
            Write-Host "  Status: OFFLINE" -ForegroundColor Red
            Write-Host "  Error: $($nexusHealth.error)" -ForegroundColor Red
        }
    }

    if ($Detailed -and $nexusHealth.details) {
        Write-Host "  Details:" -ForegroundColor Gray
        $nexusHealth.details | ConvertTo-Json -Depth 3 | Write-Host -ForegroundColor DarkGray
    }
}

# Determine overall status
if ($nexusHealth.status -eq "healthy" -and $ragnarokHealth.status -eq "healthy") {
    $healthStatus.overall = "healthy"
} elseif ($nexusHealth.status -eq "offline" -and $ragnarokHealth.status -eq "offline") {
    $healthStatus.overall = "offline"
} elseif ($nexusHealth.status -eq "healthy" -or $ragnarokHealth.status -eq "healthy") {
    $healthStatus.overall = "degraded"
} else {
    $healthStatus.overall = "unhealthy"
}

# Output
if ($Json) {
    $healthStatus | ConvertTo-Json -Depth 5
} else {
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host "  Overall Status: " -NoNewline

    switch ($healthStatus.overall) {
        "healthy" { Write-Host "ALL SYSTEMS OPERATIONAL" -ForegroundColor Green }
        "degraded" { Write-Host "PARTIALLY OPERATIONAL" -ForegroundColor Yellow }
        "unhealthy" { Write-Host "SYSTEMS UNHEALTHY" -ForegroundColor Red }
        "offline" { Write-Host "ALL SYSTEMS OFFLINE" -ForegroundColor Red }
    }

    Write-Host "============================================================`n" -ForegroundColor Cyan

    # Usage hints
    if ($healthStatus.overall -eq "offline") {
        Write-Host "Tip: Start services with:" -ForegroundColor Yellow
        Write-Host "  .\start_all.ps1`n" -ForegroundColor Gray
    }

    if (-not $Detailed) {
        Write-Host "Tip: Use -Detailed for full response data" -ForegroundColor Gray
        Write-Host "     Use -Json for machine-readable output`n" -ForegroundColor Gray
    }
}

# Return exit code based on health
if ($healthStatus.overall -eq "healthy") {
    exit 0
} else {
    exit 1
}
