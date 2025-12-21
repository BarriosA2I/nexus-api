# Run Nexus Research Oracle
param(
    [string]$Industry = "dental_practices"
)

# Set working directory
$baseDir = "C:\Users\gary\nexus_assistant_unified\backend"
Set-Location $baseDir

# Load environment variables from .env
$envFile = Get-Content "$baseDir\.env"
foreach ($line in $envFile) {
    if ($line -match '^([^#][^=]+)=(.*)$') {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim()
        [System.Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
}

Write-Host ""
Write-Host "========================================"
Write-Host "NEXUS RESEARCH ORACLE"
Write-Host "========================================"
Write-Host "Industry: $Industry"
Write-Host ""

python "$baseDir\run_research.py" $Industry
