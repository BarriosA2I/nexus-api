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

Write-Host "Environment loaded. Running Oracle test..."
if ($env:PERPLEXITY_API_KEY) {
    Write-Host "PERPLEXITY_API_KEY: $($env:PERPLEXITY_API_KEY.Substring(0,8))..."
} else {
    Write-Host "PERPLEXITY_API_KEY: NOT SET"
}
if ($env:ANTHROPIC_API_KEY) {
    Write-Host "ANTHROPIC_API_KEY: $($env:ANTHROPIC_API_KEY.Substring(0,8))..."
} else {
    Write-Host "ANTHROPIC_API_KEY: NOT SET"
}

python "$baseDir\app\services\test_oracle.py"

# To run research instead, use:
# python "$baseDir\run_research.py" dental_practices
