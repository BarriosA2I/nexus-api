# Verify Qdrant data
$baseDir = "C:\Users\gary\nexus_assistant_unified\backend"
Set-Location $baseDir

# Load environment
$envFile = Get-Content "$baseDir\.env"
foreach ($line in $envFile) {
    if ($line -match '^([^#][^=]+)=(.*)$') {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim()
        [System.Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
}

python "$baseDir\verify_qdrant.py"
