# Windows setup for hackathon-trader (PowerShell 5.1+).  Run from the project folder:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
#   .\setup.ps1 -NoTorch        skip PyTorch (ridge only)
# Safe to re-run. The Ubuntu/Theia server uses setup.sh instead.
param([switch]$NoTorch)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" } }

# Prefer 3.13/3.12/3.11: PyTorch and scientific wheels are most reliable there.
$pyArgs = $null
foreach ($v in "3.13", "3.12", "3.11") {
    try { & py "-$v" -c "import sys" 2>$null } catch { continue }
    if ($LASTEXITCODE -eq 0) { $pyArgs = @("-$v"); $chosen = $v; break }
}
if (-not $pyArgs) { $pyArgs = @(); $chosen = "default" }
Step "Using Python ($chosen)"
& py @pyArgs --version; Check "python"

Step "Creating virtualenv .venv"
if (-not (Test-Path ".venv\Scripts\python.exe")) { & py @pyArgs -m venv .venv; Check "venv" }
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip wheel setuptools; Check "pip upgrade"

if (-not $NoTorch) {
    Step "Installing PyTorch (CPU wheel)"
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu; Check "torch"
}

Step "Installing requirements"
& $py -m pip install -r requirements.txt; Check "requirements"

Step "Preparing .env and folders"
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }
foreach ($d in "data", "results", "models", "logs") { New-Item -ItemType Directory -Force $d | Out-Null }

Step "Running offline tests"
& $py -m pytest -q -p no:cacheprovider; Check "tests"

Write-Host @"

Setup complete.
  1. Fill in keys:     notepad .env
  2. Run everything:   powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1
  (no activation needed; the scripts call .venv\Scripts\python.exe directly)
"@ -ForegroundColor Green
