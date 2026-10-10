# Windows version of run_all.sh. Full pipeline; orders are a DRY RUN unless -Place is passed.
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1 [-Place] [-SkipNews]
param([switch]$Place, [switch]$SkipNews)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Run setup.ps1 first (.venv missing)" }
New-Item -ItemType Directory -Force logs | Out-Null

function Run($label, [string[]]$a, [switch]$Optional) {
    Write-Host "`n==> $label" -ForegroundColor Cyan
    & $py @a
    if ($LASTEXITCODE -ne 0) {
        if ($Optional) { Write-Warning "$label failed (non-blocking)" } else { throw "$label failed (exit $LASTEXITCODE)" }
    }
}

Run "init tables"   @("-m", "src.init_db", "--tiger")
Run "check apis"    @("-m", "src.check_apis")
Run "ingest bars"   @("-m", "src.ingest_bars")
if (-not $SkipNews) { Run "ingest news (slow, 5-15 min)" @("-m", "src.ingest_news") }
Run "sentiment"     @("-m", "src.sentiment")
Run "features"      @("-m", "src.features")
Run "train"         @("-m", "src.train")
Run "predict"       @("-m", "src.predict")
if ($Place) { Run "decide (PAPER orders)" @("-m", "src.decide") } else { Run "decide (dry run)" @("-m", "src.decide", "--dry-run") }
Run "snowflake sync" @("-m", "src.sync_snowflake") -Optional
Write-Host "`nDone. Dashboard:  .\.venv\Scripts\streamlit run src\dashboard.py" -ForegroundColor Green
