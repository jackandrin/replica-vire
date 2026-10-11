# Windows version of run_all.sh. Full v2 pipeline on one machine; orders are a DRY RUN unless -Place is passed.
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1 [-Place] [-SkipDocs] [-Since 2022-01-01]
param([switch]$Place, [switch]$SkipDocs, [string]$Since = "")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Run setup.ps1 first (.venv missing)" }
New-Item -ItemType Directory -Force logs | Out-Null
$s = @(); if ($Since) { $s = @("--since", $Since) }
$st = @(); if ($Since) { $st = @("--start", $Since) }

function Run($label, [string[]]$a, [switch]$Optional) {
    Write-Host "`n==> $label" -ForegroundColor Cyan
    & $py @a
    if ($LASTEXITCODE -ne 0) {
        if ($Optional) { Write-Warning "$label failed (non-blocking)" } else { throw "$label failed (exit $LASTEXITCODE)" }
    }
}

Run "init tables"      @("-m", "src.init_db", "--tiger")
Run "check apis"       @("-m", "src.check_apis") -Optional
Run "bars (15m + 1d)"  (@("-m", "src.ingest_intraday") + $st)
Run "events"           (@("-m", "src.ingest_events") + $s)
if (-not $SkipDocs) { Run "documents (slow)" (@("-m", "src.ingest_docs") + $s) }
Run "export bundle"    (@("-m", "src.export_dataset") + $st)
Run "sentiment agent"  @("-m", "src.sentiment_agent", "--backend", "cortex", "--source", "bundle", "--sink", "bundle")
Run "import sentiment" @("-m", "src.sentiment_agent", "--import-bundle")
Run "re-export bundle" (@("-m", "src.export_dataset") + $st)
Run "train"            @("-m", "src.train", "--rebuild-features")
Run "publish model"    @("-m", "src.publish_models", "--no-sentiment")
Run "predict"          @("-m", "src.predict")
if ($Place) { Run "autotrade (PAPER orders)" @("-m", "src.autotrade") } else { Run "autotrade (alerts only)" @("-m", "src.autotrade", "--no-orders") }
Run "snowflake sync"   @("-m", "src.sync_snowflake") -Optional
Write-Host "`nDone. API: .\.venv\Scripts\python -m uvicorn src.web_api:app --port 8000 ; frontend: cd ..\frontend; npm run dev" -ForegroundColor Green
