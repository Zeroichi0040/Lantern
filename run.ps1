# Launch Lantern on Windows.
# If PowerShell blocks this script: powershell -ExecutionPolicy Bypass -File .\run.ps1
Set-Location -Path $PSScriptRoot

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command py -ErrorAction SilentlyContinue }
if (-not $py) {
    Write-Error "Python 3.9+ is required but was not found on PATH."
    exit 1
}
& $py.Source app.py @args
