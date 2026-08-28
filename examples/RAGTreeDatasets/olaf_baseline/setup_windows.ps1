$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

$Py = $null
try {
    & py -3.11 --version | Out-Null
    $Py = @("py", "-3.11")
} catch {
    $Py = @("python")
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating isolated .venv..."
    if ($Py[0] -eq "py") {
        & py -3.11 -m venv .venv
    } else {
        & python -m venv .venv
    }
}

$Python = Join-Path $Here ".venv\Scripts\python.exe"
$Pip = Join-Path $Here ".venv\Scripts\pip.exe"

& $Python -m pip install --upgrade pip setuptools wheel
& $Pip install -r requirements-windows.txt

# Install the vendored OLAF package without its upstream dependency list.
# The upstream list pins triton, which is not appropriate for native Windows.
& $Pip install -e vendor\olaf --no-deps

# Small spaCy model is enough for this baseline and keeps the environment light.
& $Python -m spacy download en_core_web_sm
& $Python -c "import spacy; spacy.load('en_core_web_sm'); print('spaCy model verified: en_core_web_sm')"

# Register a dedicated Jupyter kernel.
& $Python -m ipykernel install --user --name olaf-baseline --display-name "OLAF Baseline (.venv)"

Write-Host ""
Write-Host "OLAF baseline environment ready."
Write-Host "Kernel: OLAF Baseline (.venv)"
Write-Host "Python: $Python"
