# Rebuild Python Virtual Environment
$venv = ".venv"

Write-Host "Removing existing .venv..."
if (Test-Path $venv) {
    Remove-Item -Recurse -Force $venv
}

Write-Host "Creating new .venv..."
python -m venv $venv

Write-Host "Activating .venv..."
& ".\$venv\Scripts\Activate.ps1"

Write-Host ("Python: " + (python -c "import sys; print(sys.executable)"))

# Verify we are in the correct venv
python -c "import sys, os; sys.exit(0 if os.path.abspath(sys.executable).lower().startswith(os.path.abspath('.venv').lower()) else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to activate .venv. Aborting to protect global environment."
    exit 1
}

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing requirements..."
python -m pip install -r requirements.txt -c constraints.txt
python -m pip install -r requirements-dev.txt -c constraints.txt

Write-Host "Running tests..."
python -m pytest -q

Write-Host "Done."
