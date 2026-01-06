# Setup Script
Write-Host "Setting up Aegis-X..."

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Write-Host "Setup Complete."
Write-Host "Run 'tools\run_sim.ps1' to test."

