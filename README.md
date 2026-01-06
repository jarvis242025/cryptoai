# Aegis-X Sniper–Shield

![CI](https://github.com/placeholder/aegisx/actions/workflows/ci.yml/badge.svg)

A production-grade algorithmic trading system with a simulation-first architecture.

## Features
- **Triple-Barrier Labeling**: Precise target definition for ML training.
- **Simulation-First**: Realistic accounting including fees, slippage, and partial fills simulation.
- **Risk Gates**: Volatility caps, news veto (RSS based), anomaly detection (Isolation Forest).
- **Ratchet Stops**: Trailing logic to lock in profits.
- **Safety**: Hard kill-switches and strict live-mode gating.

## Setup

1. **Environment Setup**
   Always activate the virtual environment (`.venv`) before running commands to ensure dependencies are loaded correctly. Running with the system Python may cause version conflicts.

   **Recommended:** Run the included script to clean and rebuild the environment:
   ```powershell
   .\scripts\rebuild_venv.ps1
   ```

   **Manual Setup:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt -c constraints.txt
   pip install -r requirements-dev.txt -c constraints.txt
   ```

2. **Configuration**
   Copy `.env.example` to `.env` and adjust settings.
   ```bash
   cp .env.example .env
   ```

## Usage

### 1. Training
Fetch data, engineer features, and train the Random Forest model.
```bash
python -m aegisx.cli train --days 365
```

### 2. Simulation
Run a backtest/simulation using the trained model.
```bash
python -m aegisx.cli sim --days 30
```

### 3. Live Trading (CAUTION)
Requires `LIVE_ALLOWED=true` in `.env` and explicit flags.
```bash
python -m aegisx.cli live --i-understand-this-can-lose-money
```

## Testing
Run the test suite:
```bash
python -m pytest
```
