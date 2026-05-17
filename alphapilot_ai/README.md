# AlphaPilot AI

**Advanced AI-powered trading intelligence, paper-trading, wallet-management, and strategy-training platform.**

> Disclaimer: AlphaPilot AI is for educational and research purposes only. **This is not financial advice.** The first version is **paper-trading only** and live trading is **locked and disabled by default**.

---

## What is AlphaPilot AI?

AlphaPilot AI is a complete Python-based trading intelligence system that lets you:

- Manage multiple "wallets" (each representing a trading platform/broker account)
- Run a **paper trading engine** with realistic simulated fills, fees, slippage, stop-loss, and take-profit
- Train an **AI trading engine** in the AI Training Lab using mock market data
- Build, save and backtest **trading strategies**
- Scan mock **market opportunities** across crypto, stocks, and prediction markets
- View advanced **portfolio analytics** with Plotly charts
- Log every action with full **activity logs**
- Eventually plug in real broker APIs through a clean **connector architecture** (Polymarket, Kalshi, Webull, Crypto.com, Robinhood, E*TRADE, Coinbase, Binance, Kraken, Fidelity, IBKR, custom)

The current build uses fully **mocked data** — no real API keys are required.

---

## Tech Stack

- Python 3.11+
- FastAPI (backend API)
- Streamlit (dashboard GUI)
- SQLite + SQLAlchemy (local database)
- Pydantic (validation/schemas)
- Pandas / NumPy (analytics)
- Plotly (charts)
- scikit-learn-ready structure (future ML)

---

## Installation

```bash
# 1. Clone or download the project, then cd into it
cd alphapilot_ai

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate           # macOS / Linux
.venv\Scripts\activate              # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment example
cp .env.example .env                # macOS / Linux
copy .env.example .env              # Windows
```

---

## Running the App

A single launcher starts both the FastAPI backend and the Streamlit dashboard:

```bash
python main.py
```

This will:

1. Initialize the SQLite database (`alphapilot.db`) and seed mock wallets/trades on first run
2. Start the **FastAPI backend** at `http://127.0.0.1:8000` (docs at `/docs`)
3. Start the **Streamlit dashboard** at `http://127.0.0.1:8501`

You can also run them separately:

```bash
# Backend only
uvicorn backend.api:app --reload --port 8000

# Frontend only
streamlit run app/streamlit_app.py
```

---

## Using the App

### Dashboard
Top-level command center showing portfolio value, P&L, win rate, AI confidence, drawdown, recent trades, and charts.

### Wallets
Manage one card per platform (Polymarket, Crypto.com, Webull, etc.). Each wallet stores its paper balance, trade history, P&L, and risk score.

### Add Wallet
Choose a platform, give the wallet a name, set a starting paper balance and risk profile. API fields exist as placeholders for future use; **the "Test Connection" call is mocked**.

### Wallet Detail
Drill down into a single wallet: trade history, open positions, P&L chart, AI recommendations, activity logs.

### AI Training Lab
Run simulated trading sessions on mocked market data. The AI:
- Picks trades based on a configurable strategy and risk profile
- Logs decisions, mistakes, and lessons learned
- Adjusts confidence and strategy weights over time
- Stores everything in `ai_learning_memory`

### Strategy Builder
Create, save, edit, and backtest strategies. Backtests produce mock results (win rate, P&L, drawdown, risk score, recommendation).

### Market Scanner
Scans mocked opportunities across crypto, stocks, and prediction markets. Shows AI edge %, confidence, liquidity, volatility, and a suggested action.

### Analytics
Deep portfolio analytics: P&L by wallet/strategy/market, win rate, drawdown, profit factor, Sharpe placeholder, AI improvement over time.

### Activity Logs
Every API attempt, paper trade, AI decision, warning, strategy change, and emergency stop is logged.

### Settings
Account, trading, wallet, AI, security, and database settings, including:
- **Live trading lock** (on by default)
- Emergency stop button
- Reset paper trades / AI memory / database

---

## How Paper Trading Works

The paper trading engine (`trading/paper_trading_engine.py`):

- Validates every trade through the **risk manager** (max position, max daily loss, confidence threshold, etc.)
- Estimates fees and slippage
- Tracks open and closed positions
- Calculates realized + unrealized P&L
- Persists trades to SQLite

Live trading methods exist as **placeholders only** and raise an exception:

```python
raise PermissionError("Live trading is locked by default. Enable explicitly in settings.")
```

---

## How AI Training Works

The AI engine (`ai/ai_engine.py`) coordinates:

- `decision_engine.py` – chooses trades based on strategy + signals
- `mistake_analyzer.py` – detects bad entries, holds-too-long, overtrading, etc.
- `learning_memory.py` – stores lessons learned (e.g. "avoid low-liquidity markets after 3 slippage losses")
- `strategy_optimizer.py` – adjusts strategy weights based on performance
- `model_placeholder.py` – scaffold for a future scikit-learn / PyTorch model

The current logic is rule-based + stochastic, but the architecture is ready for real ML.

---

## Where Future Real APIs Will Be Added

Each broker has a connector in `connectors/` inheriting from `BaseConnector`. They currently return mock data. To plug in a real API:

1. Open the relevant connector (e.g. `connectors/coinbase_connector.py`)
2. Replace the body of `connect`, `fetch_balance`, `fetch_positions`, `fetch_market_data`, etc. with real API calls
3. Encrypt and store credentials in `api_credentials_placeholder` (replace placeholder with proper encrypted storage)
4. Only enable `place_live_trade` after reviewing all risk controls and explicitly setting `LIVE_TRADING_ENABLED=true`

---

## Resetting the Database

From the **Settings → Database** page, or:

```bash
python -c "from database.db import reset_db; reset_db()"
```

---

## Extending Connectors / Adding a Platform

1. Create `connectors/myplatform_connector.py` extending `BaseConnector`
2. Implement the required methods (start with mocks)
3. Register the platform name in `utils/constants.py → SUPPORTED_PLATFORMS`
4. The "Add Wallet" page will pick it up automatically

---

## Roadmap

- Real API integrations (one connector at a time)
- Encrypted credential vault
- Real ML model (scikit-learn → PyTorch)
- Live trading with manual approval and hard risk caps
- Multi-user accounts
- Cloud sync (optional)
- Mobile companion app

---

## Safety

- Paper trading only by default
- Live trading **locked**
- Emergency stop always one click away
- Every trade attempt is risk-checked
- All major actions are logged

**This software is provided AS IS, with no warranty. It is not financial advice. Trade at your own risk.**
