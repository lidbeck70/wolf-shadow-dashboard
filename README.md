# WOLF x SHADOW Screener v1.0

Scannar aktier med samma 4-lager regime scoring som Pine Script-strategin.

## Installation

```bash
pip install -r requirements.txt
```

## Användning

```bash
# Alla marknader (Commodity, S&P 500, Stockholm, Oslo)
python wolf_shadow_screener.py

# Bara en marknad
python wolf_shadow_screener.py --market commodity
python wolf_shadow_screener.py --market sp500
python wolf_shadow_screener.py --market stockholm
python wolf_shadow_screener.py --market oslo

# Filtrera på minsta score
python wolf_shadow_screener.py --min-score 70
```

## Scoring

| Layer | Max | Vad den mäter |
|-------|-----|---------------|
| Market (SPY) | 30 | SPY > EMA50/200, RSI > 50, normal ATR |
| Sector (ETF) | 30 | Sektor-ETF > EMA50/200, RSI > 50 |
| Stock | 50 | EMA stack, RSI, momentum |
| Ichimoku | 15 | Ovanför Kumo, TK-cross, Chikou, Kumo twist |
| **Total** | **125** | |

## Output

Sparar CSV i `output/`-mappen med alla scores, entry zones, SL och TP-nivåer.

## Schemalagd körning

### Windows (Task Scheduler)
1. Öppna Task Scheduler
2. Create Basic Task → Daglig → 08:00
3. Action: Start a program → `python` → Arguments: `wolf_shadow_screener.py`

### Mac/Linux (cron)
```bash
# Kör varje vardag kl 08:00
0 8 * * 1-5 cd /path/to/screener && python wolf_shadow_screener.py
```

## Marknader

- **Commodity**: Olja, gas, guld, silver (miners + ETFer)
- **S&P 500**: Top 70 aktier efter vikt
- **Stockholm**: OMXS30 + commodity-relaterade svenska bolag
- **Oslo**: OBX-index (Equinor, Aker BP, Norsk Hydro m.fl.)
