# CHANGELOG

## [2.1.0] — 2026-04-04

### Added
- **Consolidated SCREENER tab** with dropdown: Swing / Long / OVTLYR
- **Consolidated BACKTEST tab** with dropdown: Swing / Long / OVTLYR / RS Sector
- **OVTLYR Screener** (`screener_ovtlyr.py`): Z-score normalized, weighted composite
  - Trend 30%, Momentum 25%, Volatility 15%, Volume 15%, ADX 15%
  - Universe support: Nordic, US, Canada, All
- **Unified Backtest Engine** (`backtest_engine.py`): Pandas-based, 3 modes
  - Swing (EMA 10/20), Long (EMA 50/200), OVTLYR (EMA 10/20 + ADX + Volume)
- **Test Top N button**: Sends top N screener results to backtest tab
- **pytest regression tests**: Verify Long & Swing screeners unchanged
- **GitHub Actions CI**: lint (flake8) + pytest on push/PR

### Changed
- Tab count: 11 → 8 (consolidated screeners and backtests into dropdowns)
- Tab order: SCREENER | BACKTEST | OVTLYR | REGIME | SECTOR | SENTIMENT | HEATMAP | RULES

### Unchanged (verified by regression tests)
- `wolf_shadow_screener.py` — all market defs, presets, scoring logic
- `cagr/` — all fundamental scoring, technical scoring, signal gates
- `ovtlyr/` — all indicator modules, order blocks, signals, charts
- `rules_page.py` — all 44 rules (11 swing + 10 long + 23 OVTLYR)
