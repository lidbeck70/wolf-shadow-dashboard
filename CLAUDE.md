# CLAUDE.md

## Project
This repository contains my modular Python/Streamlit trading dashboard: Nordic Alpha Systems / wolf-shadow-dashboard.

The dashboard already includes multiple strategy views, screening logic, rules pages, holdings logic, regime views, long-term trend logic, OVTLYR-related modules, and swing trading workflows.

The goal is to extend the dashboard with an AI Trading Copilot that helps with:
- candidate discovery
- deterministic rule checking
- AI commentary and ranking
- risk gating
- trade journaling
- weekly setup review

This is NOT a full-autonomous trading bot in v1.

## Core product rules
- Preserve existing dashboard behavior unless explicitly told otherwise.
- Never perform large refactors without approval.
- Prefer reusing existing strategy and indicator logic over rewriting it.
- Make small, reviewable changes.
- Always show a plan before major implementation.
- If repo structure is unclear, inspect files first and summarize findings before editing.
- Avoid breaking Streamlit tab layout, imports, session_state, or deploy behavior.

## Architecture goals
Build an AI layer on top of the existing strategy system.

Target separation of concerns:
- data/
- indicators/
- strategies/
- ai/
- risk/
- journal/
- app/
- config/

The deterministic rule engines remain the source of truth.
AI should analyze, explain, rank, and review — not replace core rules.

## Trading scope
### Swing trading
Strategies:
- OVTLYR Golden Ticket
- EMA + Ichimoku

Typical swing rules include:
- trade with trend only
- avoid consolidation
- require key level
- enter after pullback
- candlestick trigger
- volume confirmation
- minimum 1:2 R:R, preferably 1:3
- max 1% risk per trade
- move SL to breakeven after new HH/LL
- max 2 losses per day

### Long-term investing
- regime/trend-based logic
- concentrated portfolio
- 8–10 holdings target
- max about 10% position size
- exit or reduce on red regime
- horizon roughly 0.5–3 years

## Implementation priorities
1. Analyze current repo structure.
2. Reuse existing data loaders, indicators, strategy functions, holdings logic, and rules views where possible.
3. Define shared data contracts for candidate evaluation.
4. Build deterministic engines first.
5. Build AI commentary layer second.
6. Build journaling and review layer third.
7. UI integration last, unless a thin stub UI is useful for testing.

## Safety rules
- No live autotrading in v1.
- Manual approval or paper-trading only.
- AI cannot decide risk sizing without deterministic constraints.
- Every AI recommendation must be explainable in plain text.
- Every candidate should show passed rules, failed rules, and reason for status.

## Expected statuses
Use clear statuses such as:
- PASS
- WATCH
- REJECT
- BUY CANDIDATE
- REDUCE
- EXIT

## Candidate card fields
Each candidate card should aim to include:
- ticker
- strategy
- regime
- sector score
- setup score
- passed rules
- failed rules
- AI comment
- entry zone
- stop
- target or R-multiple
- recommendation

## Journal fields
Journal entries should aim to include:
- date
- ticker
- strategy
- entry
- exit
- stop
- target
- result in %
- result in R
- exit reason
- did trade follow plan? yes/no
- AI post-trade review

## Coding rules
- Python first
- Keep modules small and focused
- Prefer typed structures / dataclasses when helpful
- Reuse existing indicator code when available
- Avoid duplication
- Add short docstrings where useful
- Add tests when practical
- Do not silently rename existing modules without reason
- Do not remove current functionality unless explicitly asked

## Workflow rules
For any non-trivial task:
1. Inspect the repo/files first.
2. Summarize current architecture.
3. Propose a file-by-file plan.
4. Wait for approval if the change is broad.
5. Implement in small steps.
6. After coding, summarize exactly what changed.

## Output style
When asked to implement:
- first give a concise plan
- then list files to create/update
- then implement
- then summarize changes and any follow-up recommendations
