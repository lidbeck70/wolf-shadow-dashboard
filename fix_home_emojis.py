#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fix emoji encoding in tabs/home.py"""

path = r'C:\Users\lidbe\wolf-shadow-dashboard\tabs\home.py'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix nav_items with correct emojis
old_nav = '''    nav_items = [
        ("??", "ARC SCREENER",       "Scan equities across all Arc strategies"),
        ("?", "CONTRARIAN ALPHA",   "Undervalued, hated, necessary sector stocks"),
        ("??", "MARKET CYCLE",       "14-phase psychology cycle detector"),
        ("?", "WOLF REGIME",        "EMA-stack regime for swing setups"),
        ("???", "VIKING REGIME",      "OVTLYR NINE score + order-block overlay"),
        ("??", "ALPHA REGIME",       "Long-term cycle monitor for positions"),
        ("??", "FLOW DIVERGENCE",    "Global sector breadth and macro cycle"),
        ("?",  "ODIN\'S BLINDSPOT",  "Contrarian sector intelligence"),
        ("??", "HOLDINGS",           "Portfolio positions and risk exposure"),
        ("??", "TRADE JOURNAL",      "Log trades, review P&L, tag patterns"),
    ]'''

new_nav = '''    nav_items = [
        ("\U0001f531", "ARC SCREENER",       "Scan equities across all Arc strategies"),
        ("\u26a1", "CONTRARIAN ALPHA",   "Undervalued, hated, necessary sector stocks"),
        ("\U0001f504", "MARKET CYCLE",       "14-phase psychology cycle detector"),
        ("\U0001f43a", "WOLF REGIME",        "EMA-stack regime for swing setups"),
        ("\u2694\ufe0f", "VIKING REGIME",      "OVTLYR NINE score + order-block overlay"),
        ("\U0001f4c8", "ALPHA REGIME",       "Long-term cycle monitor for positions"),
        ("\U0001f310", "FLOW DIVERGENCE",    "Global sector breadth and macro cycle"),
        ("\U0001f441",  "ODIN\'S BLINDSPOT",  "Contrarian sector intelligence"),
        ("\U0001f4bc", "HOLDINGS",           "Portfolio positions and risk exposure"),
        ("\U0001f4d3", "TRADE JOURNAL",      "Log trades, review P&L, tag patterns"),
    ]'''

if old_nav in content:
    content = content.replace(old_nav, new_nav)
    print("Nav items fixed!")
else:
    # Try to find and replace just the emoji characters
    replacements = [
        ('("??", "ARC SCREENER"', '("\U0001f531", "ARC SCREENER"'),
        ('("?", "CONTRARIAN ALPHA"', '("\u26a1", "CONTRARIAN ALPHA"'),
        ('("??", "MARKET CYCLE"', '("\U0001f504", "MARKET CYCLE"'),
        ('("?", "WOLF REGIME"', '("\U0001f43a", "WOLF REGIME"'),
        ('("???", "VIKING REGIME"', '("\u2694\ufe0f", "VIKING REGIME"'),
        ('("??", "ALPHA REGIME"', '("\U0001f4c8", "ALPHA REGIME"'),
        ('("??", "FLOW DIVERGENCE"', '("\U0001f310", "FLOW DIVERGENCE"'),
        ('("?",  "ODIN\'S BLINDSPOT"', '("\U0001f441",  "ODIN\'S BLINDSPOT"'),
        ('("??", "HOLDINGS"', '("\U0001f4bc", "HOLDINGS"'),
        ('("??", "TRADE JOURNAL"', '("\U0001f4d3", "TRADE JOURNAL"'),
    ]
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"Fixed: {old[:30]}")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Done!")
