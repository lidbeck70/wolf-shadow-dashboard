import re

path = r'C:\Users\lidbe\wolf-shadow-dashboard\wolf_panel.py'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace broken emoji tab labels with correct ones
replacements = {
    '"  ?? HOME  "': '"  🏠 HOME  "',
    '"  ?? ARC SCREENER  "': '"  🔱 ARC SCREENER  "',
    '"  ? CONTRARIAN ALPHA  "': '"  ⚡ CONTRARIAN ALPHA  "',
    '"  ?? MARKET CYCLE  "': '"  🔄 MARKET CYCLE  "',
    '"  ? WOLF REGIME  "': '"  🐺 WOLF REGIME  "',
    '"  ?? ALPHA REGIME  "': '"  📈 ALPHA REGIME  "',
    '"  ??? VIKING REGIME  "': '"  ⚔️ VIKING REGIME  "',
    '"  ?? FLOW DIVERGENCE  "': '"  🌐 FLOW DIVERGENCE  "',
    '"  ? ODIN\'S BLINDSPOT  "': '"  👁 ODIN\'S BLINDSPOT  "',
    '"  ?? SENTIMENT  "': '"  📡 SENTIMENT  "',
    '"  ?? RETAIL PULSE  "': '"  🛒 RETAIL PULSE  "',
    '"  ?? HEATMAP  "': '"  📊 HEATMAP  "',
    '"  ?? HOLDINGS  "': '"  💼 HOLDINGS  "',
    '"  ?? TRADE JOURNAL  "': '"  📓 TRADE JOURNAL  "',
    '"  ??? BACKTEST  "': '"  ⚙️ BACKTEST  "',
    '"  ?? RULES  "': '"  📋 RULES  "',
    '"  ?? ALERTS  "': '"  🔔 ALERTS  "',
    '"  ?? ARC STRATEGIES  "': '"  🧬 ARC STRATEGIES  "',
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Done")
