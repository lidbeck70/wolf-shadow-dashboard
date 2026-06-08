with open(r'C:\Users\lidbe\wolf-shadow-dashboard\tabs\home.py', encoding='utf-8') as f:
    lines = f.readlines()
for i, l in enumerate(lines[213:220], 214):
    print(i, repr(l[:80]))
