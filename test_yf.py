import yfinance as yf
import pandas as pd

print("Testing yfinance...")
df = yf.download('SPY', period='1y', auto_adjust=True, progress=False)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"MultiIndex: {isinstance(df.columns, pd.MultiIndex)}")
if not df.empty:
    print(f"First row: {df.head(1)}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        print(f"Flattened columns: {df.columns.tolist()}")
else:
    print("EMPTY DataFrame!")
