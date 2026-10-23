"""Binance mum verisini indirir ve data/ klasörüne CSV olarak kaydeder.

Kullanım: python fetch_data.py [interval] [sembol]   (varsayılan: 1h BTCUSDT)
Dosya zaten varsa sadece eksik kalan son kısmı indirip ekler.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

SYMBOL = "BTCUSDT"
URL = "https://api.binance.com/api/v3/klines"
DATA_DIR = Path(__file__).parent / "data"
COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]


def fetch(interval: str, symbol: str = SYMBOL) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{symbol.lower()}_{interval}.csv"

    rows = []
    start = 0
    if path.exists():
        old = pd.read_csv(path)
        start = int(old["open_time"].iloc[-1]) + 1
    else:
        old = None

    while True:
        r = requests.get(URL, params={"symbol": symbol, "interval": interval,
                                      "startTime": start, "limit": 1000}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        now_ms = time.time() * 1000
        # Sadece kapanmış mumlar (k[6] = kapanış zamanı); oluşmakta olan mum alınmaz.
        rows.extend(k[:6] for k in batch if k[6] < now_ms)
        start = batch[-1][0] + 1
        print(f"\r{len(rows)} mum indirildi, son: {pd.to_datetime(batch[-1][0], unit='ms')}", end="")
        if len(batch) < 1000:
            break
        time.sleep(0.1)
    print()

    new = pd.DataFrame(rows, columns=COLUMNS)
    new[COLUMNS[1:]] = new[COLUMNS[1:]].astype(float)
    df = pd.concat([old, new]) if old is not None else new
    df = df.drop_duplicates("open_time")
    if len(new):
        df.to_csv(path, index=False)
    print(f"{path.name}: {len(df)} mum ({len(new)} yeni), "
          f"{pd.to_datetime(df.open_time.iloc[0], unit='ms')} -> "
          f"{pd.to_datetime(df.open_time.iloc[-1], unit='ms')}")
    return path


if __name__ == "__main__":
    fetch(sys.argv[1] if len(sys.argv) > 1 else "1h", sys.argv[2] if len(sys.argv) > 2 else SYMBOL)
