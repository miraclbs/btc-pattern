"""Büyük zaman dilimi (HTF) bağlamı: her an için, sadece o ana kadar KAPANMIŞ mumlarla hesaplanan
etiketler. Simülasyonda sızıntı olmasın diye her bilgi ancak kullanılabilir olduğu andan sonra
eşleştirilir (günlük mum kapanınca, 4S tepe/dip onaylanınca).

Etiketler:
  g_trend   : günlük trend      - yukarı / yatay / aşağı   (fiyat 50 günlük ortalamaya göre + ortalamanın eğimi)
  sezon     : boğa / ayı        (fiyat 200 günlük ortalamanın üstünde mi)
  s4_trend  : 4 saatlik trend   - yukarı / yatay / aşağı
  konum     : son 90 günlük aralıkta - dipte / ortada / tepede
  momentum  : son 30 gün        - pozitif / negatif
  seviye    : en yakın 4S tepe/dip seviyesine göre - dirence yakın / desteğe yakın / arada
"""
import numpy as np
import pandas as pd

from matcher import load

PIVOT_MULT = 4.0       # 4S tepe/dip: fiyat 4S normal mum boyunun bu katı dönünce onaylanır (engine ile aynı)
LEVEL_DAYS = 90        # seviye olarak son kaç günün tepe/diplerine bakılır
NEAR = 0.5             # seviyeye "yakın": günlük normal mum boyunun bu katından yakın


def _trend(c: pd.Series, n: int = 50, slope_lag: int = 10) -> pd.Series:
    sma = c.rolling(n).mean()
    rising = sma > sma.shift(slope_lag)
    out = np.where((c > sma) & rising, "yukarı", np.where((c < sma) & ~rising, "aşağı", "yatay"))
    return pd.Series(np.where(sma.isna(), None, out), index=c.index)


def daily_features(symbol: str = "BTCUSDT") -> pd.DataFrame:
    d = load("1d", symbol)
    c = d.close
    f = pd.DataFrame({"avail": d.open_time // 1000 + 86400})          # mum kapanınca kullanılabilir
    f["g_trend"] = _trend(c)
    sma200 = c.rolling(200).mean()
    f["sezon"] = np.where(sma200.isna(), None, np.where(c > sma200, "boğa", "ayı"))
    lo, hi = d.low.rolling(90).min(), d.high.rolling(90).max()
    pos = (c - lo) / (hi - lo)
    f["konum"] = np.where(pos.isna(), None, np.where(pos < 0.25, "dipte", np.where(pos > 0.75, "tepede", "ortada")))
    ret = c / c.shift(30) - 1
    f["momentum"] = np.where(ret.isna(), None, np.where(ret > 0, "pozitif", "negatif"))
    f["g_atr"] = ((d.high - d.low) / c).rolling(14).mean()            # günlük normal mum boyu (oran)
    return f


def h4_features(symbol: str = "BTCUSDT") -> pd.DataFrame:
    d = load("4h", symbol)
    return pd.DataFrame({"avail": d.open_time // 1000 + 4 * 3600, "s4_trend": _trend(d.close)})


def h4_levels(mult: float = PIVOT_MULT, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """4S zigzag tepe/dipleri: (seviye fiyatı, tür, oluştuğu an, onaylandığı an). Onaydan önce bilinmez.
    mult: dönüş eşiği, 4S normal mum boyunun katı (küçüldükçe daha çok, daha küçük seviye)."""
    d = load("4h", symbol)
    t = (d.open_time // 1000).to_numpy()
    h, l = d.high.to_numpy(), d.low.to_numpy()
    avg = pd.Series((h - l) / d.close).rolling(180, min_periods=24).mean().to_numpy()
    rows, direction, hi, lo = [], 0, 0, 0
    for i in range(1, len(h)):
        if not avg[i] > 0:
            hi = lo = i
            continue
        thr = mult * avg[i]
        if direction >= 0 and h[i] >= h[hi]:
            hi = i
        if direction <= 0 and l[i] <= l[lo]:
            lo = i
        confirm = t[i] + 4 * 3600                     # onay, i. mum kapanınca bilinir
        if direction == 0:
            if lo < hi and h[hi] >= l[lo] * (1 + thr):
                rows.append((l[lo], "dip", t[lo], confirm)); direction = 1
            elif hi < lo and l[lo] <= h[hi] * (1 - thr):
                rows.append((h[hi], "tepe", t[hi], confirm)); direction = -1
        elif direction == 1 and hi < i and l[i] <= h[hi] * (1 - thr):
            rows.append((h[hi], "tepe", t[hi], confirm)); direction = -1; lo = i
        elif direction == -1 and lo < i and h[i] >= l[lo] * (1 + thr):
            rows.append((l[lo], "dip", t[lo], confirm)); direction = 1; hi = i
    return pd.DataFrame(rows, columns=["price", "kind", "formed", "confirmed"])


def features(ts: np.ndarray, price: np.ndarray, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """ts anlarında (unix saniye) ve price fiyatında bağlam etiketleri. Sıra korunur."""
    left = pd.DataFrame({"ts": ts.astype(np.int64), "price": price, "_order": np.arange(len(ts))}).sort_values("ts")
    for right in (daily_features(symbol), h4_features(symbol)):
        left = pd.merge_asof(left, right.sort_values("avail"), left_on="ts", right_on="avail",
                             direction="backward").drop(columns="avail")
    lv = h4_levels(symbol=symbol)
    lv_price, lv_formed, lv_conf = lv.price.to_numpy(), lv.formed.to_numpy(), lv.confirmed.to_numpy()
    up, down = [], []
    for t, p in zip(left.ts.to_numpy(), left.price.to_numpy()):
        m = (lv_conf <= t) & (lv_formed >= t - LEVEL_DAYS * 86400)
        above, below = lv_price[m & (lv_price > p)], lv_price[m & (lv_price < p)]
        up.append((above.min() - p) / p if len(above) else np.nan)
        down.append((p - below.max()) / p if len(below) else np.nan)
    left["direnc_uzaklik"] = np.array(up) / left.g_atr          # günlük normal mum boyu cinsinden
    left["destek_uzaklik"] = np.array(down) / left.g_atr
    left["seviye"] = np.where(left.direnc_uzaklik < NEAR, "dirence yakın",
                              np.where(left.destek_uzaklik < NEAR, "desteğe yakın", "arada"))
    return left.sort_values("_order").drop(columns="_order").reset_index(drop=True)
