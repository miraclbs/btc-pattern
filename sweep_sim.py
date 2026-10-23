"""Likidite alımı ve geri alma (sweep & reclaim) testi + kırılan seviye geri testinde ters işlem testi.

A. Ters işlem: setup_sim'deki geri test kurulumlarının aynısında yön ters (stop/hedef mesafesi aynı).

B. Likidite alımı (dip örneği; tepe tarafı aynadaki görüntüsü, short):
1. Seviye: onaylanmış 4S dip, onaydan sonra LEVEL_DAYS gün geçerli (ilk sarkmaya kadar).
2. Likidite alımı: 15dk'da fiyat dibin altına sarkar. Seviyenin MAX_DEPTH x 4S mum boyundan daha
   altına inerse gerçek kırılım sayılır, kurulum iptal.
3. Geri alma: sarkmanın başladığı mumdan itibaren en fazla R mum içinde bir 15dk mumu seviyenin
   üstünde kapanır.
4. İşlem: bir sonraki mumun açılışından long. Stop: sarkmanın en düşüğünün STOP_BUFFER x 4S mum
   boyu altı (MIN_STOP'tan darsa genişletilir, MAX_STOP'tan genişse atlanır). Hedef R katı.
Kaydedilenler: sarkmadaki en yüksek hacim / 30 günlük ortalama, derinlik, seviyenin ötesinde geçen
süre, günlük trend.

Önceden konan seçim kuralı: geliştirme döneminde >=100 işlemi olan varyantlardan en yüksek ortalama
R'yi veren 2024'te bir kez test edilir.

Kullanım: .venv/Scripts/python sweep_sim.py [dev|val]
"""
import sys

import numpy as np
import pandas as pd

import context
import engine
from matcher import load
from setup_sim import HORIZON, STOP_KS, TARGETS, H4Info, _trade_from, find_setups, stop_entry_fill, summary
from simulate import COST, PERIODS, RESULTS, _trade

LEVEL_MULT = 3.0
LEVEL_DAYS = 30
MAX_DEPTH = 1.5        # sarkma derinliği sınırı (4S mum boyu)
RECLAIMS = (4, 16)     # geri kapanış için en fazla bekleme: 1 saat / 4 saat (15dk mum)
STOP_BUFFER = 0.1
MIN_STOP, MAX_STOP = 0.4, 2.5


def reversed_retests(eng, a_ts, b_ts) -> pd.DataFrame:
    """A. setup_sim geri test kurulumlarında ters yön."""
    h, l, c = eng.h, eng.l, eng.c
    rows = []
    for st in find_setups(eng, LEVEL_MULT, a_ts, b_ts):
        side = -st["side"]                          # ters işlem
        i, p = st["i"], st["level"]
        # Ters yönde giriş bir STOP emridir: mum seviyenin ötesinde açtıysa açılıştan dolar.
        # (DÜZELTME: ilk sürümde her zaman seviyeden dolduğu varsayılmıştı, sonuçları şişiriyordu.)
        entry = stop_entry_fill(eng.o, i, p, side)
        row = {"time": pd.Timestamp(st["ts"], unit="s"), "idx": i, "side": "long" if side == 1 else "short"}
        for k in STOP_KS:
            risk = k * st["a4"] * p                 # stop/hedef fiyatları seviyeye göre sabit
            ra = side * (entry - (p - side * risk)) # gerçek risk
            for t in TARGETS:
                t_act = side * (p + side * t * risk - entry) / ra
                r, kind, bars = _trade_from(h, l, c, i, entry, ra / entry, t_act, HORIZON, side)
                key = f"k{k:g}_t{t:g}"
                row[f"real_{key}"], row[f"kind_{key}"], row[f"bars_{key}"] = r - COST / (ra / entry), kind, bars
        rows.append(row)
    return pd.DataFrame(rows)


def sweeps(eng, vol, avg_vol, h4: H4Info, a_ts: int, b_ts: int, reclaim_bars: int) -> pd.DataFrame:
    """B. Likidite alımı ve geri alma işlemleri."""
    o, h, l, c, t = eng.o, eng.h, eng.l, eng.c, eng.t
    rows, skipped = [], {"sarkma yok": 0, "gerçek kırılım/geri alma yok": 0, "stop sınır dışı": 0}
    for p, kind, _, confirmed in context.h4_levels(LEVEL_MULT).itertuples(index=False):
        side = 1 if kind == "dip" else -1           # dip altı sarkma -> long, tepe üstü sarkma -> short
        i0, i1 = np.searchsorted(t, [confirmed, confirmed + LEVEL_DAYS * 86400])
        pierce = np.flatnonzero(l[i0:i1] < p) if side == 1 else np.flatnonzero(h[i0:i1] > p)
        if not len(pierce):
            skipped["sarkma yok"] += 1
            continue
        s0 = int(i0 + pierce[0])
        a4 = h4.avg_at(int(t[s0]))
        ext, reclaim = s0, None
        for j in range(s0, min(s0 + reclaim_bars, len(c) - 1)):
            if side == 1:
                if l[j] < l[ext]:
                    ext = j
                if l[ext] < p * (1 - MAX_DEPTH * a4):
                    break
                if c[j] > p:
                    reclaim = j
                    break
            else:
                if h[j] > h[ext]:
                    ext = j
                if h[ext] > p * (1 + MAX_DEPTH * a4):
                    break
                if c[j] < p:
                    reclaim = j
                    break
        if reclaim is None:
            skipped["gerçek kırılım/geri alma yok"] += 1
            continue
        i = reclaim + 1
        if not a_ts <= t[i] < b_ts:
            continue
        sweep_price = l[ext] if side == 1 else h[ext]
        stop = sweep_price * (1 - side * STOP_BUFFER * a4)
        risk_frac = side * (o[i] - stop) / o[i]
        if risk_frac <= 0 or risk_frac > MAX_STOP * a4:
            skipped["stop sınır dışı"] += 1
            continue
        risk_frac = max(risk_frac, MIN_STOP * a4)
        row = {"time": pd.Timestamp(int(t[i]), unit="s"), "idx": i, "side": "long" if side == 1 else "short",
               "level": p, "a4": a4, "stop_pct": 100 * risk_frac,
               "vol_ratio": vol[s0:reclaim + 1].max() / avg_vol[s0 - 1],
               "depth": side * (p - sweep_price) / p / a4, "bars_beyond": reclaim - s0 + 1}
        for tg in TARGETS:
            r, k, bars = _trade(o, h, l, c, i, risk_frac, tg, HORIZON, side)
            key = f"t{tg:g}"
            row[f"real_{key}"], row[f"kind_{key}"], row[f"bars_{key}"] = r - COST / risk_frac, k, bars
        rows.append(row)
    print(f"  geri alma <= {reclaim_bars} mum: {len(rows)} işlem, atlananlar {skipped}")
    df = pd.DataFrame(rows)
    ctx = context.features(t[df.idx.to_numpy()], o[df.idx.to_numpy()])
    return pd.concat([df, ctx[["g_trend", "s4_trend", "sezon"]]], axis=1)


def main(period: str):
    eng = engine.Engine("15m")
    h4 = H4Info()
    m = load("15m")
    vol = m.volume.to_numpy()
    avg_vol = pd.Series(vol).rolling(2880, min_periods=96).mean().to_numpy()
    a_ts, b_ts = (int(pd.Timestamp(x).timestamp()) for x in PERIODS[period])

    print("########## A. Kırılan seviyenin geri testinde TERS işlem ##########")
    rev = reversed_retests(eng, a_ts, b_ts).sort_values("time").reset_index(drop=True)
    years = sorted(rev.time.dt.year.unique())
    for k in STOP_KS:
        for tg in TARGETS:
            key = f"k{k:g}_t{tg:g}"
            print(f"=== stop {k:g}x 4S mum, hedef {tg:g}R ===")
            summary("tüm (ters)", rev, key, years)
            summary("long (kırılan destek geri alınır)", rev[rev.side == "long"], key, years)
            summary("short (kırılan direnç geri verilir)", rev[rev.side == "short"], key, years)

    print("\n########## B. Likidite alımı ve geri alma ##########")
    candidates = []
    for rb in RECLAIMS:
        df = sweeps(eng, vol, avg_vol, h4, a_ts, b_ts, rb).sort_values("time").reset_index(drop=True)
        df.to_csv(RESULTS / f"sweep_15m_{period}_r{rb}.csv", index=False)
        years = sorted(df.time.dt.year.unique())
        trend_side = df.g_trend.map({"yukarı": "long", "aşağı": "short"})
        vol_hi = df.vol_ratio >= df.vol_ratio.median()
        print(f"  ortanca sarkma hacmi: normalin {df.vol_ratio.median():.1f} katı, ortanca derinlik "
              f"{df.depth.median():.2f}x 4S mum, ortanca stop %{df.stop_pct.median():.2f}")
        filters = {"tümü": df.index == df.index,
                   "günlük trend yönünde": (df.side == trend_side).to_numpy(),
                   "yüksek hacimli sarkma": vol_hi.to_numpy(),
                   "düşük hacimli sarkma": (~vol_hi).to_numpy(),
                   "long (dip altı sarkma)": (df.side == "long").to_numpy(),
                   "short (tepe üstü sarkma)": (df.side == "short").to_numpy()}
        for tg in TARGETS:
            key = f"t{tg:g}"
            print(f"=== geri alma <= {rb} mum ({rb / 4:g} saat), hedef {tg:g}R ===")
            for name, mask in filters.items():
                g = df[mask]
                summary(name, g, key, years)
                if name in ("tümü", "günlük trend yönünde", "yüksek hacimli sarkma") and len(g) >= 100:
                    candidates.append((g[f"real_{key}"].mean(), f"geri alma {rb}, hedef {tg:g}R, {name}", len(g)))
    print("\nseçim kuralına göre (>=100 işlem, en yüksek ort. R):")
    for mean, name, n in sorted(candidates, reverse=True)[:5]:
        print(f"  {mean:+.3f}R  {n} işlem  {name}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
