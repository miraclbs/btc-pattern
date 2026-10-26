"""Yürüyen test, 1. aşama: sıralama yöntemlerini karşılaştırmak için aday havuzu kaydı.

Her karar anında (simulate.py ile aynı anlar), sadece o ana kadarki veriyle, arayüzdeki 1. gruba
(son tepe/dipten başlayan yapı) DTW ve birebir yöntemle şekle göre en benzer engine.POOL olay
kaydedilir. Olayların "sonrası" da karar anından önce bitmiş olmalı (sızıntı yok).

Bağlam, yeniden sıralama ve sonuçlar analyze_rank.py'de bu havuzdan hesaplanır; böylece yeni bir
sıralama fikrini denemek için arama baştan koşturulmaz.

Kullanım: .venv/Scripts/python rank_sim.py [dev|val]
"""
import sys
import time

import numpy as np
import pandas as pd

import engine
from simulate import GROUP, HORIZON, INTERVAL, PERIODS, RESULTS, STEP

METHODS = {"dtw": True, "euclid": False}


def save(path, now_idx, q_start, pools):
    arrays = {"now": np.array(now_idx, np.int64), "q_start": np.array(q_start, np.int64)}
    for name, p in pools.items():
        for k, v in p.items():
            arrays[f"{name}_{k}"] = np.array(v)
    np.savez_compressed(path, **arrays)


def run(period: str):
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"pool_{INTERVAL}_{period}.npz"
    eng = engine.Engine(INTERVAL)
    a, b = (int(np.searchsorted(eng.t, pd.Timestamp(x).timestamp())) for x in PERIODS[period])
    b = min(b, len(eng.t) - HORIZON)
    points = range(a, b, STEP)
    now_idx, q_start = [], []
    pools = {name: {"start": [], "length": [], "dist": []} for name in METHODS}
    t0 = time.perf_counter()

    for n, now in enumerate(points, 1):
        groups = eng.groups(now, eng.pivots(now))
        if len(groups) > GROUP:
            g = groups[GROUP]
            now_idx.append(now)
            q_start.append(g["start_idx"])
            for name, use_dtw in METHODS.items():
                d, s, m = eng.search(g["start_idx"], now, use_dtw, end_limit=now - HORIZON)
                picks = engine.distinct_events(d, s, m, engine.POOL)
                k = len(picks)
                # Havuzu dolmayan anlar: boş yerler start=-1, dist=inf.
                row_s, row_m = np.full(engine.POOL, -1, np.int32), np.zeros(engine.POOL, np.int32)
                row_d = np.full(engine.POOL, np.inf, np.float32)
                row_s[:k], row_m[:k], row_d[:k] = s[picks], m[picks], d[picks]
                pools[name]["start"].append(row_s)
                pools[name]["length"].append(row_m)
                pools[name]["dist"].append(row_d)

        if n % 200 == 0 or n == len(points):
            el = time.perf_counter() - t0
            print(f"{period}: {n}/{len(points)} karar anı, {len(now_idx)} kayıt, "
                  f"{el / 60:.1f} dk, kalan ~{el / n * (len(points) - n) / 60:.0f} dk", flush=True)
            save(out, now_idx, q_start, pools)
    print(f"bitti: {out}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "dev")
