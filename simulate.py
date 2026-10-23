"""Yürüyen (walk-forward) simülasyon: benzer yapılar işleme çevrilince para kazandırıyor mu?

Her karar anında, sadece o ana kadarki veriyle:
1. Son tepe/dipten başlayan yapıya (arayüzdeki 1. grup) en benzer TOP_K geçmiş olay bulunur.
   Olayların "sonrası" da karar anından önce bitmiş olmalı (sızıntı yok).
2. Her işlem ayarı (stop, hedef) ve yön için aynı işlem analogların sonrasında denenir:
   tahmini ortalama R ve hedefe ulaşma oranı.
3. Aynı işlemin gerçekte ne verdiği hesaplanır.

Kayıtlar results/sim/ altına CSV olarak yazılır, analiz analyze_sim.py ile yapılır.
Son sınav dönemi (2025 ve sonrası) burada bilerek yok; sadece kurallar kesinleşince eklenecek.

Kullanım: .venv/Scripts/python simulate.py [dev|val]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

import engine

INTERVAL = "15m"
PERIODS = {"dev": ("2018-07-01", "2024-01-01"),     # geliştirme: kurallar burada denenir
           "val": ("2024-01-01", "2025-01-01")}     # kontrol: seçilen kurallar burada doğrulanır
STEP = 32                       # karar anları arası mum (15dk'da 8 saat)
GROUP = 0                       # hangi yapı: 0 = son tepe/dipten başlayan
TOP_K = 50                      # kaç benzer olaya bakılır
HORIZON = 288                   # işlem en fazla bu kadar mum açık kalır (15dk'da 3 gün)
STOP_MULTS = np.array([3.0, 6.0])        # stop = normal mum boyunun bu katı
TARGETS = np.array([1.5, 2.0, 3.0])      # hedef = stop mesafesinin bu katı (R)
FEE, SLIP = 0.0005, 0.0002               # vadeli komisyon ve kayma, giriş ve çıkışta ayrı ayrı
COST = 2 * (FEE + SLIP)                  # bir işlemin toplam masrafı (fiyatın oranı olarak)
SIDES = ("long", "short")
RESULTS = Path(__file__).parent / "results" / "sim"


@njit(cache=True)
def _trade(o, h, l, c, i, risk_frac, target_r, horizon, side):
    """i. mumun açılışında girilen işlem: (R sonucu, tür, kaç mum sürdü).
    Tür: 1 hedef, -1 stop, 0 süre doldu (kapanıştan çıkış). Aynı mumda ikisi de görülürse stop sayılır."""
    entry = o[i]
    risk = risk_frac * entry
    stop = entry - side * risk
    target = entry + side * target_r * risk
    end = min(i + horizon, len(c))
    for j in range(i, end):
        if side == 1:
            if l[j] <= stop:
                return -1.0, -1, j - i + 1
            if h[j] >= target:
                return target_r, 1, j - i + 1
        else:
            if h[j] >= stop:
                return -1.0, -1, j - i + 1
            if l[j] <= target:
                return target_r, 1, j - i + 1
    return side * (c[end - 1] - entry) / risk, 0, end - i


@njit(cache=True)
def _outcomes(o, h, l, c, avg_range, entries, stop_mults, targets, horizon):
    """Her giriş noktası, stop, hedef ve yön için sonuç: R, tür ve süre dizileri (n, stop, hedef, yön)."""
    shape = (len(entries), len(stop_mults), len(targets), 2)
    R, kind, bars = np.empty(shape), np.empty(shape, np.int64), np.empty(shape, np.int64)
    for a in range(len(entries)):
        i = entries[a]
        for si in range(len(stop_mults)):
            frac = stop_mults[si] * avg_range[i - 1]     # stop, o dönemin normal mum boyuna göre
            for ti in range(len(targets)):
                for sd in range(2):
                    r, k, n = _trade(o, h, l, c, i, frac, targets[ti], horizon, 1 if sd == 0 else -1)
                    R[a, si, ti, sd], kind[a, si, ti, sd], bars[a, si, ti, sd] = r, k, n
    return R, kind, bars


def key(si: int, ti: int, sd: int) -> str:
    return f"s{STOP_MULTS[si]:g}_t{TARGETS[ti]:g}_{SIDES[sd]}"


def run(period: str):
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{INTERVAL}_{period}.csv"
    eng = engine.Engine(INTERVAL)
    o, h, l, c, avg = eng.o, eng.h, eng.l, eng.c, eng.avg_range
    a, b = (int(np.searchsorted(eng.t, pd.Timestamp(x).timestamp())) for x in PERIODS[period])
    b = min(b, len(eng.t) - HORIZON)
    points = range(a, b, STEP)
    rows, t0 = [], time.perf_counter()

    for n, now in enumerate(points, 1):
        groups = eng.groups(now, eng.pivots(now))
        if len(groups) > GROUP:
            g = groups[GROUP]
            d, s, m = eng.search(g["start_idx"], now, True, end_limit=now - HORIZON)
            picks = engine.distinct_events(d, s, m, TOP_K)
            if len(picks) >= TOP_K // 2:
                ends = (s[picks] + m[picks]).astype(np.int64)
                Ra, Ka, _ = _outcomes(o, h, l, c, avg, ends, STOP_MULTS, TARGETS, HORIZON)
                Rn, Kn, Bn = _outcomes(o, h, l, c, avg, np.array([now], np.int64), STOP_MULTS, TARGETS, HORIZON)
                row = {"time": pd.Timestamp(eng.t[now], unit="s"), "idx": now, "kind": g["kind"],
                       "length": g["length"], "n_analogs": len(picks),
                       "similarity": float(np.mean(100 * (1 - d[picks] / 2)))}
                for si in range(len(STOP_MULTS)):
                    row[f"cost_s{STOP_MULTS[si]:g}"] = COST / (STOP_MULTS[si] * avg[now - 1])
                    for ti in range(len(TARGETS)):
                        for sd in range(2):
                            k = key(si, ti, sd)
                            row[f"est_{k}"] = Ra[:, si, ti, sd].mean()          # tahmini ortalama R (masrafsız)
                            row[f"phit_{k}"] = (Ka[:, si, ti, sd] == 1).mean()  # analoglarda hedefe ulaşma oranı
                            row[f"real_{k}"] = Rn[0, si, ti, sd]                # gerçek sonuç (masrafsız)
                            row[f"kind_{k}"] = Kn[0, si, ti, sd]
                            row[f"bars_{k}"] = Bn[0, si, ti, sd]
                rows.append(row)

        if n % 100 == 0 or n == len(points):
            el = time.perf_counter() - t0
            print(f"{period}: {n}/{len(points)} karar anı, {len(rows)} kayıt, "
                  f"{el / 60:.1f} dk, kalan ~{el / n * (len(points) - n) / 60:.0f} dk", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    print(f"bitti: {out}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "dev")
