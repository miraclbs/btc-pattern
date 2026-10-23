"""Trend takibi testi: günlük trend yukarıysa long, aşağıysa short, yataysa işlem yok.

Kurallar 2018-2023 (geliştirme) verisinde seçilir; 2024 (kontrol) sadece bir kez doğrulama içindir.
- 15m: 1. aşamadan DONDURULMUŞ kural (stop 6x normal mum, hedef 3R, 3 gün, 8 saatte bir karar).
- 4h : birkaç ayar geliştirme döneminde denenir; 2024'te sadece en yüksek ortalama R'yi veren test edilir
       (seçim kuralı sonuçlar görülmeden önce konuldu).

Kullanım: .venv/Scripts/python trend_sim.py <15m|4h> <dev|val> [stop hedef]
"""
import sys

import numpy as np
import pandas as pd

import context
import engine
from analyze_sim import block_ci, equity
from simulate import COST, PERIODS, _trade

SETUPS = {
    "15m": {"step": 32, "horizon": 288, "configs": [(6.0, 3.0)]},                 # 8 saat, 3 gün
    "4h": {"step": 6, "horizon": 180, "configs": [(1.5, 2.0), (1.5, 3.0), (3.0, 2.0), (3.0, 3.0)]},  # 1 gün, 30 gün
}


def run(interval: str, period: str, configs=None) -> pd.DataFrame:
    cfg = SETUPS[interval]
    eng = engine.Engine(interval)
    o, h, l, c, avg = eng.o, eng.h, eng.l, eng.c, eng.avg_range
    a, b = np.searchsorted(eng.t, [int(pd.Timestamp(x).timestamp()) for x in PERIODS[period]])
    idx = np.arange(a, min(b, len(eng.t) - cfg["horizon"]), cfg["step"])
    ctx = context.features(eng.t[idx], c[idx - 1])          # sadece o ana kadar kapanmış mumlar
    side = ctx.g_trend.map({"yukarı": 1, "aşağı": -1}).to_numpy()
    time = pd.to_datetime(eng.t[idx], unit="s")
    print(f"{interval} {period}: {len(idx)} karar anı ({time[0]:%Y-%m-%d} -> {time[-1]:%Y-%m-%d}); "
          f"trend yukarı {np.sum(side == 1)}, aşağı {np.sum(side == -1)}, yatay {np.sum(np.isnan(side))}")
    out = []
    for s, t in configs or cfg["configs"]:
        rows = []
        for i, sd in zip(idx, side):
            if np.isnan(sd):
                continue
            frac = s * avg[i - 1]
            r, kind, bars = _trade(o, h, l, c, i, frac, t, cfg["horizon"], int(sd))
            rows.append({"time": pd.Timestamp(int(eng.t[i]), unit="s"), "idx": i, "side": int(sd),
                         "real": r - COST / frac, "win": kind == 1, "bars": bars, "stop_pct": 100 * frac})
        g = pd.DataFrame(rows)
        lo, hi = block_ci(g.real, g.time.dt.to_period("M"))
        years = "  ".join(f"{y}:{g.real[g.time.dt.year == y].mean():+.2f}" for y in sorted(g.time.dt.year.unique()))
        eq = equity(g, pd.Series(True, index=g.index))
        print(f"  stop {s:g}x (ort. %{g.stop_pct.mean():.1f}), hedef {t:g}R: {len(g)} işlem, kazanma %{100 * g.win.mean():.0f}, "
              f"ort {g.real.mean():+.3f}R [{lo:+.2f},{hi:+.2f}] | long {g.real[g.side == 1].mean():+.2f}, "
              f"short {g.real[g.side == -1].mean():+.2f} | {years} | tek poz. {eq}")
        out.append({"stop": s, "target": t, "mean_r": g.real.mean()})
    return pd.DataFrame(out)


if __name__ == "__main__":
    interval, period = sys.argv[1], sys.argv[2]
    configs = [(float(sys.argv[3]), float(sys.argv[4]))] if len(sys.argv) > 4 else None
    run(interval, period, configs)
