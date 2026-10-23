"""Onaylı geri test: kırılan 4S seviyesinin geri testinde, 15dk'da dönüş (yapı kırılımı) görülünce giriş.

Short örneği (kırılan destek, aşağıdan geri test); long tarafı aynadaki görüntüsü:
1. Geri test: kırılımdan en az 4 saat, en fazla 7 gün sonra fiyat seviyeye yaklaşır
   (seviyenin TOUCH_TOL x 4S normal mum boyu kadar altına gelmesi yeterli).
2. Onay ("fiyat döndü"): geri testte yapılan tepeden önceki son K mumun en düşüğünün altında
   bir 15dk kapanışı (yapı kırılımı).
3. Giriş: bir sonraki 15dk mumun açılışı. Stop: geri test tepesinin STOP_BUFFER x 4S mum boyu üstü
   (yapısal stop). Stop mesafesi MIN_STOP'tan darsa genişletilir, MAX_STOP'tan genişse işlem atlanır.
4. İptal: fiyat onaydan önce seviyenin INVALID x 4S mum boyu ötesine geçerse ya da geri testten
   sonra CONFIRM_BARS mum içinde onay gelmezse.

Kullanım: .venv/Scripts/python confirm_sim.py [dev|val] [seviye_hassasiyeti=3]
"""
import sys
import time

import numpy as np
import pandas as pd

import context
import engine
from setup_sim import (HORIZON, RETEST_DAYS, RETEST_GAP, TARGETS, H4Info, level_breaks, summary)
from simulate import COST, PERIODS, RESULTS, TOP_K, _outcomes, _trade

TOUCH_TOL = 0.25      # seviyeye bu kadar (4S mum boyu) yaklaşmak geri test sayılır
STOP_BUFFER = 0.1     # stop, geri test ucunun bu kadar (4S mum boyu) ötesinde
MIN_STOP, MAX_STOP = 0.4, 2.5     # stop mesafesi sınırları (4S mum boyu)
INVALID = 1.0         # onaydan önce fiyat seviyenin bu kadar ötesine geçerse kurulum iptal
CONFIRM_BARS = 96     # geri testten sonra onay için en fazla bekleme (15dk'da 24 saat)
SWING_KS = (8, 16)    # "dönüşten önceki son dip/tepe" için geriye bakılan mum (2 saat / 4 saat)


def confirm(eng: engine.Engine, p: float, side: int, break_ts: int, a4: float, k: int):
    """Onaylı girişi arar: (giriş mumu, geri test ucu, geri testin başladığı mum) ya da None.
    Her adımda sadece o mumun kapanışına kadarki veri kullanılır."""
    h, l, c = eng.h, eng.l, eng.c
    tol = TOUCH_TOL * a4 * p
    i0, i1 = np.searchsorted(eng.t, [break_ts + RETEST_GAP, break_ts + RETEST_DAYS * 86400])
    touch = np.flatnonzero(h[i0:i1] >= p - tol) if side == -1 else np.flatnonzero(l[i0:i1] <= p + tol)
    if not len(touch):
        return None
    t0 = int(i0 + touch[0])
    ext = t0
    for j in range(t0, min(t0 + CONFIRM_BARS, len(c) - 1)):
        if side == -1:                                   # short: geri test tepesi, sonra aşağı kırılım
            if h[j] >= h[ext]:
                ext = j
            if h[ext] > p * (1 + INVALID * a4):
                return None
            if j > ext and c[j] < l[max(0, ext - k):ext].min():
                return j + 1, h[ext], t0
        else:                                            # long: geri test dibi, sonra yukarı kırılım
            if l[j] <= l[ext]:
                ext = j
            if l[ext] < p * (1 - INVALID * a4):
                return None
            if j > ext and c[j] > h[max(0, ext - k):ext].max():
                return j + 1, l[ext], t0
    return None


def run(period: str, level_mult: float) -> pd.DataFrame:
    eng = engine.Engine("15m")
    o, h, l, c, avg = eng.o, eng.h, eng.l, eng.c, eng.avg_range
    h4 = H4Info()
    a_ts, b_ts = (int(pd.Timestamp(x).timestamp()) for x in PERIODS[period])
    breaks = level_breaks(h4, level_mult)
    rows, skipped, t0 = [], {"onay yok/iptal": 0, "stop çok geniş": 0}, time.perf_counter()
    for n, (p, side, break_ts) in enumerate(breaks, 1):
        if not a_ts - RETEST_DAYS * 86400 <= break_ts < b_ts:
            continue
        a4 = h4.avg_at(break_ts)
        for k in SWING_KS:
            found = confirm(eng, p, side, break_ts, a4, k)
            if found is None:
                skipped["onay yok/iptal"] += 1
                continue
            i, ext_price, touch_i = found
            if not a_ts <= eng.t[i] < b_ts:
                continue
            stop = ext_price * (1 - side * STOP_BUFFER * a4)
            risk_frac = side * (o[i] - stop) / o[i]
            if risk_frac > MAX_STOP * a4 or risk_frac <= 0:
                skipped["stop çok geniş"] += 1
                continue
            risk_frac = max(risk_frac, MIN_STOP * a4)
            cost_r = COST / risk_frac
            sd = 0 if side == 1 else 1
            row = {"time": pd.Timestamp(int(eng.t[i]), unit="s"), "idx": i, "side": "long" if side == 1 else "short",
                   "level": p, "K": k, "a4": a4, "stop_a4": risk_frac / a4, "stop_pct": 100 * risk_frac,
                   "wait_bars": i - touch_i, "cost": cost_r}
            groups = eng.groups(i, eng.pivots(i))
            ends = np.empty(0, np.int64)
            if groups:
                d, s, m = eng.search(groups[0]["start_idx"], i, True, end_limit=i - HORIZON)
                picks = engine.distinct_events(d, s, m, TOP_K)
                ends = (s[picks] + m[picks]).astype(np.int64)
            Ra = (_outcomes(o, h, l, c, avg, ends, np.array([risk_frac / avg[i - 1]]), TARGETS, HORIZON)[0]
                  if len(ends) else None)
            for ti, t in enumerate(TARGETS):
                r, kind, bars = _trade(o, h, l, c, i, risk_frac, t, HORIZON, side)
                key = f"t{t:g}"
                row[f"real_{key}"], row[f"kind_{key}"], row[f"bars_{key}"] = r - cost_r, kind, bars
                row[f"est_{key}"] = Ra[:, 0, ti, sd].mean() - cost_r if Ra is not None else np.nan
            rows.append(row)
        if n % 100 == 0:
            print(f"  {n}/{len(breaks)} seviye, {len(rows)} işlem, {time.perf_counter() - t0:.0f} sn", flush=True)
    print(f"atlananlar: {skipped}")
    df = pd.DataFrame(rows)
    ctx = context.features(eng.t[df.idx.to_numpy()], o[df.idx.to_numpy()])
    df = pd.concat([df, ctx.drop(columns=["ts", "price"])], axis=1)
    df.to_csv(RESULTS / f"confirm_15m_{period}_m{level_mult:g}.csv", index=False)
    return df


def report(df: pd.DataFrame) -> None:
    df = df.sort_values("time").reset_index(drop=True)
    years = sorted(df.time.dt.year.unique())
    for k in SWING_KS:
        g = df[df.K == k]
        trend_side = g.g_trend.map({"yukarı": "long", "aşağı": "short"})
        print(f"\n##### onay: son {k} mumun ({k / 4:g} saat) dibi/tepesi kırılınca  -  {len(g)} işlem, "
              f"stop ortalama %{g.stop_pct.mean():.2f} ({g.stop_a4.mean():.2f}x 4S mum), masraf ort. {g.cost.mean():.2f}R, "
              f"onay için ort. bekleme {g.wait_bars.mean() / 4:.1f} saat")
        for t in TARGETS:
            key = f"t{t:g}"
            print(f"=== hedef {t:g}R ===")
            summary("tüm kurulumlar", g, key, years)
            summary("long (direnç kırılımı)", g[g.side == "long"], key, years)
            summary("short (destek kırılımı)", g[g.side == "short"], key, years)
            summary("günlük trend yönünde", g[g.side == trend_side], key, years)
            summary("günlük trende karşı", g[trend_side.notna() & (g.side != trend_side)], key, years)
            summary("analog tahmini >= 0", g[g[f"est_{key}"] >= 0], key, years)
            summary("trend yönünde + analog >= 0", g[(g.side == trend_side) & (g[f"est_{key}"] >= 0)], key, years)


if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "dev"
    mult = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    report(run(period, mult))
