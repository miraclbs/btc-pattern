"""İtki + FVG + iç likidite süpürmesi testi (kullanıcının yöntemi, long için; short aynasıdır).

Her karar anında (rank_sim.py havuzundaki anlar), arayüzdeki 1. gruptaki yapı (son tepe/dipten şu ana):
1. Yapı bir dipten başlıyorsa long (tepeden başlıyorsa short) aranır. İtkinin ilk FVG'si en az MIN_FVG
   normal mum büyüklüğünde olmalı.
2. Likidite: FVG'den sonra yapının içinde oluşan dipler (5 mumluk fraktal, karar anında onaylı); en
   düşüğü. Karar anına kadar alınmamış olmalı.
3. Giriş: karar anından sonra en fazla WAIT mum içinde o seviyeyi süpüren mum, seviyenin üstünde kapanırsa
   kapanışta. Önce karşı taraf (yapının tepesi) alınırsa ya da süpüren mum stopu da alırsa kurulum iptal.
4. Stop: FVG dizisinin ilk mumunun dibi.
5. Hedef: (a) karşı taraf likiditesi (yapının tepesi), (b) girişin üstündeki ilk iç tepe,
   (c) uyarlamalı: aynı kurulumu veren benzer olaylarda (DTW ilk 50) hangisi ortalama daha iyiyse o.
En fazla HOLD mum; aynı mumda stop ve hedef görülürse stop. Sonuç masraflı R.

Bu bir keşif testi: kural seçildikten sonra val döneminde bir kez denenmeden geçerli sayılmaz.

Kullanım: .venv/Scripts/python fvg_sim.py [dev|val]
"""
import sys

import numpy as np
import pandas as pd
from numba import njit

import engine
from analyze_sim import block_ci
from range_sim import align, trends
from simulate import COST, INTERVAL, RESULTS

MIN_FVG = 0.5             # FVG en az bu kadar normal mum büyüklüğünde
WAIT = 192                # süpürme için en fazla bekleme (15dk'da 2 gün)
HOLD = 192                # işlem en fazla bu kadar mum açık kalır (2 gün)
ANALOG_GAP = WAIT + HOLD + 1     # analoğun işlemi karar anından önce bitsin diye
TOP_K = 50
MIN_ANALOG_TRADES = 3


@njit(cache=True)
def _run(h, l, c, i, entry, stop, target, side, hold):
    """i. mumdan itibaren işlem: (masrafsız R, hedefe ulaştı mı)."""
    risk = side * (entry - stop)
    end = min(i + hold, len(c))
    for t in range(i, end):
        if (side == 1 and l[t] <= stop) or (side == -1 and h[t] >= stop):
            return -1.0, False
        if (side == 1 and h[t] >= target) or (side == -1 and l[t] <= target):
            return side * (target - entry) / risk, True
    return side * (c[end - 1] - entry) / risk, False


@njit(cache=True)
def _swing(h, l, k, side):
    """k. mum 5 mumluk fraktal dip (side 1) / tepe (side -1) mi."""
    if side == 1:
        return l[k] < l[k - 1] and l[k] < l[k - 2] and l[k] <= l[k + 1] and l[k] <= l[k + 2]
    return h[k] > h[k - 1] and h[k] > h[k - 2] and h[k] >= h[k + 1] and h[k] >= h[k + 2]


@njit(cache=True)
def _setup(h, l, c, avg, s, e, cost, wait, hold):
    """[s, e) yapısı için kurulum ve iki hedefle sonuç:
    (yön, giriş mumu, R karşı taraf, R ilk iç seviye, hedef R karşı, hedef R ilk, kazandı karşı, kazandı ilk).
    Kurulum yoksa yön 0."""
    n = len(c)
    none = (0, -1, np.nan, np.nan, np.nan, np.nan, False, False)
    hi, lo = -np.inf, np.inf
    for t in range(s, e):
        hi = max(hi, h[t])
        lo = min(lo, l[t])
    if l[s] == lo:
        side = 1
    elif h[s] == hi:
        side = -1
    else:
        return none
    # 1. İtkinin ilk FVG'si; stop, dizinin ilk mumunun dibi (short'ta tepesi).
    f = -1
    for i in range(s + 1, e - 1):
        gap = l[i + 1] - h[i - 1] if side == 1 else l[i - 1] - h[i + 1]
        if gap >= MIN_FVG * avg[i] * c[i]:
            f = i
            break
    if f < 0:
        return none
    stop = l[f - 1] if side == 1 else h[f - 1]
    # 2. FVG'den sonra oluşan iç dipler (short'ta tepeler); en uçtaki likidite seviyesi.
    level, kl = np.nan, -1
    for k in range(f + 2, e - 2):
        if _swing(h, l, k, side):
            p = l[k] if side == 1 else h[k]
            if side * (p - stop) > 0 and (kl < 0 or side * (p - level) < 0):
                level, kl = p, k
    if kl < 0:
        return none
    for t in range(kl + 1, e):
        if (side == 1 and l[t] < level) or (side == -1 and h[t] > level):
            return none                                   # likidite karar anından önce alınmış
    # 3. Süpürme ve giriş.
    far = hi if side == 1 else lo
    j = -1
    for t in range(e, min(e + wait, n - 1)):
        if (side == 1 and h[t] > far) or (side == -1 and l[t] < far):
            return none                                   # önce karşı taraf alındı
        if (side == 1 and l[t] < level) or (side == -1 and h[t] > level):
            j = t
            break
    if j < 0:
        return none
    if (side == 1 and l[j] <= stop) or (side == -1 and h[j] >= stop):
        return none                                       # süpüren mum stopu da aldı
    entry = c[j]
    if side * (entry - level) <= 0:
        return none                                       # seviyenin ötesinde kapandı: kırılım
    risk = side * (entry - stop)
    if risk <= 0 or side * (far - entry) <= 0:
        return none
    # 4. Hedefler: karşı uç ve girişin ötesindeki ilk iç tepe (short'ta dip).
    near = far
    for k in range(f + 2, e - 2):
        if _swing(h, l, k, -side):
            p = h[k] if side == 1 else l[k]
            if side * (p - entry) > 0 and side * (p - near) < 0:
                near = p
    fee = cost * entry / risk
    r1, w1 = _run(h, l, c, j + 1, entry, stop, far, side, hold)
    r2, w2 = _run(h, l, c, j + 1, entry, stop, near, side, hold)
    return side, j, r1 - fee, r2 - fee, side * (far - entry) / risk, side * (near - entry) / risk, w1, w2


@njit(cache=True)
def _many(h, l, c, avg, starts, ends, cost, wait, hold):
    n = len(starts)
    side, entry = np.zeros(n, np.int64), np.full(n, -1, np.int64)
    r1, r2, t1, t2 = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    w1, w2 = np.zeros(n, np.bool_), np.zeros(n, np.bool_)
    for a in range(n):
        side[a], entry[a], r1[a], r2[a], t1[a], t2[a], w1[a], w2[a] = _setup(
            h, l, c, avg, starts[a], ends[a], cost, wait, hold)
    return side, entry, r1, r2, t1, t2, w1, w2


def summary(name: str, R: np.ndarray, win: np.ndarray, target_r: np.ndarray, month, year) -> dict:
    r = pd.Series(R)
    lo, hi = block_ci(r, pd.Series(month))
    by_year = r.groupby(np.asarray(year)).mean()
    return {"hedef": name, "işlem": len(r), "hedefe ulaşan %": round(100 * win.mean(), 1),
            "ort. hedef R": round(float(np.median(target_r)), 2), "ort. R": round(r.mean(), 3),
            "%90 aralık": f"[{lo:+.3f}, {hi:+.3f}]", "artı yıl": f"{(by_year > 0).sum()}/{len(by_year)}"}


def main(period: str):
    eng = engine.Engine(INTERVAL)
    h, l, c, avg = eng.h, eng.l, eng.c, eng.avg_range
    z = np.load(RESULTS / f"pool_{INTERVAL}_{period}.npz")
    now, q_start = z["now"], z["q_start"]
    side, entry, r1, r2, t1, t2, w1, w2 = _many(h, l, c, avg, q_start, now, COST, WAIT, HOLD)
    ok = side != 0
    _, first = np.unique(entry[ok], return_index=True)     # aynı giriş birden çok karar anından gelebilir
    idx = np.flatnonzero(ok)[np.sort(first)]
    print(f"{INTERVAL} {period}: {len(now)} karar anı, {len(idx)} kurulumda işlem açıldı "
          f"(long {np.sum(side[idx] == 1)}, short {np.sum(side[idx] == -1)})\n")

    # Uyarlamalı hedef: aynı kurulumu veren benzer olaylarda hangi hedef ortalama daha iyiydi?
    a_s, a_m = z["dtw_start"][idx].astype(np.int64), z["dtw_length"][idx].astype(np.int64)
    a_e = a_s + a_m
    usable = (a_s >= 0) & (a_e <= now[idx][:, None] - ANALOG_GAP)
    usable &= np.cumsum(usable, 1) <= TOP_K
    fs, fe = np.where(usable, a_s, 0).ravel(), np.where(usable, a_e, 1).ravel()
    a_side, _, a_r1, a_r2, *_ = _many(h, l, c, avg, fs, fe, COST, WAIT, HOLD)
    a_side, a_r1, a_r2 = (x.reshape(a_s.shape) for x in (a_side, a_r1, a_r2))
    took = usable & (a_side == side[idx][:, None])
    cnt = took.sum(1)
    m1 = np.where(took, a_r1, 0).sum(1) / np.maximum(cnt, 1)
    m2 = np.where(took, a_r2, 0).sum(1) / np.maximum(cnt, 1)
    use_far = np.where(cnt >= MIN_ANALOG_TRADES, m1 >= m2, True)
    print(f"benzer olaylardan en az {MIN_ANALOG_TRADES}'ü aynı kurulumu vermiş: {np.mean(cnt >= MIN_ANALOG_TRADES):.0%} "
          f"(ortalama {cnt.mean():.1f} tane); uyarlamalı hedef {np.mean(~use_far):.0%} işlemde ilk iç seviyeyi seçti\n")

    month = pd.to_datetime(eng.t[entry[idx]], unit="s").to_period("M")
    year = month.year
    R1, R2, T1, T2, W1, W2 = r1[idx], r2[idx], t1[idx], t2[idx], w1[idx], w2[idx]
    R3, T3, W3 = np.where(use_far, R1, R2), np.where(use_far, T1, T2), np.where(use_far, W1, W2)
    pd.set_option("display.width", 220)
    rows = [summary("(a) karşı taraf likiditesi", R1, W1, T1, month, year),
            summary("(b) ilk iç seviye", R2, W2, T2, month, year),
            summary("(c) benzer olaylara göre", R3, W3, T3, month, year)]
    print(pd.DataFrame(rows).to_string(index=False), "\n")

    lab = trends(eng.t[entry[idx]])
    sd = side[idx]
    for col in ("s4_trend", "g_trend"):
        a = align(sd, lab[col])
        rows = []
        for key in ("trend yönünde", "yatay", "trende karşı"):
            mk = a == key
            if mk.sum() >= 20:
                rows.append({"trend": key, **summary("(a)", R1[mk], W1[mk], T1[mk], month[mk], year[mk])})
        print(("4S trend" if col == "s4_trend" else "günlük trend") + ", hedef (a):")
        print(pd.DataFrame(rows).drop(columns="hedef").to_string(index=False), "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
