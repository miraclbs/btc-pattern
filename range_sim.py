"""Range karşı taraf testi: yapının bir tarafı süpürülüp geri alınınca karşı tarafa işlem.

Her karar anında (rank_sim.py havuzundaki anlar), arayüzdeki 1. gruptaki yapı (son tepe/dipten şu ana):
1. Yapı range gibi bitmeli: son kapanış, yapının aralığında RANGE_POS arasında.
2. Süpürme: karar anından sonra en fazla min(3 x uzunluk, WAIT) mum içinde tepe ya da dip geçilir.
   Aynı mumda ikisi birden geçilirse atlanır.
3. Geri alma: süpürmenin başladığı mumdan itibaren en fazla RECLAIM mum içinde bir mum yapının içinde
   kapanır (kapanmazsa gerçek kırılım sayılır, işlem yok). Bir sonraki mumun açılışında karşı tarafa işlem.
4. Stop: süpürmenin uç noktasının 1 normal mum ötesi. Hedef: yapının karşı ucu. En fazla HOLD mum.
   Aynı mumda stop ve hedef görülürse stop sayılır. Sonuç masraflı R.

Bölümler: günlük ve 4S trende göre (işlem trend yönünde mi, tersine mi), seans, hafta sonu, yapının
döneme göre boyu. Analoglar: DTW havuzunda aynı yapıyı aynı taraftan süpürüp geri alan geçmiş olayların
ortalama sonucu; bütün analoglar ile sadece aynı günlük trenddekiler karşılaştırılır.

Bu bir keşif testi: çok sayıda bölüm yan yana bakıldığı için şansla iyi görünen olur. Buradan çıkan bir
kural ancak val döneminde (bir kez) tutarsa geçerli sayılır.

Kullanım: .venv/Scripts/python range_sim.py [dev|val]
"""
import sys

import numpy as np
import pandas as pd
from numba import njit

import context
import engine
from analyze_sim import block_ci
from simulate import COST, INTERVAL, RESULTS

RANGE_POS = (0.25, 0.75)
WAIT = 192                # süpürme için en fazla bekleme (15dk'da 2 gün)
RECLAIM = 8               # geri kapanış için en fazla bekleme (2 saat)
HOLD = 192                # işlem en fazla bu kadar mum açık kalır (2 gün)
ANALOG_GAP = WAIT + RECLAIM + HOLD + 1    # analoğun işlemi karar anından önce bitsin diye
MIN_ANALOGS = 5
SESSIONS = [(0, "Asya 00-07"), (7, "Avrupa 07-13"), (13, "ABD 13-20"), (20, "Gece 20-24")]   # UTC


@njit(cache=True)
def _sweep_trade(o, h, l, c, avg, s, e, cost):
    """[s, e) yapısı için süpür-geri al-karşı tarafa işlem: (yön, giriş indeksi, R, hedefe ulaştı mı).
    İşlem yoksa yön 0. Yön +1: dip süpürüldü, long; -1: tepe süpürüldü, short."""
    n = len(c)
    hi, lo = -np.inf, np.inf
    for t in range(s, e):
        hi = max(hi, h[t])
        lo = min(lo, l[t])
    j = -1
    for t in range(e, min(e + min(3 * (e - s), WAIT), n)):
        up, dn = h[t] > hi, l[t] < lo
        if up and dn:
            return 0, -1, np.nan, False
        if up or dn:
            j = t
            break
    if j < 0:
        return 0, -1, np.nan, False
    side = -1 if h[j] > hi else 1
    ext, k = (h[j] if side == -1 else l[j]), -1
    for t in range(j, min(j + RECLAIM, n - 1)):
        ext = max(ext, h[t]) if side == -1 else min(ext, l[t])
        if (side == -1 and c[t] < hi) or (side == 1 and c[t] > lo):
            k = t
            break
    if k < 0:
        return 0, -1, np.nan, False
    i = k + 1
    entry = o[i]
    stop = ext - side * avg[k] * entry
    target = lo if side == -1 else hi
    risk = side * (entry - stop)
    if risk <= 0 or side * (target - entry) <= 0:       # açılış stopun ya da hedefin ötesinde
        return 0, -1, np.nan, False
    fee = cost * entry / risk
    for t in range(i, min(i + HOLD, n)):
        if (side == 1 and l[t] <= stop) or (side == -1 and h[t] >= stop):
            return side, i, -1.0 - fee, False
        if (side == 1 and h[t] >= target) or (side == -1 and l[t] <= target):
            return side, i, side * (target - entry) / risk - fee, True
    end = min(i + HOLD, n) - 1
    return side, i, side * (c[end] - entry) / risk - fee, False


@njit(cache=True)
def _many(o, h, l, c, avg, starts, ends, cost):
    n = len(starts)
    side, entry, R, win = np.zeros(n, np.int64), np.full(n, -1, np.int64), np.full(n, np.nan), np.zeros(n, np.bool_)
    for a in range(n):
        side[a], entry[a], R[a], win[a] = _sweep_trade(o, h, l, c, avg, starts[a], ends[a], cost)
    return side, entry, R, win


def trends(ts: np.ndarray) -> pd.DataFrame:
    """ts anlarında günlük ve 4S trend (sadece o ana kadar kapanmış mumlarla, context.py)."""
    left = pd.DataFrame({"ts": ts.astype(np.int64), "_o": np.arange(len(ts))}).sort_values("ts")
    for right in (context.daily_features()[["avail", "g_trend"]], context.h4_features()):
        left = pd.merge_asof(left, right.sort_values("avail"), left_on="ts", right_on="avail",
                             direction="backward").drop(columns="avail")
    return left.sort_values("_o").reset_index(drop=True)


def align(side: np.ndarray, trend: pd.Series) -> np.ndarray:
    up = np.where(side == 1, "trend yönünde", "trende karşı")
    down = np.where(side == 1, "trende karşı", "trend yönünde")
    return np.where(trend == "yukarı", up, np.where(trend == "aşağı", down, "yatay"))


def table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, sort=False):
        lo, hi = block_ci(g.R, g.month)
        rows.append({by: key, "işlem": len(g), "hedef %": round(100 * g.win.mean(), 1),
                     "ort. hedef R": round(g.target_r.median(), 1), "ort. R": round(g.R.mean(), 3),
                     "%90 aralık": f"[{lo:+.3f}, {hi:+.3f}]"})
    return pd.DataFrame(rows)


def main(period: str):
    eng = engine.Engine(INTERVAL)
    o, h, l, c, avg = eng.o, eng.h, eng.l, eng.c, eng.avg_range
    z = np.load(RESULTS / f"pool_{INTERVAL}_{period}.npz")
    now, q_start = z["now"], z["q_start"]
    hi = np.array([h[s:e].max() for s, e in zip(q_start, now)])
    lo = np.array([l[s:e].min() for s, e in zip(q_start, now)])
    pos = (c[now - 1] - lo) / (hi - lo)
    keep = (pos >= RANGE_POS[0]) & (pos <= RANGE_POS[1])
    print(f"{INTERVAL} {period}: {len(now)} karar anı, {keep.sum()} tanesinde yapı range gibi bitiyor")

    side, entry, R, win = _many(o, h, l, c, avg, q_start, now, COST)
    ok = keep & (side != 0)
    # Aynı süpürmeye birden çok karar anı denk gelebilir: her giriş mumu bir kez sayılır.
    _, first = np.unique(entry[ok], return_index=True)
    idx = np.flatnonzero(ok)[np.sort(first)]
    print(f"süpürülüp geri alınan ve işlem açılan: {len(idx)} (aynı girişler tekilleştirildi)\n")

    e = entry[idx]
    ts = eng.t[e]
    lab = trends(ts)
    sd = side[idx]
    rel = np.array([engine._normalize(_feat(eng, s, m), m, avg[s + m - 1])
                    for s, m in zip(q_start[idx], now[idx] - q_start[idx])])
    utc = pd.to_datetime(ts, unit="s")
    df = pd.DataFrame({
        "R": R[idx], "win": win[idx], "month": utc.to_period("M"),
        "yön": np.where(sd == 1, "long (dip süpürüldü)", "short (tepe süpürüldü)"),
        "günlük trend": align(sd, lab.g_trend), "4S trend": align(sd, lab.s4_trend),
        "seans": pd.cut(utc.hour, [0, 7, 13, 20, 24], right=False, labels=[n for _, n in SESSIONS]).astype(str),
        "gün": np.where(utc.dayofweek >= 5, "hafta sonu", "hafta içi"),
        "boy (döneme göre)": pd.qcut(rel, 3, labels=["küçük", "orta", "büyük"]).astype(str),
    })
    df["target_r"] = _target_r(o, h, l, avg, q_start[idx], now[idx], e, sd)

    pd.set_option("display.width", 200)
    lo_ci, hi_ci = block_ci(df.R, df.month)
    print(f"HEPSİ: {len(df)} işlem, hedefe ulaşan %{100 * df.win.mean():.1f}, ortalama hedef "
          f"{df.target_r.median():.1f}R, ortalama sonuç {df.R.mean():+.3f}R [{lo_ci:+.3f}, {hi_ci:+.3f}]\n")
    for by in ("yön", "günlük trend", "4S trend", "seans", "gün", "boy (döneme göre)"):
        print(table(df, by).to_string(index=False), "\n")
    both = df.assign(**{"günlük + 4S": df["günlük trend"] + " / " + df["4S trend"]})
    print(table(both, "günlük + 4S").sort_values("işlem", ascending=False).to_string(index=False), "\n")

    # Analoglar: aynı yapıyı aynı taraftan süpürüp geri alan geçmiş olaylar ne demiş?
    a_s, a_m = z["dtw_start"][idx], z["dtw_length"][idx]
    a_end = a_s + a_m
    usable = (a_s >= 0) & (a_end <= now[idx][:, None] - ANALOG_GAP)
    flat_s, flat_e = np.where(usable, a_s, 0).ravel(), np.where(usable, a_end, 1).ravel()
    a_side, _, a_R, _ = _many(o, h, l, c, avg, flat_s.astype(np.int64), flat_e.astype(np.int64), COST)
    a_side, a_R = a_side.reshape(a_s.shape), a_R.reshape(a_s.shape)
    q_tr = trends(eng.t[now[idx] - 1]).g_trend.to_numpy()
    a_tr = trends(eng.t[np.maximum(flat_e - 1, 0)]).g_trend.to_numpy().reshape(a_s.shape)
    rank_in_pool = np.cumsum(usable, 1)
    print("Analogların tahmini (aynı taraftan işlem açan analogların ortalama R'si) ile gerçek sonuç:")
    for name, extra in (("ilk 50 analog", rank_in_pool <= 50),
                        ("aynı günlük trenddeki ilk 50 analog", (a_tr == q_tr[:, None]) & (np.cumsum(usable & (a_tr == q_tr[:, None]), 1) <= 50))):
        m = usable & extra & (a_side == sd[:, None])
        cnt = m.sum(1)
        est = np.where(cnt >= MIN_ANALOGS, np.nansum(np.where(m, a_R, 0), 1) / np.maximum(cnt, 1), np.nan)
        f = df.assign(est=est).dropna(subset=["est"])
        rho = f.est.rank().corr(f.R.rank())
        pos_ = f[f.est > 0]
        neg_ = f[f.est <= 0]
        print(f"  {name:<36} {len(f)} işlem, sıra korelasyonu {rho:+.3f} | analog > 0 dediğinde "
              f"{len(pos_)} işlem ort. {pos_.R.mean():+.3f}R, ≤ 0 dediğinde {len(neg_)} işlem ort. {neg_.R.mean():+.3f}R")
    return df


def _feat(eng, s, m):
    x = np.empty((engine.K, 3))
    engine._features(eng.h, eng.l, eng.c, s, m, x)
    return x


def _target_r(o, h, l, avg, starts, ends, entries, sides):
    """Her işlemin hedefe ulaşırsa kazanacağı R (masrafsız): karşı uç / stop mesafesi."""
    out = np.empty(len(starts))
    for a, (s, e, i, sd) in enumerate(zip(starts, ends, entries, sides)):
        hi, lo = h[s:e].max(), l[s:e].min()
        # Süpürmenin uç noktası: yapı bittikten sonra girişe kadar süpürülen taraftaki en uç fiyat
        # (_sweep_trade ile aynı, çünkü süpürmeden önce yapının dışına çıkılmamıştır).
        ext = h[e:i].max() if sd == -1 else l[e:i].min()
        stop = ext - sd * avg[i - 1] * o[i]
        out[a] = sd * ((lo if sd == -1 else hi) - o[i]) / (sd * (o[i] - stop))
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
