"""simulate.py kayıtlarını değerlendirir: analogların tahmini, masraflardan sonra kazandırıyor mu?

Kullanım: .venv/Scripts/python analyze_sim.py [dev|val]

Bölümler:
1. Temel oran: her karar anında körü körüne long / short açılsaydı (çıta).
2. Sistem: analogların daha iyi bulduğu yönde, tahmini beklenti eşiği geçince işlem.
3. Kalibrasyon: tahmini beklenti yükseldikçe gerçek sonuç da yükseliyor mu?
4. Aynı anda tek pozisyonla bakiye: toplam R, en büyük düşüş.
"""
import sys

import numpy as np
import pandas as pd

from simulate import INTERVAL, RESULTS, STOP_MULTS, TARGETS

THRESHOLDS = [None, 0.0, 0.1, 0.2, 0.3]     # None = her karar anında işlem
rng = np.random.default_rng(0)


def block_ci(r: pd.Series, months: pd.Series, n: int = 2000) -> tuple[float, float]:
    """Aylık blok bootstrap ile ortalama R için %90 güven aralığı (aynı haftadaki işlemler bağımsız değil)."""
    groups = [g.to_numpy() for _, g in r.groupby(months)]
    if len(groups) < 3:
        return np.nan, np.nan
    sums = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    idx = rng.integers(0, len(groups), size=(n, len(groups)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    return tuple(np.percentile(means, [5, 95]))


def config_frame(df: pd.DataFrame, s: float, t: float) -> pd.DataFrame:
    """Bir stop/hedef ayarı için masraf düşülmüş tahmin ve gerçek sonuçlar."""
    cost = df[f"cost_s{s:g}"]
    out = pd.DataFrame({"time": df.time, "month": df.time.dt.to_period("M"), "year": df.time.dt.year})
    for side in ("long", "short"):
        k = f"s{s:g}_t{t:g}_{side}"
        out[f"est_{side}"] = df[f"est_{k}"] - cost
        out[f"real_{side}"] = df[f"real_{k}"] - cost
        out[f"win_{side}"] = df[f"kind_{k}"] == 1
        out[f"bars_{side}"] = df[f"bars_{k}"]
    out["side"] = np.where(out.est_long >= out.est_short, "long", "short")
    pick = lambda col: np.where(out.side == "long", out[f"{col}_long"], out[f"{col}_short"])
    out["est"], out["real"], out["win"], out["bars"] = pick("est"), pick("real"), pick("win"), pick("bars")
    out["idx"] = df.idx
    out["cost"] = cost
    return out


def equity(f: pd.DataFrame, sel: pd.Series) -> dict:
    """Aynı anda tek pozisyon: işlem kapanmadan yeni işlem açılmaz."""
    free_at, rs = -1, []
    for i, b, r, ok in zip(f.idx, f.bars, f.real, sel):
        if ok and i >= free_at:
            rs.append(r)
            free_at = i + b
    rs = np.array(rs)
    if not len(rs):
        return {"n": 0, "toplam R": 0.0, "en büyük düşüş R": 0.0}
    curve = np.cumsum(rs)
    dd = (np.maximum.accumulate(np.r_[0, curve])[1:] - curve).max()
    return {"n": len(rs), "toplam R": round(float(curve[-1]), 1), "en büyük düşüş R": round(float(dd), 1)}


def main(period: str):
    df = pd.read_csv(RESULTS / f"{INTERVAL}_{period}.csv", parse_dates=["time"])
    years = sorted(df.time.dt.year.unique())
    print(f"{INTERVAL} {period}: {len(df)} karar anı, {df.time.min():%Y-%m-%d} -> {df.time.max():%Y-%m-%d}, "
          f"ort. analog benzerliği %{df.similarity.mean():.1f}\n")
    for s in STOP_MULTS:
        for t in TARGETS:
            f = config_frame(df, s, t)
            print(f"=== stop {s:g}x normal mum, hedef {t:g}R  (masraf ort. {df[f'cost_s{s:g}'].mean():.2f}R) ===")
            base_l, base_s = f.real_long.mean(), f.real_short.mean()
            print(f"  temel oran: her seferinde long {base_l:+.3f}R, her seferinde short {base_s:+.3f}R")
            rows = []
            for th in THRESHOLDS:
                sel = pd.Series(True, index=f.index) if th is None else f.est >= th
                g = f[sel]
                if len(g) < 30:
                    continue
                lo, hi = block_ci(g.real, g.month)
                share_long = (g.side == "long").mean()
                base_mix = share_long * base_l + (1 - share_long) * base_s
                row = {"eşik": "hepsi" if th is None else f">={th:.1f}R", "işlem": len(g),
                       "long %": round(100 * share_long), "kazanma %": round(100 * g.win.mean(), 1),
                       "ort R": round(g.real.mean(), 3), "%90 aralık": f"[{lo:+.3f}, {hi:+.3f}]",
                       "temel orana göre": round(g.real.mean() - base_mix, 3)}
                for y in years:
                    gy = g[g.year == y]
                    row[str(y)] = round(gy.real.mean(), 2) if len(gy) >= 10 else None
                rows.append(row)
            table = pd.DataFrame(rows)
            print(table.to_string(index=False))
            # Kalibrasyon: tahmini beklentiye göre beşe böl, gerçek ortalamaya bak. Sıralama masrafsız
            # tahminle yapılır: masraf karar anına göre değişip tahmine de gerçeğe de aynı eklendiği için
            # masraflı sıralama, analoglar hiçbir şey bilmese de sahte bir korelasyon gösterir.
            for side in ("long", "short"):
                raw = f[f"est_{side}"] + f.cost
                q = pd.qcut(raw, 5, labels=False, duplicates="drop")
                cal = f.groupby(q).agg(tahmin=(f"est_{side}", "mean"), gercek=(f"real_{side}", "mean"))
                rho = raw.rank().corr((f[f"real_{side}"] + f.cost).rank())   # Spearman, masrafsız
                print(f"  kalibrasyon {side:<5} (sıra korelasyonu {rho:+.3f}): " + "  ".join(
                    f"{a:+.2f}->{b:+.2f}" for a, b in zip(cal.tahmin, cal.gercek)))
            for th in (0.0, 0.2):
                print(f"  tek pozisyon, eşik >={th}R: {equity(f, f.est >= th)}")
            print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
