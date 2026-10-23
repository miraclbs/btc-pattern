"""2. aşama: HTF bağlamı sonuçları değiştiriyor mu, ve bağlamın içinde analoglar bir şey ekliyor mu?

simulate.py kayıtlarına context.py etiketlerini ekler (sadece o ana kadar kapanmış mumlarla).
Kullanım: .venv/Scripts/python analyze_context.py [dev|val]

A. Her bağlamda körü körüne long / short (yeni arama yok, mevcut kayıtlar).
B. Basit trend takibi kuralları: trend yukarıysa long, aşağıysa short.
C. Aynı kural içinde analog tahmini iyi olanlar ile kötü olanlar arasında fark var mı?
"""
import sys

import numpy as np
import pandas as pd

import context
from analyze_sim import block_ci, config_frame, equity
from matcher import load
from simulate import INTERVAL, RESULTS

CONFIGS = [(6.0, 2.0), (3.0, 2.0), (6.0, 3.0)]
LABELS = ["g_trend", "s4_trend", "sezon", "konum", "momentum", "seviye"]


def with_context(period: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / f"{INTERVAL}_{period}.csv", parse_dates=["time"])
    m = load(INTERVAL)
    idx = df.idx.to_numpy()
    ts = m.open_time.to_numpy()[idx] // 1000                  # karar anı: bu mumun açılışı (unix sn)
    ctx = context.features(ts, m.close.to_numpy()[idx - 1])   # fiyat: bir önceki mumun kapanışı
    return pd.concat([df, ctx.drop(columns=["ts", "price"])], axis=1)


def fmt(r: pd.Series, months: pd.Series) -> str:
    lo, hi = block_ci(r, months)
    return f"{r.mean():+.3f} [{lo:+.2f},{hi:+.2f}]"


def rule_report(name: str, f: pd.DataFrame, side: pd.Series, years) -> None:
    """side: her karar anı için 'long' / 'short' / None. Seçilen yöndeki gerçek sonuçlar."""
    real = np.where(side == "long", f.real_long, np.where(side == "short", f.real_short, np.nan))
    bars = np.where(side == "long", f.bars_long, f.bars_short)
    g = pd.DataFrame({"real": real, "month": f.month, "year": f.year}).dropna()
    by_year = "  ".join(f"{y}:{g.real[g.year == y].mean():+.2f}" for y in years if (g.year == y).sum() >= 10)
    eq = equity(f.assign(real=np.nan_to_num(real), bars=bars), side.notna() & pd.notna(real))
    print(f"  {name:<34} {len(g):>5} işlem  ort {fmt(g.real, g.month)}  | {by_year}  | tek poz. {eq}")


def main(period: str):
    df = with_context(period)
    years = sorted(df.time.dt.year.unique())
    print(f"{INTERVAL} {period}: {len(df)} karar anı\n")
    for label in LABELS:
        print(f"{label}: " + ", ".join(f"{k} {v}" for k, v in df[label].value_counts(dropna=False).items()))
    print()

    for s, t in CONFIGS:
        f = config_frame(df, s, t)
        print(f"=== stop {s:g}x, hedef {t:g}R ===")
        print("A. Bağlama göre körü körüne işlem (ort R [%90 aralık]):")
        for label in LABELS:
            for cat, idx in df.groupby(label).groups.items():
                g = f.loc[idx]
                if len(g) < 100:
                    continue
                print(f"  {label:<9} {cat:<14} {len(g):>5}  long {fmt(g.real_long, g.month)}   "
                      f"short {fmt(g.real_short, g.month)}")
        print("B. Trend takibi kuralları:")
        for col in ("g_trend", "s4_trend"):
            side = df[col].map({"yukarı": "long", "aşağı": "short"})
            rule_report(f"{col}: yukarı->long, aşağı->short", f, side, years)
        both = df.g_trend.where(df.g_trend == df.s4_trend).map({"yukarı": "long", "aşağı": "short"})
        rule_report("günlük ve 4S aynı yönde", f, both, years)
        print("C. Trend kuralının içinde analoglar (günlük trend yönünde, analog tahmini o yön için):")
        side = df.g_trend.map({"yukarı": "long", "aşağı": "short"})
        est = pd.Series(np.where(side == "long", f.est_long, np.where(side == "short", f.est_short, np.nan)))
        for lo, hi, name in ((-9, 0.0, "analog tahmini < 0"), (0.0, 0.2, "0 - 0.2R"), (0.2, 9, ">= 0.2R")):
            rule_report(f"analog {name}", f, side.where((est >= lo) & (est < hi)), years)
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
