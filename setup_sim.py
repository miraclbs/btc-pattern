"""3. adım: kurulum bazlı test - kırılan 4S seviyesinin geri testi (destek kırılınca dirence döner).

Kurulum (hepsi sadece o ana kadarki veriyle):
1. Seviye: onaylanmış bir 4S tepe/dip (context.h4_levels).
2. Kırılım: onaydan sonraki BREAK_DAYS gün içinde bir 4S mumu seviyenin ötesinde kapanır.
   Dip (destek) kırılırsa short, tepe (direnç) kırılırsa long aranır.
3. Geri test: kırılımdan en az 4 saat, en fazla RETEST_DAYS gün sonra 15dk'da fiyat seviyeye
   geri dokunur; işleme tam seviyeden girilir. Her seviyede sadece ilk geri test.
4. Stop: seviyenin ötesinde, 4S normal mum boyunun k katı. Hedef: R katı. En fazla HORIZON mum.
   Dolum mumunda stop görülürse stop sayılır, hedef sayılmaz (temkinli).

Her kurulum için bağlam etiketleri ve analogların aynı işlem için tahmini de kaydedilir.
Kullanım: .venv/Scripts/python setup_sim.py [dev|val] [seviye_hassasiyeti=3]
"""
import sys
import time

import numpy as np
import pandas as pd
from numba import njit

import context
import engine
from analyze_sim import block_ci, equity
from matcher import load
from simulate import COST, PERIODS, RESULTS, TOP_K, _outcomes

BREAK_DAYS = 30           # seviye onaylandıktan sonra en fazla bu kadar gün içinde kırılmalı
RETEST_GAP = 4 * 3600     # kırılımdan sonra geri test için en az bekleme
RETEST_DAYS = 7           # kırılımdan sonra geri test için en fazla bekleme
STOP_KS = np.array([1.0, 2.0])           # stop: seviyenin ötesinde 4S normal mum boyunun bu katı
TARGETS = np.array([1.5, 2.0, 3.0])
HORIZON = 480                             # 15dk'da 5 gün
H4 = 4 * 3600


@njit(cache=True)
def _trade_from(h, l, c, i, entry, risk_frac, target_r, horizon, side):
    """i. mumda 'entry' fiyatından (limit emir) girilen işlem: (R, tür, süre)."""
    risk = risk_frac * entry
    stop = entry - side * risk
    target = entry + side * target_r * risk
    end = min(i + horizon, len(c))
    for j in range(i, end):
        if side == 1:
            if l[j] <= stop:
                return -1.0, -1, j - i + 1
            if j > i and h[j] >= target:
                return target_r, 1, j - i + 1
        else:
            if h[j] >= stop:
                return -1.0, -1, j - i + 1
            if j > i and l[j] <= target:
                return target_r, 1, j - i + 1
    return side * (c[end - 1] - entry) / risk, 0, end - i


def stop_entry_fill(o, i: int, p: float, side: int) -> float:
    """Seviyede bekleyen STOP emrinin gerçekçi dolum fiyatı (long: buy stop, short: sell stop).
    Mum seviyenin ötesinde açtıysa emir açılıştan (daha kötü fiyattan) dolar, seviyeden değil.
    (Limit emirde tersi geçerlidir: seviyeden dolum varsayımı temkinlidir.)"""
    if (side == 1 and o[i] > p) or (side == -1 and o[i] < p):
        return float(o[i])
    return p


class H4Info:
    """4S mumları ve 'o anki 4S normal mum boyu' (sadece kapanmış mumlarla)."""

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        d4 = load("4h", symbol)
        self.t, self.c = (d4.open_time // 1000).to_numpy(), d4.close.to_numpy()
        self.avg = pd.Series((d4.high - d4.low) / d4.close).rolling(180, min_periods=24).mean().to_numpy()

    def avg_at(self, ts: int) -> float:
        """ts anından önce kapanmış son 4S mumun normal mum boyu."""
        return float(self.avg[np.searchsorted(self.t, ts - H4, side="right") - 1])


def level_breaks(h4: H4Info, level_mult: float) -> list[tuple[float, int, int]]:
    """Kırılan 4S seviyeleri: (seviye, işlem yönü, kırılımın bilindiği an). Dip kırılırsa short (-1),
    tepe kırılırsa long (+1) aranır."""
    out = []
    for p, kind, _, confirmed in context.h4_levels(level_mult, h4.symbol).itertuples(index=False):
        j0, j1 = np.searchsorted(h4.t, [confirmed, confirmed + BREAK_DAYS * 86400])
        broke = np.flatnonzero(h4.c[j0:j1] < p) if kind == "dip" else np.flatnonzero(h4.c[j0:j1] > p)
        if len(broke):
            out.append((float(p), -1 if kind == "dip" else 1, int(h4.t[j0 + broke[0]]) + H4))
    return out


def find_setups(eng: engine.Engine, level_mult: float, a_ts: int, b_ts: int) -> list[dict]:
    """Geri testte seviyeden limit emirle (onaysız) giriş."""
    h4 = H4Info(eng.symbol)
    out = []
    for p, side, break_ts in level_breaks(h4, level_mult):
        i0, i1 = np.searchsorted(eng.t, [break_ts + RETEST_GAP, break_ts + RETEST_DAYS * 86400])
        touch = np.flatnonzero(eng.h[i0:i1] >= p) if side == -1 else np.flatnonzero(eng.l[i0:i1] <= p)
        if not len(touch):
            continue
        i = int(i0 + touch[0])
        if not a_ts <= eng.t[i] < b_ts:
            continue
        out.append({"i": i, "ts": int(eng.t[i]), "level": p, "side": side,
                    "break_ts": break_ts, "a4": h4.avg_at(int(eng.t[i]))})
    return out


def run(period: str, level_mult: float) -> pd.DataFrame:
    eng = engine.Engine("15m")
    o, h, l, c, avg = eng.o, eng.h, eng.l, eng.c, eng.avg_range
    a_ts, b_ts = (int(pd.Timestamp(x).timestamp()) for x in PERIODS[period])
    setups = find_setups(eng, level_mult, a_ts, b_ts)
    print(f"{len(setups)} kurulum bulundu, analoglar hesaplanıyor...", flush=True)
    rows, t0 = [], time.perf_counter()
    for n, st in enumerate(setups, 1):
        i, side = st["i"], st["side"]
        sd = 0 if side == 1 else 1
        row = {"time": pd.Timestamp(st["ts"], unit="s"), "idx": i, "side": "long" if side == 1 else "short",
               "level": st["level"], "a4": st["a4"]}
        groups = eng.groups(i, eng.pivots(i))
        picks = np.empty(0, np.int64)
        if groups:
            d, s, m = eng.search(groups[0]["start_idx"], i, True, end_limit=i - HORIZON)
            picks = engine.distinct_events(d, s, m, TOP_K)
        ends = (s[picks] + m[picks]).astype(np.int64) if len(picks) else np.empty(0, np.int64)
        row["n_analogs"] = len(ends)
        for k in STOP_KS:
            risk_frac = k * st["a4"]
            cost_r = COST / risk_frac
            row[f"cost_k{k:g}"] = cost_r
            # Analoglar: aynı stop mesafesi, kendi dönemlerinin oynaklığına çevrilerek.
            s_eq = np.array([risk_frac / avg[i - 1]])
            Ra = _outcomes(o, h, l, c, avg, ends, s_eq, TARGETS, HORIZON)[0] if len(ends) else None
            for ti, t in enumerate(TARGETS):
                key = f"k{k:g}_t{t:g}"
                r, kind, bars = _trade_from(h, l, c, i, st["level"], risk_frac, t, HORIZON, side)
                row[f"real_{key}"], row[f"kind_{key}"], row[f"bars_{key}"] = r - cost_r, kind, bars
                row[f"est_{key}"] = (Ra[:, 0, ti, sd].mean() - cost_r) if Ra is not None else np.nan
        rows.append(row)
        if n % 100 == 0:
            print(f"  {n}/{len(setups)}  {time.perf_counter() - t0:.0f} sn", flush=True)
    df = pd.DataFrame(rows)
    ctx = context.features(df.time.map(pd.Timestamp.timestamp).astype(np.int64).to_numpy(), df.level.to_numpy())
    df = pd.concat([df, ctx.drop(columns=["ts", "price"])], axis=1)
    df.to_csv(RESULTS / f"setups_15m_{period}_m{level_mult:g}.csv", index=False)
    return df


def summary(name: str, g: pd.DataFrame, key: str, years) -> None:
    if len(g) < 20:
        print(f"  {name:<30} {len(g):>4} işlem (az)")
        return
    month = g.time.dt.to_period("M")
    lo, hi = block_ci(g[f"real_{key}"], month)
    by_year = "  ".join(f"{y}:{g[f'real_{key}'][g.time.dt.year == y].mean():+.2f}"
                        for y in years if (g.time.dt.year == y).sum() >= 10)
    eq = equity(pd.DataFrame({"idx": g.idx, "bars": g[f"bars_{key}"], "real": g[f"real_{key}"]}),
                pd.Series(True, index=g.index))
    print(f"  {name:<30} {len(g):>4} işlem  kazanma %{100 * (g[f'kind_{key}'] == 1).mean():.0f}  "
          f"ort {g[f'real_{key}'].mean():+.3f}R [{lo:+.2f},{hi:+.2f}]  | {by_year}  | tek poz. {eq}")


def report(df: pd.DataFrame) -> None:
    df = df.sort_values("time").reset_index(drop=True)
    years = sorted(df.time.dt.year.unique())
    print(f"\n{len(df)} kurulum: long {(df.side == 'long').sum()}, short {(df.side == 'short').sum()}; "
          + ", ".join(f"{y}: {(df.time.dt.year == y).sum()}" for y in years))
    trend_side = df.g_trend.map({"yukarı": "long", "aşağı": "short"})
    for k in STOP_KS:
        for t in TARGETS:
            key = f"k{k:g}_t{t:g}"
            print(f"=== stop seviyenin {k:g}x 4S mum ötesinde, hedef {t:g}R (masraf ort. {df[f'cost_k{k:g}'].mean():.2f}R) ===")
            summary("tüm kurulumlar", df, key, years)
            summary("long (direnç kırılımı)", df[df.side == "long"], key, years)
            summary("short (destek kırılımı)", df[df.side == "short"], key, years)
            summary("günlük trend yönünde", df[df.side == trend_side], key, years)
            summary("günlük trende karşı", df[trend_side.notna() & (df.side != trend_side)], key, years)
            summary("analog tahmini >= 0", df[df[f"est_{key}"] >= 0], key, years)
            summary("analog tahmini < 0", df[df[f"est_{key}"] < 0], key, years)
            summary("trend yönünde + analog >= 0", df[(df.side == trend_side) & (df[f"est_{key}"] >= 0)], key, years)
            print()


if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "dev"
    mult = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    report(run(period, mult))
