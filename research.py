"""Kuralı iyileştirme araştırması: çıkış yönetimi, rejim/trend, kırılım hacmi, RSI, daha çok kurulum.

!! DÜZELTME: results/research/ altındaki dev_setups.csv, holdout_setups.csv ve frozen.json, stop emrinin
her zaman seviyeden dolduğunu varsayan HATALI ilk sürümle üretildi. Gerçekçi dolumla (mum seviyenin
ötesinde açtıysa açılıştan) dondurulan ayarın avantajı kayboluyor. Kod düzeltildi; sonuçlar yeniden
üretilmeden o dosyalara güvenilmemeli.

Disiplin (sonuçlardan önce konuldu):
- Geliştirme coinleri: BTC, ETH, BNB, SOL. Saklı coinler (XRP, ADA, DOGE, LINK, AVAX) en sona kadar açılmaz.
- Yürüyen seçim: her yıl için en iyi ayar SADECE önceki yılların verisiyle seçilir (ölçüt: eğitim
  verisindeki toplam R, en az 40 işlem) ve o yıla uygulanır. Böylece optimizasyonun kendisi de test edilir.
- Son: tüm geliştirme verisiyle aynı ölçütle seçilen ayar dondurulur ve saklı coinlerde BİR KEZ test edilir.
  Karşılaştırma için şu anki basit kural (BASE) da aynı coinlerde test edilir.
  Short ailesi sadece yürüyen seçimde toplamı pozitifse portföye alınır.

Kurulum aileleri:
- long : kırılan 4S desteğinin geri testinde seviyeden long (şu anki kural).
- short: kırılan 4S direncinin geri testinde seviyeden short.

Kullanım:
  .venv/Scripts/python research.py build      # geliştirme coinlerinde kurulumlar + tüm çıkışlar
  .venv/Scripts/python research.py analyze    # etkiler, yürüyen seçim, dondurma, portföy
  .venv/Scripts/python research.py holdout    # dondurulmuş ayarı saklı coinlerde BİR KEZ test eder
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

import context
import engine
from analyze_sim import block_ci
from matcher import load
from setup_sim import H4, H4Info, find_setups, stop_entry_fill
from simulate import COST

OUT = Path(__file__).parent / "results" / "research"
FIRST, END, WARMUP_DAYS = "2018-07-01", "2026-09-26", 60
DEV_COINS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
HOLDOUT_COINS = ("XRPUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT", "AVAXUSDT")
LEVEL_MULTS = (2.0, 3.0)
STOP_KS = (1.0, 1.5)
# çıkış: (sabit hedef R [0 = yok], başabaşa çekme R [0 = yok], izleyen stop: son kaç 15dk mumun ucu, en fazla süre)
EXITS = {
    "hedef 1,5R": (1.5, 0, 0, 480),
    "hedef 2R": (2.0, 0, 0, 480),
    "hedef 3R": (3.0, 0, 0, 480),
    "hedef 4R": (4.0, 0, 0, 960),
    "1R'de başabaş + hedef 3R": (3.0, 1.0, 0, 480),
    "1R'de başabaş + hedef 5R": (5.0, 1.0, 0, 960),
    "1R'de başabaş + izleyen stop (12s)": (0.0, 1.0, 48, 960),
    "1R'de başabaş + izleyen stop (24s)": (0.0, 1.0, 96, 1440),
}
PARTIAL = {"yarısı 2R'de, yarısı izleyen stop (12s)": ("hedef 2R", "1R'de başabaş + izleyen stop (12s)")}
ALL_EXITS = list(EXITS) + list(PARTIAL)
FILTERS = {
    "filtre yok": lambda d: pd.Series(True, index=d.index),
    "boğa (fiyat 200g ort. üstünde)": lambda d: d.sezon == "boğa",
    "ayı (fiyat 200g ort. altında)": lambda d: d.sezon == "ayı",
    "günlük trend yukarı": lambda d: d.g_trend == "yukarı",
    "günlük trend aşağı": lambda d: d.g_trend == "aşağı",
    "günlük trend aşağı değil": lambda d: d.g_trend.isin(["yukarı", "yatay"]),
    "günlük trend yukarı değil": lambda d: d.g_trend.isin(["aşağı", "yatay"]),
    "düşük hacimli kırılım (<1,5x)": lambda d: d.brk_vol < 1.5,
    "yüksek hacimli kırılım (>=1,5x)": lambda d: d.brk_vol >= 1.5,
    "RSI(4s) < 50": lambda d: d.rsi < 50,
    "RSI(4s) >= 50": lambda d: d.rsi >= 50,
    "sığ kırılım (<1x 4S mum)": lambda d: d.depth < 1.0,
    "hızlı geri test (<48 saat)": lambda d: d.hours < 48,
}
BASE = {"family": "long", "lm": 3.0, "k": 1.0, "exit": "hedef 3R", "filter": "filtre yok"}   # şu anki kural
MIN_TRAIN = 40


@njit(cache=True)
def _manage(h, l, c, i, entry, risk, target_r, be_at, trail_bars, horizon, side):
    """i. mumda 'entry' fiyatından girilen işlem: (R, süre). risk fiyat cinsinden stop mesafesi.
    Her mumda önce stop (temkinli), sonra hedef kontrol edilir; dolum mumunda hedef sayılmaz.
    Başabaş ve izleyen stop, mum kapandıktan sonra bir sonraki mumdan itibaren geçerli olur."""
    stop = entry - side * risk
    tgt = entry + side * target_r * risk if target_r > 0 else side * np.inf
    be_done = False
    end = min(i + horizon, len(c))
    for j in range(i, end):
        if side == 1:
            if l[j] <= stop:
                return (stop - entry) / risk, j - i + 1
            if j > i and h[j] >= tgt:
                return target_r, j - i + 1
            if be_at > 0 and j > i and not be_done and h[j] >= entry + be_at * risk:
                be_done = True
                stop = max(stop, entry)
            if trail_bars > 0 and be_done and j - trail_bars + 1 > i:
                stop = max(stop, l[j - trail_bars + 1:j + 1].min())
        else:
            if h[j] >= stop:
                return (entry - stop) / risk, j - i + 1
            if j > i and l[j] <= tgt:
                return target_r, j - i + 1
            if be_at > 0 and j > i and not be_done and l[j] <= entry - be_at * risk:
                be_done = True
                stop = min(stop, entry)
            if trail_bars > 0 and be_done and j - trail_bars + 1 > i:
                stop = min(stop, h[j - trail_bars + 1:j + 1].max())
    return side * (c[end - 1] - entry) / risk, end - i


def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(close, prepend=close[0])
    up = pd.Series(np.maximum(d, 0)).ewm(alpha=1 / n, adjust=False).mean()
    dn = pd.Series(np.maximum(-d, 0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn)).to_numpy()


def build(coins, out_name: str) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for sym in coins:
        eng = engine.Engine("15m", sym)
        h4 = H4Info(sym)
        d4 = load("4h", sym)
        vol4 = d4.volume.to_numpy()
        avgvol4 = pd.Series(vol4).shift(1).rolling(180, min_periods=24).mean().to_numpy()   # önceki 30 gün
        rsi4 = rsi(d4.close.to_numpy())
        start = max(pd.Timestamp(FIRST), pd.Timestamp(int(eng.t[0]), unit="s") + pd.Timedelta(days=WARMUP_DAYS))
        a, b = int(start.timestamp()), int(pd.Timestamp(END).timestamp())
        rows = []
        for lm in LEVEL_MULTS:
            for st in find_setups(eng, lm, a, b):
                side = -st["side"]                 # dip kırılımı -> geri testte long, tepe kırılımı -> short
                i, p, a4 = st["i"], st["level"], st["a4"]
                jb = np.searchsorted(h4.t, st["break_ts"] - H4)                     # kırılım mumu (4S)
                jl = np.searchsorted(h4.t, eng.t[i] - H4, side="right") - 1         # girişten önce kapanmış son 4S mum
                ib = np.searchsorted(eng.t, st["break_ts"])
                if i > ib:
                    ext = eng.l[ib:i].min() if side == 1 else eng.h[ib:i].max()
                    depth = side * (p - ext) / p / a4
                else:
                    depth = 0.0
                # Seviyede bekleyen stop emri: mum seviyenin ötesinde açtıysa açılıştan dolar (DÜZELTME:
                # ilk sürümde her zaman seviyeden dolduğu varsayılmıştı, bu sonuçları ciddi şişiriyordu).
                entry = stop_entry_fill(eng.o, i, p, side)
                row = {"coin": sym, "lm": lm, "family": "long" if side == 1 else "short", "side": side,
                       "idx": i, "ts": int(eng.t[i]), "level": p, "entry": entry, "a4": a4,
                       "brk_vol": vol4[jb] / avgvol4[jb], "rsi": rsi4[jl], "depth": depth,
                       "hours": (eng.t[i] - st["break_ts"]) / 3600}
                for k in STOP_KS:
                    risk = k * a4 * p                           # emir seviyeye göre kurulur: stop/hedef fiyatı sabit
                    ra = side * (entry - (p - side * risk))     # gerçek risk (açılıştan dolduysa daha büyük)
                    cost_r = COST / (ra / entry)
                    for name, (tr, be, trail, hz) in EXITS.items():
                        t_act = side * (p + side * tr * risk - entry) / ra if tr > 0 else 0.0
                        be_act = side * (p + side * be * risk - entry) / ra if be > 0 else 0.0
                        r, bars = _manage(eng.h, eng.l, eng.c, i, entry, ra, t_act, be_act, trail, hz, side)
                        row[f"R|{k:g}|{name}"], row[f"B|{k:g}|{name}"] = r - cost_r, bars
                    for name, (e1, e2) in PARTIAL.items():
                        row[f"R|{k:g}|{name}"] = 0.5 * row[f"R|{k:g}|{e1}"] + 0.5 * row[f"R|{k:g}|{e2}"]
                        row[f"B|{k:g}|{name}"] = max(row[f"B|{k:g}|{e1}"], row[f"B|{k:g}|{e2}"])
                rows.append(row)
        g = pd.DataFrame(rows)
        ctx = context.features(g.ts.to_numpy(), g.level.to_numpy(), sym)
        g = pd.concat([g, ctx[["sezon", "g_trend"]]], axis=1)
        print(f"{sym}: {len(g)} kurulum ({start:%Y-%m-%d} -> {END})", flush=True)
        frames.append(g)
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df.ts, unit="s")
    df.to_csv(OUT / out_name, index=False)
    return df


def select_rows(df: pd.DataFrame, rule: dict) -> pd.Series:
    g = df[(df.family == rule["family"]) & (df.lm == rule["lm"])]
    g = g[FILTERS[rule["filter"]](g)]
    return g[f"R|{rule['k']:g}|{rule['exit']}"]


def describe(rule: dict) -> str:
    return f"{rule['family']}, hassasiyet {rule['lm']:g}, stop {rule['k']:g}x, {rule['exit']}, {rule['filter']}"


def line(name: str, df: pd.DataFrame, r: pd.Series, years: float) -> str:
    if len(r) < 10:
        return f"  {name:<44} {len(r):>4} işlem (az)"
    t = df.loc[r.index, "time"]
    lo, hi = block_ci(r, t.dt.to_period("M"))
    return (f"  {name:<44} {len(r):>4} işlem ({len(r) / years:4.1f}/yıl)  kârlı %{100 * (r > 0).mean():.0f}  "
            f"ort {r.mean():+.3f}R [{lo:+.2f},{hi:+.2f}]  toplam {r.sum():+.0f}R")


def all_rules(family: str):
    for lm, k, ex, flt in itertools.product(LEVEL_MULTS, STOP_KS, ALL_EXITS, FILTERS):
        yield {"family": family, "lm": lm, "k": k, "exit": ex, "filter": flt}


def best_rule(train: pd.DataFrame, family: str):
    best = None
    for rule in all_rules(family):
        r = select_rows(train, rule)
        if len(r) >= MIN_TRAIN and (best is None or r.sum() > best[0]):
            best = (r.sum(), rule)
    return best[1] if best else None


def portfolio(df: pd.DataFrame, rules: list[dict]) -> dict:
    """Coin başına aynı anda tek pozisyon (aileler aynı yuvayı paylaşır). Yıllık işlem, R ve en büyük düşüş."""
    parts = []
    for rule in rules:
        r = select_rows(df, rule)
        bars = df.loc[r.index, f"B|{rule['k']:g}|{rule['exit']}"]
        parts.append(pd.DataFrame({"coin": df.loc[r.index, "coin"], "ts": df.loc[r.index, "ts"], "R": r,
                                   "exit_ts": df.loc[r.index, "ts"] + bars * 900}))
    t = pd.concat(parts).sort_values("ts")
    taken, free = [], {}
    for row in t.itertuples():
        if row.ts >= free.get(row.coin, 0):
            taken.append(row)
            free[row.coin] = row.exit_ts
    p = pd.DataFrame(taken).sort_values("exit_ts")
    curve = p.R.cumsum().to_numpy()
    dd = float((np.maximum.accumulate(np.r_[0, curve])[1:] - curve).max())
    span_years = (p.ts.max() - p.ts.min()) / (365.25 * 86400)
    p["year"] = pd.to_datetime(p.ts, unit="s").dt.year
    by_year = p.groupby("year").R.agg(["size", "sum"])
    return {"işlem": len(p), "yıllık işlem": len(p) / span_years, "haftalık işlem": len(p) / span_years / 52.18,
            "toplam R": float(p.R.sum()), "yıllık R": float(p.R.sum()) / span_years, "en büyük düşüş R": dd,
            "yıllar": by_year, "trades": p[["coin", "ts", "exit_ts", "R"]].reset_index(drop=True)}


def print_portfolio(name: str, pf: dict) -> None:
    print(f"  {name}: {pf['işlem']} işlem, yılda {pf['yıllık işlem']:.0f} (haftada {pf['haftalık işlem']:.1f}), "
          f"toplam {pf['toplam R']:+.0f}R, yılda {pf['yıllık R']:+.1f}R, en büyük düşüş {pf['en büyük düşüş R']:.1f}R")
    print("    yıllar: " + "  ".join(f"{y}: {int(n)} işlem {s:+.1f}R" for y, (n, s) in pf["yıllar"].iterrows()))


def analyze():
    df = pd.read_csv(OUT / "dev_setups.csv", parse_dates=["time"])
    years = (df.time.max() - df.time.min()).days / 365.25
    base = df[(df.family == "long") & (df.lm == 3.0)]
    print(f"Geliştirme verisi: {len(df)} kurulum, {df.coin.nunique()} coin, {df.time.min():%Y-%m} -> {df.time.max():%Y-%m}\n")

    print("1) ÇIKIŞ YÖNTEMİ (long, hassasiyet 3, filtre yok)")
    for k in STOP_KS:
        for ex in ALL_EXITS:
            print(line(f"stop {k:g}x, {ex}", df, base[f"R|{k:g}|{ex}"], years))
    print("\n2) FİLTRE (long, hassasiyet 3, stop 1x, hedef 3R)")
    for flt, fn in FILTERS.items():
        print(line(flt, df, base[fn(base)]["R|1|hedef 3R"], years))
    print("\n3) DAHA ÇOK KURULUM (long, stop 1x, hedef 3R)")
    for lm in LEVEL_MULTS:
        print(line(f"seviye hassasiyeti {lm:g}", df, df[(df.family == "long") & (df.lm == lm)]["R|1|hedef 3R"], years))
    print("\n4) SHORT AİLESİ (kırılan direncin geri testinde short, hassasiyet 3, stop 1x)")
    sh = df[(df.family == "short") & (df.lm == 3.0)]
    for flt in ("filtre yok", "ayı (fiyat 200g ort. altında)", "günlük trend aşağı", "günlük trend yukarı değil"):
        for ex in ("hedef 2R", "hedef 3R"):
            print(line(f"{flt}, {ex}", df, sh[FILTERS[flt](sh)][f"R|1|{ex}"], years))

    print("\n5) YÜRÜYEN SEÇİM (her yıl sadece önceki yıllara bakarak seçilir)")
    wf = {"long": [], "short": [], "base": []}
    for y in range(2020, 2027):
        train, test = df[df.time.dt.year < y], df[df.time.dt.year == y]
        base_r = select_rows(test, BASE)
        wf["base"].append(base_r)
        msg = f"  {y}: şu anki kural {len(base_r):>3} işlem {base_r.sum():+6.1f}R"
        chosen = {}
        for fam in ("long", "short"):
            rule = chosen[fam] = best_rule(train, fam)
            r = select_rows(test, rule) if rule else pd.Series(dtype=float)
            wf[fam].append(r)
            msg += f" | seçilen {fam}: {len(r):>3} işlem {r.sum():+6.1f}R"
        print(msg)
        for fam, rule in chosen.items():
            print(f"        {fam} seçimi: {describe(rule) if rule else 'yeterli işlem yok'}")
    tot = {k: pd.concat(v) for k, v in wf.items()}
    for k, name in (("base", "şu anki kural"), ("long", "yürüyen seçim, long"), ("short", "yürüyen seçim, short")):
        r = tot[k]
        print(f"  TOPLAM {name:<22} {len(r):>4} işlem, ort {r.mean():+.3f}R, toplam {r.sum():+.1f}R")

    frozen = [best_rule(df, "long")]
    use_short = tot["short"].sum() > 0
    if use_short:
        frozen.append(best_rule(df, "short"))
    print("\n6) DONDURULAN AYAR (tüm geliştirme verisiyle, aynı ölçütle seçildi)")
    for rule in frozen:
        print(f"  {describe(rule)}")
    print(f"  short ailesi {'portföye alındı' if use_short else 'alınmadı (yürüyen seçim toplamı pozitif değil)'}")
    (OUT / "frozen.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n7) PORTFÖY (geliştirme coinleri birlikte, coin başına aynı anda tek pozisyon)")
    print_portfolio("şu anki kural", portfolio(df, [BASE]))
    print_portfolio("dondurulan ayar", portfolio(df, frozen))


def holdout():
    frozen = json.loads((OUT / "frozen.json").read_text(encoding="utf-8"))
    df = build(HOLDOUT_COINS, "holdout_setups.csv")
    years = (df.time.max() - df.time.min()).days / 365.25
    print("\nSAKLI COINLER - TEK SEFERLİK SINAV")
    for name, rules in (("şu anki kural", [BASE]), ("dondurulan ayar", frozen)):
        print(f"\n{name}:")
        for rule in rules:
            print(f"  [{describe(rule)}]")
            for sym in HOLDOUT_COINS:
                print(line(sym, df, select_rows(df[df.coin == sym], rule), years))
            print(line("hepsi birlikte", df, select_rows(df, rule), years))
        print_portfolio("portföy", portfolio(df, rules))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    {"build": lambda: build(DEV_COINS, "dev_setups.csv"), "analyze": analyze, "holdout": holdout}[cmd]()
