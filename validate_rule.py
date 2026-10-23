"""Dondurulmuş kuralın bağımsız veride testi.

KURAL (BTC 2018-2023'te geliştirildi, değiştirilmez):
  Kırılan 4S desteğinin geri testinde seviyeden long.
  - Seviye: onaylanmış 4S dip (zigzag hassasiyeti 3).
  - Kırılım: onaydan sonraki 30 gün içinde bir 4S mumu seviyenin altında kapanır.
  - Geri test: kırılımdan 4 saat - 7 gün sonra 15dk'da fiyat aşağıdan seviyeye dokunur -> seviyeden long.
  - Stop: seviyenin 1 x 4S normal mum boyu altı. Hedef 3R. En fazla 5 gün. Masraf %0,14.
GEÇME ŞARTI (sonuçtan önce konuldu): ortalama R > 0 ve tek pozisyon toplamı > 0.

Kullanım: .venv/Scripts/python validate_rule.py <SEMBOL> <başlangıç> <bitiş>
   örn.:  .venv/Scripts/python validate_rule.py ETHUSDT 2018-07-01 2025-01-01
"""
import sys

import numpy as np
import pandas as pd

import engine
from analyze_sim import block_ci, equity
from simulate import RESULTS
from sweep_sim import reversed_retests

KEY = "k1_t3"


def main(symbol: str, start: str, end: str):
    eng = engine.Engine("15m", symbol)
    a, b = int(pd.Timestamp(start).timestamp()), int(pd.Timestamp(end).timestamp())
    df = reversed_retests(eng, a, b)
    g = df[df.side == "long"].sort_values("time").reset_index(drop=True)
    g.to_csv(RESULTS / f"rule_{symbol.lower()}_{start}_{end}.csv", index=False)
    r = g[f"real_{KEY}"]
    lo, hi = block_ci(r, g.time.dt.to_period("M"))
    eq = equity(pd.DataFrame({"idx": g.idx, "bars": g[f"bars_{KEY}"], "real": r}), pd.Series(True, index=g.index))
    print(f"{symbol} {start} -> {end}: {len(g)} işlem, kazanma %{100 * (g[f'kind_{KEY}'] == 1).mean():.0f}, "
          f"ort {r.mean():+.3f}R  %90 aralık [{lo:+.2f}, {hi:+.2f}]")
    print("  yıllara göre: " + "  ".join(
        f"{y}: {r[g.time.dt.year == y].mean():+.2f} ({(g.time.dt.year == y).sum()})" for y in sorted(g.time.dt.year.unique())))
    print(f"  tek pozisyon: {eq}")
    passed = r.mean() > 0 and eq["toplam R"] > 0
    print(f"  GEÇME ŞARTI (ort > 0 ve tek poz. toplam > 0): {'GEÇTİ' if passed else 'GEÇEMEDİ'}"
          + ("  | güven aralığı alt sınırı > 0: EVET" if lo > 0 else "  | güven aralığı alt sınırı > 0: hayır"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
