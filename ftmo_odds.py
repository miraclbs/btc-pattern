"""BTC + ETH için düzeltilmiş (gerçekçi dolumlu) istatistikler ve FTMO 2 aşamalı challenge geçme ihtimali.

Kurallar (sadece long; short taraf düzeltmeden sonra negatif):
  basit     : kırılan 4S desteğinin geri testinde long, seviye hassasiyeti 3, stop 1x 4S mum, hedef 3R
  çok işlem : aynısı, seviye hassasiyeti 2 (daha çok kurulum), hedef 4R
Coin başına aynı anda tek pozisyon.

FTMO 2 aşama: 1. aşama +%10, 2. aşama +%5, toplam kayıp limiti %10 (başlangıçtan), günlük %5.
Senaryolar: düzeltilmiş geçmiş / temkinli (her işlem -0,1R) / sıfır avantaj (ortalama 0'a çekilmiş).
Bakiye sadece işlem kapanınca değişir (FTMO açık pozisyonun zararını da sayar - gerçekte biraz daha sıkı).

Kullanım: .venv/Scripts/python ftmo_odds.py [şu anki bakiye yüzdesi, örn. -1.98]
"""
import sys

import numpy as np
import pandas as pd

from research import OUT, build, portfolio

COINS = ("BTCUSDT", "ETHUSDT")
RULES = {
    "basit (hassasiyet 3, hedef 3R)": {"family": "long", "lm": 3.0, "k": 1.0, "exit": "hedef 3R", "filter": "filtre yok"},
    "çok işlem (hassasiyet 2, hedef 4R)": {"family": "long", "lm": 2.0, "k": 1.0, "exit": "hedef 4R", "filter": "filtre yok"},
}
RISKS = (0.25, 0.5, 0.75, 1.0)
PHASES, MAX_LOSS, DAILY = (10.0, 5.0), 10.0, 5.0


def challenge(R, day, t_exit, start, risk, start_eq=0.0):
    """start. işlemden başlar: ('geçti' / 'kaldı' / 'bitmedi', gün). İlk aşama start_eq'ten başlar."""
    t0 = t_exit[start]
    for n, target in enumerate(PHASES):
        eq = start_eq if n == 0 else 0.0
        day_pnl, cur_day = 0.0, None
        while True:
            if start >= len(R):
                return "bitmedi", None
            if day[start] != cur_day:
                cur_day, day_pnl = day[start], 0.0
            eq += R[start] * risk
            day_pnl += R[start] * risk
            start += 1
            if eq <= -MAX_LOSS or day_pnl <= -DAILY:
                return "kaldı", None
            if eq >= target:
                break
    return "geçti", (t_exit[start - 1] - t0) / 86400


def main(current: float):
    path = OUT / "btceth_fixed.csv"
    df = pd.read_csv(path) if path.exists() else build(COINS, path.name)
    for name, rule in RULES.items():
        pf = portfolio(df, [rule])
        tr = pf["trades"].sort_values("exit_ts").reset_index(drop=True)
        R0 = tr.R.to_numpy()
        losses = (R0 <= -0.99).astype(int)
        streak = losses.groupby if False else max((len(s) for s in "".join(map(str, losses)).split("0")), default=0)
        years = pf["yıllar"]
        print(f"\n################ {name} ################")
        print(f"  {len(tr)} işlem ({pf['haftalık işlem']:.1f}/hafta), kârlı %{100 * (R0 > 0).mean():.0f}, "
              f"işlem başına ort {R0.mean():+.3f}R, en büyük düşüş {pf['en büyük düşüş R']:.1f}R, "
              f"arka arkaya en çok {streak} tam kayıp")
        print("  coin bazında: " + "  ".join(f"{c[:3]}: {tr.R[tr.coin == c].mean():+.3f}R ({(tr.coin == c).sum()} işlem)"
                                            for c in COINS))
        print("  yıllar: " + "  ".join(f"{y}: {s:+.1f}R" for y, (n, s) in years.iterrows()))
        day, t_exit = (tr.exit_ts // 86400).to_numpy(), tr.exit_ts.to_numpy()
        scen = {"düzeltilmiş geçmiş": R0, "temkinli (-0,1R)": R0 - 0.1, "sıfır avantaj": R0 - R0.mean()}
        for start_name, start_eq in (("yeni challenge", 0.0), (f"şu anki hesap (%{current:+.2f}'den)", current)):
            print(f"  --- {start_name}: geçme ihtimali (geçenlerin ortanca süresi) ---")
            print(f"  {'risk':<8}" + "".join(f"{s:>30}" for s in scen))
            for risk in RISKS:
                cells = []
                for R in scen.values():
                    res = [challenge(R, day, t_exit, s, risk, start_eq) for s in range(len(R))]
                    done = [r for r in res if r[0] != "bitmedi"]
                    p = [d for s, d in done if s == "geçti"]
                    cells.append(f"%{100 * len(p) / len(done):4.0f} ({np.median(p):4.0f} gün)" if p else "%   0")
                print(f"  %{risk:<7g}" + "".join(f"{c:>30}" for c in cells))


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else -1.98)
