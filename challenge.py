"""Fon şirketi challenge simülasyonu: dondurulan ayarla gerçek işlem sırası üzerinden geçme olasılığı.

Yöntem: geçmişteki her işlem tarihinde yeni bir challenge başlatılır ve işlemler GERÇEK sırasıyla
uygulanır (kayıp serileri korunur). Risk, başlangıç bakiyesinin sabit yüzdesi. Bakiye sadece işlem
kapanınca değişir (açık pozisyonun anlık zararı hesaba katılmaz - gerçekte biraz daha sıkıdır).

Challenge tipleri (firmalarda sık görülen tipik kurallar, belirli bir firmanınkiler değil):
  A: 2 aşama. 1. aşama +%10, 2. aşama +%5 hedef. Toplam kayıp limiti %10 (başlangıçtan), günlük %5.
  B: 1 aşama. +%10 hedef. Kayıp limiti zirveden %6 (izleyen), günlük %3.

Kullanım: .venv/Scripts/python challenge.py
"""
import json

import numpy as np
import pandas as pd

from research import OUT, portfolio

COINS = ("BTCUSDT", "ETHUSDT")
RISKS = (0.25, 0.5, 0.75, 1.0, 1.5)          # işlem başına risk, başlangıç bakiyesinin yüzdesi
HAIRCUTS = {"geçmiş aynen": 0.0, "temkinli (her işlem -0,1R)": 0.1}
TYPES = {
    "A: 2 aşama (+10% / +5%), kayıp limiti 10%, günlük 5%": {"targets": (10, 5), "max_loss": 10, "trailing": False, "daily": 5},
    "B: 1 aşama (+10%), izleyen kayıp limiti 6%, günlük 3%": {"targets": (10,), "max_loss": 6, "trailing": True, "daily": 3},
}


def run_challenge(R: np.ndarray, day: np.ndarray, t_exit: np.ndarray, start: int, risk: float, rules: dict):
    """start. işlemden başlar. Döner: ('geçti' / 'kaldı' / 'bitmedi', geçene kadar geçen gün)."""
    t0 = t_exit[start]
    for target in rules["targets"]:                      # her aşamada bakiye sıfırdan başlar
        eq = peak = 0.0
        day_pnl, cur_day = 0.0, None
        while True:
            if start >= len(R):
                return "bitmedi", None
            if day[start] != cur_day:
                cur_day, day_pnl = day[start], 0.0
            pnl = R[start] * risk
            eq += pnl
            day_pnl += pnl
            peak = max(peak, eq)
            start += 1
            floor = (peak if rules["trailing"] else 0.0) - rules["max_loss"]
            if eq <= floor or day_pnl <= -rules["daily"]:
                return "kaldı", None
            if eq >= target:
                break
    return "geçti", (t_exit[start - 1] - t0) / 86400


def main():
    frozen = json.loads((OUT / "frozen.json").read_text(encoding="utf-8"))
    dev = pd.read_csv(OUT / "dev_setups.csv", parse_dates=["time"])
    tr = portfolio(dev[dev.coin.isin(COINS)], frozen)["trades"].sort_values("exit_ts").reset_index(drop=True)
    print(f"İşlem akışı: {' + '.join(c[:-4] for c in COINS)}, dondurulan ayar, {len(tr)} işlem, "
          f"işlem başına ort {tr.R.mean():+.3f}R ({pd.Timestamp(tr.ts.min(), unit='s'):%Y-%m} -> "
          f"{pd.Timestamp(tr.ts.max(), unit='s'):%Y-%m})")
    streak = (tr.R < 0).astype(int).groupby((tr.R >= 0).cumsum()).sum().max()
    print(f"En uzun kayıp serisi: {streak} işlem\n")
    day = (tr.exit_ts // 86400).to_numpy()
    t_exit = tr.exit_ts.to_numpy()
    for hname, cut in HAIRCUTS.items():
        R = tr.R.to_numpy() - cut
        print(f"##### {hname}: işlem başına ort {R.mean():+.3f}R")
        for tname, rules in TYPES.items():
            print(f"  {tname}")
            for risk in RISKS:
                res = [run_challenge(R, day, t_exit, s, risk, rules) for s in range(len(R))]
                done = [r for r in res if r[0] != "bitmedi"]
                passed = [d for s, d in done if s == "geçti"]
                rate = 100 * len(passed) / len(done)
                med = f"{np.median(passed):.0f} gün" if passed else "-"
                print(f"    işlem başına risk %{risk:<5g} geçme %{rate:5.1f}   kalma %{100 - rate:5.1f}   "
                      f"geçenlerin ortanca süresi {med}  ({len(done)} deneme)")
        print()


if __name__ == "__main__":
    main()
