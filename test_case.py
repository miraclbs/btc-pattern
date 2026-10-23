"""Test vakası: 21 Eylül 2026 yapısını (grafik 1) "şu an" kabul edip geçmişte arar.
Kullanıcının "bu benim yapım" dediği olayların kaçıncı sırada çıktığına bakar.

Kullanım: python test_case.py [ayak izi ağırlıkları...]   örn: python test_case.py 0 1
Grafikler results/ klasörüne kaydedilir.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import matcher

QUERY_START = "2026-09-21 06:00"                      # yükselişin biraz öncesi
NOW_POINTS = ["2026-09-24 00:00", "2026-09-25 00:00", "2026-09-26 00:00"]
TARGETS = {                                           # kullanıcının benzer bulduğu olaylar
    "grafik 2": ("2026-09-03 00:00", "2026-09-08 23:00"),
    "24 Ara 2024": ("2024-12-24 12:00", "2024-12-27 06:00"),
}
FOOT_WEIGHTS = [0.0, 1.0, 2.0]                        # 0 = eski yöntem (sadece şekil)
PLOT_FOOT = 1.0
TOP = 20
RESULTS = Path(__file__).parent / "results"

BG, UP, DOWN, HL = "#161a25", "#4caf50", "#e6e6e6", "#2a3550"


def candles(ax, mkt, a, b, x0=0):
    x = np.arange(b - a) + x0
    o, h, l, c = mkt.o[a:b], mkt.h[a:b], mkt.l[a:b], mkt.c[a:b]
    col = np.where(c >= o, UP, DOWN)
    ax.vlines(x, l, h, color=col, lw=0.6)
    ax.bar(x, np.maximum(abs(c - o), (h.max() - l.min()) * 0.002), bottom=np.minimum(o, c),
           width=0.7, color=col)


def panel(ax, mkt, s, m, title, color="w", after=True):
    """Eşleşmeyi (vurgulu), öncesini ve sonrasını çizer."""
    pre = m // 2
    post = m if after else 0
    a, b = max(0, s - pre), min(len(mkt.c), s + m + post)
    candles(ax, mkt, a, b)
    ax.axvspan(s - a - 0.5, s - a + m - 0.5, color=HL, zorder=0)
    ax.set_xlim(-1, b - a)
    ax.set_ylim(mkt.l[a:b].min() * 0.998, mkt.h[a:b].max() * 1.002)
    ax.set_facecolor(BG)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, color=color, fontsize=7)


def target_of(mkt, s, m):
    """Aday, takip edilen olaylardan biriyle en az yarı yarıya çakışıyorsa onun adını döner."""
    for name, (a, b) in TARGETS.items():
        t0, t1 = mkt.index_of(a), mkt.index_of(b)
        if max(0, min(s + m, t1) - max(s, t0)) >= 0.5 * m:
            return name
    return None


def run(mkt, now, method, foot):
    qs, qe = mkt.index_of(QUERY_START), mkt.index_of(now)
    dist, start, length = matcher.search(mkt, qs, qe, method, foot=foot)
    picks = matcher.distinct_events(dist, start, length, limit=500)
    ranks = {}
    for r, k in enumerate(picks):
        name = target_of(mkt, start[k], length[k])
        if name and name not in ranks:
            ranks[name] = r + 1
    return qs, qe, dist, start, length, picks[:TOP], ranks


def plot(mkt, now, method, foot, qs, qe, dist, start, length, picks):
    sim = matcher.similarity(dist)
    fig, axes = plt.subplots(3, 7, figsize=(21, 8), facecolor=BG)
    axes = axes.ravel()
    panel(axes[0], mkt, qs, qe - qs, f"SORGU (şu an = {now[5:16]})\n{qe - qs} mum", after=False)
    for ax, (r, k) in zip(axes[1:], enumerate(picks)):
        s, m = start[k], length[k]
        hit = target_of(mkt, s, m)
        title = (f"#{r + 1}  {mkt.time[s]:%Y-%m-%d %H:%M}\n%{sim[k]:.0f} benzer, {m} mum"
                 + (f"  <- {hit.upper()}" if hit else ""))
        panel(ax, mkt, s, m, title, color="#ffcc00" if hit else "w")
    fig.suptitle(f"Yöntem: {method}, ayak izi ağırlığı {foot:g}   "
                 f"(vurgulu alan = eşleşen yapı, sağı = sonrası)", color="w")
    plt.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{method}_foot{foot:g}_{now[:10]}.png"
    plt.savefig(out, dpi=80, facecolor=BG)
    plt.close(fig)
    return out


def main():
    global FOOT_WEIGHTS
    if len(sys.argv) > 1:
        FOOT_WEIGHTS = [float(a) for a in sys.argv[1:]]
    mkt = matcher.Market(matcher.load("1h"))
    names = list(TARGETS)
    print(f"{'şu an':<12}{'yöntem':<8}{'ayak izi':<10}" + "".join(f"{n:<16}" for n in names))
    for now in NOW_POINTS:
        for method in ("euclid", "dtw"):
            for foot in FOOT_WEIGHTS:
                qs, qe, dist, start, length, picks, ranks = run(mkt, now, method, foot)
                if foot == PLOT_FOOT:
                    plot(mkt, now, method, foot, qs, qe, dist, start, length, picks)
                cells = []
                for n in names:
                    r = ranks.get(n)
                    cells.append(f"{'-' if r is None else r}{' ✓' if r and r <= TOP else ''}")
                print(f"{now[5:10]:<12}{method:<8}{foot:<10g}" + "".join(f"{c:<16}" for c in cells),
                      flush=True)


if __name__ == "__main__":
    main()
