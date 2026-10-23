"""Geçmişte, verilen mum dizisine şekil olarak benzeyen anları bulan çekirdek fonksiyonlar.

Fikir:
- Her pencere (sorgu ya da geçmişteki aday) K tane "birleştirilmiş muma" indirgenir.
  Böylece 50 mumluk bir yapı ile 100 mumluk bir yapı aynı boyda karşılaştırılabilir.
- Her birleştirilmiş mumdan [yüksek, düşük, kapanış] alınır (fitiller dahil).
- Ayak izi: her birleştirilmiş mumun içindeki en büyük yeşil ve en büyük kırmızı gövde,
  yapının toplam boyuna oranla. Birleştirme "tek dev mum" ile "birkaç küçük mum" farkını
  silmesin diye.
- Fiyat seviyesi yok sayılır (şekil karşılaştırılır), ama yapının o dönemin oynaklığına
  göre büyüklüğü çok farklıysa aday elenir.
- Aynı olayın bir mum kaymış kopyaları tek olay sayılır.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

DATA_DIR = Path(__file__).parent / "data"

K = 48                                  # her pencere bu kadar birleştirilmiş muma indirgenir
SCALES = np.geomspace(0.4, 2.5, 13)     # adayın sorguya göre uzunluk oranları (1.0 dahil)
FOOT_CAP = 0.25                         # yapı boyunun %25'inden büyük gövde "büyük mum" sayılır, fazlası önemsiz
FOOT_GAIN = 4.0                         # 0..FOOT_CAP aralığını 0..1'e taşır (şekil kanallarıyla aynı ölçek)
FOOT_SPREAD = 3                         # büyük mum bu kadar birleştirilmiş mum kaymış olsa da eşleşir


def weights(foot: float) -> np.ndarray:
    """Kanal ağırlıkları: [yüksek, düşük, kapanış, büyük yeşil, büyük kırmızı]. Fitillere yarım ağırlık."""
    return np.array([0.5, 0.5, 1.0, foot, foot])


def load(interval: str = "1h", symbol: str = "BTCUSDT") -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{symbol.lower()}_{interval}.csv")
    df["time"] = pd.to_datetime(df.open_time, unit="ms")
    return df


class Market:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.time = df.time.reset_index(drop=True)
        self.o, self.h, self.l, self.c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        self.up = np.maximum(self.c - self.o, 0)      # yeşil gövde boyu (kırmızıda 0)
        self.down = np.maximum(self.o - self.c, 0)    # kırmızı gövde boyu (yeşilde 0)
        # Son 30 günün ortalama mum boyu (yüzde): "o dönemin normali".
        self.avg_range = pd.Series((self.h - self.l) / self.c).rolling(720, min_periods=24).mean().to_numpy()
        self._cache = {}

    def index_of(self, ts: str) -> int:
        return int(self.time.searchsorted(pd.Timestamp(ts)))

    def _rolling(self, arr, size, fn):
        key = (id(arr), size)
        if key not in self._cache:
            self._cache[key] = fn(sliding_window_view(arr, size), axis=1)
        return self._cache[key]

    def windows(self, starts: np.ndarray, m: int) -> np.ndarray:
        """starts'ta başlayan m mumluk pencereleri K birleştirilmiş muma indirger -> (N, K, 5).

        Kanallar: yüksek, düşük, kapanış, en büyük yeşil gövde, en büyük kırmızı gövde (fiyat cinsinden).
        """
        edges = np.floor(np.arange(K + 1) * m / K).astype(int)
        lo = edges[:-1]
        hi = np.maximum(edges[1:], lo + 1)
        X = np.empty((len(starts), K, 5))
        for b in range(K):
            size = hi[b] - lo[b]
            first = starts + lo[b]
            X[:, b, 0] = self._rolling(self.h, size, np.max)[first]
            X[:, b, 1] = self._rolling(self.l, size, np.min)[first]
            X[:, b, 2] = self.c[starts + hi[b] - 1]
            X[:, b, 3] = self._rolling(self.up, size, np.max)[first]
            X[:, b, 4] = self._rolling(self.down, size, np.max)[first]
        return X

    def rel_size(self, X: np.ndarray, starts: np.ndarray, m: int) -> np.ndarray:
        """Yapının boyu / (dönemin normal mum boyu * sqrt(uzunluk)). Dönemden bağımsız büyüklük."""
        span = (X[:, :, 0].max(1) - X[:, :, 1].min(1)) / X[:, :, 2].mean(1)
        return span / (self.avg_range[starts + m - 1] * np.sqrt(m))


def normalize(X: np.ndarray) -> np.ndarray:
    """Fiyat seviyesini ve ölçeğini kaldırır.

    Şekil kanalları: kapanışlar ortalama 0, standart sapma 1 olur.
    Ayak izi kanalları: gövde / yapının toplam boyu (grafikte gözün gördüğü oran), FOOT_CAP'te
    kesilir: "büyük mum var mı" önemli, birinin diğerinden iki kat büyük olması değil. Büyük mum
    birkaç birleştirilmiş mum kaymış olsa da eşleşsin diye FOOT_SPREAD kadar komşularına yayılır.
    """
    c = X[:, :, 2]
    mu, sd = c.mean(1)[:, None, None], c.std(1)[:, None, None] + 1e-12
    height = (X[:, :, 0].max(1) - X[:, :, 1].min(1))[:, None, None] + 1e-12
    Z = np.empty_like(X)
    Z[:, :, :3] = (X[:, :, :3] - mu) / sd
    f = np.minimum(X[:, :, 3:] / height, FOOT_CAP) * FOOT_GAIN
    Z[:, :, 3:] = f
    for shift in range(1, FOOT_SPREAD + 1):
        Z[:, shift:, 3:] = np.maximum(Z[:, shift:, 3:], f[:, :-shift])
        Z[:, :-shift, 3:] = np.maximum(Z[:, :-shift, 3:], f[:, shift:])
    return Z


def euclid(q: np.ndarray, X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Birebir karşılaştırma: i. birleştirilmiş mum i. ile karşılaştırılır."""
    return ((X - q) ** 2 * w).sum(2).mean(1) / w.sum()


def dtw(q: np.ndarray, X: np.ndarray, band: int, w: np.ndarray) -> np.ndarray:
    """DTW: yapının bölümleri farklı hızda oluşmuş olabilir; en fazla `band` kaydırmaya izin verir."""
    n = len(X)
    prev = np.full((n, K), np.inf)
    for i in range(K):
        cur = np.full((n, K), np.inf)
        for j in range(max(0, i - band), min(K, i + band + 1)):
            cost = ((X[:, j] - q[i]) ** 2 * w).sum(1) / w.sum()
            if i == 0 and j == 0:
                cur[:, 0] = cost
                continue
            best = prev[:, j]
            if j > 0:
                best = np.minimum(best, np.minimum(cur[:, j - 1], prev[:, j - 1]))
            cur[:, j] = cost + best
        prev = cur
    return prev[:, K - 1] / K


def similarity(dist: np.ndarray) -> np.ndarray:
    """Mesafeyi 0-100 arası benzerliğe çevirir (kapanışlar için korelasyona denk)."""
    return np.clip(100 * (1 - dist / 2), 0, 100)


def search(mkt: Market, q_start: int, q_end: int, method: str = "euclid",
           foot: float = 0.0, band_frac: float = 0.25, size_tol: float = 2.0):
    """[q_start, q_end) sorgusuna benzeyen, sorgudan önce bitmiş tüm adayları puanlar.

    foot: ayak izi (büyük gövdeli mum) kanallarının ağırlığı; 0 = sadece şekil.
          Test vakasında (21 Eyl 2026) 1 ve 2 ağırlıkları sıralamayı iyileştirmedi, varsayılan 0.
    Döner: (dist, start, length) dizileri, her aday için bir eleman.
    """
    L = q_end - q_start
    q_starts = np.array([q_start])
    qX = mkt.windows(q_starts, L)
    q_rel = mkt.rel_size(qX, q_starts, L)[0]
    q = normalize(qX)[0]
    band = int(round(band_frac * K))
    w = weights(foot)

    all_d, all_s, all_m = [], [], []
    for scale in SCALES:
        m = int(round(L * scale))
        starts = np.arange(0, q_start - m + 1)
        if len(starts) == 0:
            continue
        X = mkt.windows(starts, m)
        ratio = mkt.rel_size(X, starts, m) / q_rel
        keep = (ratio > 1 / size_tol) & (ratio < size_tol)
        starts, X = starts[keep], normalize(X[keep])
        d = euclid(q, X, w) if method == "euclid" else dtw(q, X, band, w)
        all_d.append(d)
        all_s.append(starts)
        all_m.append(np.full(len(starts), m))
    return np.concatenate(all_d), np.concatenate(all_s), np.concatenate(all_m)


def distinct_events(dist, start, length, limit: int = 20):
    """En iyiden başlayarak seçer; önceden seçilmiş bir olayla zamanı çakışan adayları atlar."""
    occupied = np.zeros(int((start + length).max()) + 1, bool)
    picks = []
    for k in np.argsort(dist):
        s, m = start[k], length[k]
        if occupied[s:s + m].any():
            continue
        occupied[s:s + m] = True
        picks.append(k)
        if len(picks) == limit:
            break
    return picks
