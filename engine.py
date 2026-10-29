"""Web arayüzünün kullandığı hızlı arama motoru.

matcher.py'deki yöntemin (sadece şekil, ayak izi yok) numba ile derlenmiş, çok çekirdekli hali.
"Şu an" için yapı başlangıçlarını son tepe ve diplerden seçer; her başlangıç ayrı bir gruptur.

Hız için iki aşamalı tarar: önce adaylar seyrek denenir (bkz. STRIDE_DIV), sonra en iyi 1000 adayın
atlanan komşuları tek tek denenir. Pencere K=48 parçaya indirildiği için bir parçadan küçük kaydırma
şekli pek değiştirmez. exact=True tam taramayı yapar.
"""
import time

import numpy as np
import pandas as pd
from numba import njit, prange

import context
from matcher import load

K = 48                              # her pencere bu kadar birleştirilmiş muma indirgenir
SCALES = np.geomspace(0.4, 2.5, 13) # adayın sorguya göre uzunluk oranları
W = np.array([0.5, 0.5, 1.0])       # [yüksek, düşük, kapanış]: fitillere yarım ağırlık
BAND = 12                           # DTW: yapının bölümleri en fazla K'nın %25'i kadar kayabilir
SIZE_TOL = 2.0                      # döneme göre boyu sorgunun 2 katından fazla farklı aday elenir
# Ön tarama: m mumluk adaylar max(1, m // STRIDE_DIV) mum arayla denenir. Birebir yöntem kaymaya
# DTW kadar toleranslı olmadığından daha sık taranır (25 grupta tam taramayla ortak: DTW 9,84/10,
# birebir 9,96/10).
STRIDE_DIV = {True: 8, False: 16}   # {DTW mi: bölen}
REFINE_TOP = 3000                   # ön taramanın en iyi bu kadar adayının komşuları tek tek denenir
                                    # (12 grupta ilk 50 olayın ort. 48,4'ü tam taramayla aynı)

# "O dönemin normal mum boyu" kaç mumun ortalaması (zaman dilimine göre; 1s için 30 gün).
AVG_WINDOW = {"15m": 2880, "1h": 720, "4h": 360, "1d": 120}
INTERVALS = tuple(AVG_WINDOW)

PIVOT_MULT = 4.0                    # fiyat, dönemin normal mumunun bu katı kadar dönünce tepe/dip onaylanır
PIVOT_LOOKBACK = 1500               # tepe/dip ararken geriye bakılacak mum sayısı
MIN_LEN, MAX_LEN = 24, 360          # otomatik gruplarda bir yapı en az / en fazla bu kadar mum olabilir
MIN_SEL, MAX_SEL = 10, 600          # kullanıcının seçtiği alan en az / en fazla bu kadar mum olabilir
MAX_GROUPS, TOP = 5, 50             # en fazla 5 başlangıç noktası, her birinde en benzer 50 olay
MAIN_BARS = 1000                    # ana grafikte gösterilecek (ve seçim yapılabilecek) mum sayısı
AFTER_MAX = 10                      # eşleşmenin sonrası en fazla yapı uzunluğunun bu katı kadar gösterilir
CARD_BARS = 400                     # bir sonuç kartı için gönderilen en fazla mum (fazlası birleştirilir)
DETAIL_BARS = 6000                  # karşılaştırma penceresi için gönderilen en fazla mum
DETAIL_MIN_AFTER = 1000             # karşılaştırma penceresinde sonrası en az bu kadar mum (kısa yapılar için)
DETAIL_MIN_BEFORE = 1000            # karşılaştırma penceresinde öncesi en az bu kadar mum (seviyeler görünsün)

# Destek/direnç: bir üst zaman diliminin zigzag tepe/dipleri (context.zigzag_levels), mumlar o zaman
# dilimine çevrilerek hesaplanır. Bir yapı için sadece yapı bittiğinde onaylanmış olanlar gösterilir.
LEVEL_SEC = {"15m": 4 * 3600, "1h": 4 * 3600, "4h": 86400, "1d": 7 * 86400}
LEVEL_NAME = {"15m": "4S", "1h": "4S", "4h": "1G", "1d": "1H"}
LEVEL_MULT = PIVOT_MULT             # dönüş eşiği: o zaman diliminin normal mum boyunun katı
LEVEL_DAYS = {"15m": 60, "1h": 120, "4h": 365, "1d": 1500}    # bu kadar gün içinde oluşmuş seviyeler
LEVEL_BAND = 0.5                    # yapının dışında, boyunun bu katı uzaklığa kadar olan seviyeler
                                    # (15dk'da yapı başına ortalama 3, en çok ~8 seviye)

# Üst zaman dilimi durumu (kartlardaki etiketler): trend (context.trend_labels: 50 mumluk ortalama ve eğimi)
# ve son onaylı zigzag dönüşüne göre bacak yönü (son dönüş dipse yukarı bacak, tepeyse aşağı bacak).
HTF = {"15m": [("4S", 4 * 3600), ("1G", 86400)], "1h": [("4S", 4 * 3600), ("1G", 86400)],
       "4h": [("1G", 86400), ("1H", 7 * 86400)], "1d": [("1H", 7 * 86400)]}
WEEK_ORIGIN = pd.Timestamp("1970-01-05")    # haftalık mumlar Pazartesi başlar (Binance gibi)

# BTC'nin 4 yıllık döngüsü: bir yapı halving'den kaç gün sonra oluştu. Son üç döngüde tepe halving'den
# 526-548 gün, dip 889-924 gün sonra geldi. Bir sonraki halving (~Nisan 2028) olunca listeye eklenmeli.
HALVINGS = np.array([int(pd.Timestamp(x).timestamp()) for x in
                     ("2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20")], np.int64)
CYCLE_WINDOW = 120                  # "aynı döngü dönemi": halving'den geçen gün farkı en fazla bu kadar
API_VERSION = 3                     # yanıt alanları değişince artırılır; web/app.js'teki API_VERSION ile aynı olmalı
MIN_SIM = 80                        # bundan az benzer olaylar gösterilmez: filtre sıkıyken liste alakasızlarla
                                    # dolmasın (filtresiz aramada 50. sonuç genelde %94, en kötü ~%76)
CYCLE_PHASES = ((180, "halving sonrası"), (540, "boğa"), (880, "ayı"), (1100, "ayı dibi"), (10 ** 6, "halving öncesi"))


def cycle_day(ts):
    """Son halving'den geçen gün (ts: unix saniye; dizi de olabilir)."""
    ts = np.asarray(ts, np.int64)
    return (ts - HALVINGS[np.searchsorted(HALVINGS, ts, side="right") - 1]) / 86400


def cycle_phase(day: float) -> str:
    return next(name for limit, name in CYCLE_PHASES if day < limit)
CHUNK = 512                         # adaylar bu büyüklükte paketler halinde çekirdeklere dağıtılır

# Bağlam: yapının içine değil, öncesine ve likiditeye bakar (şekil benzerliğinin görmediği kısım).
CTX_PRE = 2                         # yapıdan önceki bu kat kadar mum incelenir
POOL = 200                          # "şekil + bağlam" sıralamasında, şekle göre en iyi bu kadar olay yeniden sıralanır
CTX_WEIGHT = 0.1                    # sıralama puanı = şekil mesafesi + CTX_WEIGHT x bağlam farkı
# Yürüyen testte (analyze_rank.py, 15dk 2018-2023) hiçbir ağırlık 1 günlük tahmini şekle göre sıralamanın
# üstüne çıkarmadı ama düşürmedi de; arayüz hep bu sıralamayı kullanır, çünkü hikâyesi benzeyen örnekler okunması kolay.


@njit(cache=True)
def _features(h, l, c, s, m, out):
    """s'den başlayan m mumu K birleştirilmiş muma indirger: out[b] = [yüksek, düşük, kapanış]."""
    k = out.shape[0]
    for b in range(k):
        lo = (b * m) // k
        hi = max(((b + 1) * m) // k, lo + 1)
        mx, mn = -np.inf, np.inf
        for t in range(s + lo, s + hi):
            mx = max(mx, h[t])
            mn = min(mn, l[t])
        out[b, 0], out[b, 1], out[b, 2] = mx, mn, c[s + hi - 1]


@njit(cache=True)
def _normalize(x, m, avg_range_at_end):
    """x'i yerinde normalize eder (kapanışlar ortalama 0, std 1) ve yapının döneme göre boyunu döner."""
    k = x.shape[0]
    top, bottom, mean = -np.inf, np.inf, 0.0
    for b in range(k):
        top = max(top, x[b, 0])
        bottom = min(bottom, x[b, 1])
        mean += x[b, 2]
    mean /= k
    var = 0.0
    for b in range(k):
        var += (x[b, 2] - mean) ** 2
    sd = np.sqrt(var / k) + 1e-12
    for b in range(k):
        for f in range(3):
            x[b, f] = (x[b, f] - mean) / sd
    return (top - bottom) / mean / (avg_range_at_end * np.sqrt(m))


@njit(cache=True)
def _euclid(q, x, w):
    tot = 0.0
    for i in range(q.shape[0]):
        for f in range(3):
            d = q[i, f] - x[i, f]
            tot += w[f] * d * d
    return tot / q.shape[0] / w.sum()


@njit(cache=True)
def _dtw(q, x, w, band, prev, cur):
    """Bantlı DTW. prev/cur: k uzunlukta çalışma dizileri (her aday için yeniden ayrılmasın diye).

    Ağırlık toplamına bölme sona bırakılır (sonuç aynı, hücre başına bir bölme eksik).
    """
    k = q.shape[0]
    w0, w1, w2 = w[0], w[1], w[2]
    # İlk satır: sadece soldan gelinebilir.
    acc = 0.0
    for j in range(min(k, band + 1)):
        d0 = q[0, 0] - x[j, 0]; d1 = q[0, 1] - x[j, 1]; d2 = q[0, 2] - x[j, 2]
        acc += w0 * d0 * d0 + w1 * d1 * d1 + w2 * d2 * d2
        prev[j] = acc
    if band + 1 < k:
        prev[band + 1] = np.inf
    for i in range(1, k):
        j0, j1 = max(0, i - band), min(k, i + band + 1)
        q0, q1, q2 = q[i, 0], q[i, 1], q[i, 2]
        left = np.inf
        for j in range(j0, j1):
            d0 = q0 - x[j, 0]; d1 = q1 - x[j, 1]; d2 = q2 - x[j, 2]
            best = prev[j]
            if j > 0:
                if prev[j - 1] < best:
                    best = prev[j - 1]
                if left < best:
                    best = left
            left = w0 * d0 * d0 + w1 * d1 * d1 + w2 * d2 * d2 + best
            cur[j] = left
        if j1 < k:
            cur[j1] = np.inf                          # bir sonraki satır bandın sağ kenarını okur
        prev, cur = cur, prev
    return prev[k - 1] / k / (w0 + w1 + w2)


@njit(parallel=True, cache=True)
def _score(h, l, c, avg_range, starts, lengths, q, q_rel, w, band, use_dtw, size_tol, out):
    """Her (başlangıç, uzunluk) adayının sorguya mesafesi. Boyu uymayan adaylar inf."""
    k = q.shape[0]
    n = len(starts)
    for ci in prange((n + CHUNK - 1) // CHUNK):
        x = np.empty((k, 3))
        prev = np.empty(k)
        cur = np.empty(k)
        for i in range(ci * CHUNK, min(n, (ci + 1) * CHUNK)):
            out[i] = np.inf
            s, m = starts[i], lengths[i]
            ar = avg_range[s + m - 1]
            if ar > 0:                              # NaN (verinin ilk günleri) atlanır
                _features(h, l, c, s, m, x)
                ratio = _normalize(x, m, ar) / q_rel
                if ratio > 1.0 / size_tol and ratio < size_tol:
                    out[i] = _dtw(q, x, w, band, prev, cur) if use_dtw else _euclid(q, x, w)


@njit(cache=True)
def _context(h, l, c, starts, lengths, pre, out):
    """Yapının bağlamı, yapının boyu (en yüksek - en düşük) cinsinden; önceki pre*m muma bakılır.

    out[i, 0]: nereden geldi: o mumların net hareketi (- düşüşten, + yükselişten geldi)
    out[i, 1]: üstteki likidite: o mumların en yükseği yapının tepesinin ne kadar üstünde (- ise yapı onu almış)
    out[i, 2]: alttaki likidite: o mumların en düşüğü yapının dibinin ne kadar altında (- ise yapı onu almış)
    Öncesinde yeterli veri olmayan yapılar NaN.
    """
    for i in range(len(starts)):
        s, m = starts[i], lengths[i]
        a = s - pre * m
        if a < 0:
            out[i, :] = np.nan
            continue
        hi, lo, ph, pl = -np.inf, np.inf, -np.inf, np.inf
        for t in range(s, s + m):
            hi = max(hi, h[t])
            lo = min(lo, l[t])
        for t in range(a, s):
            ph = max(ph, h[t])
            pl = min(pl, l[t])
        r = hi - lo + 1e-12
        out[i, 0] = (c[s - 1] - c[a]) / r
        out[i, 1] = (ph - hi) / r
        out[i, 2] = (lo - pl) / r


def context_distance(q_ctx: np.ndarray, ctx: np.ndarray) -> np.ndarray:
    """Bağlam farkı, 0 (aynı) - 4. Özellikler tanh ile sıkıştırılır: 0 ile 0,5 farkı önemli, 2 ile 3 farkı
    değil. Öncesi bilinmeyen yapılar en uzak sayılır."""
    dc = np.mean((np.tanh(ctx) - np.tanh(q_ctx)) ** 2, axis=-1)
    return np.where(np.isnan(dc), 4.0, dc)


def context_similarity(dc: np.ndarray) -> np.ndarray:
    """Bağlam farkını yüzdeye çevirir (%100 = aynı bağlam)."""
    return np.clip(100 * (1 - np.sqrt(dc) / 2), 0, 100)


def _ctx_json(x: np.ndarray) -> list:
    return [None if np.isnan(v) else round(float(v), 2) for v in x]


@njit(cache=True)
def _distinct(order, start, length, limit):
    """matcher.distinct_events ile aynı: en iyiden başlayarak seçer, zamanı çakışanları atlar."""
    occupied = np.zeros((start + length).max() + 1, np.bool_)
    picks = np.empty(limit, np.int64)
    n = 0
    for k in order:
        s, m = start[k], length[k]
        free = True
        for t in range(s, s + m):
            if occupied[t]:
                free = False
                break
        if free:
            occupied[s:s + m] = True
            picks[n] = k
            n += 1
            if n == limit:
                break
    return picks[:n]


def distinct_events(dist, start, length, limit: int = TOP) -> np.ndarray:
    if len(dist) == 0:
        return np.empty(0, np.int64)
    return _distinct(np.argsort(dist), start, length, limit)


class Engine:
    def __init__(self, interval: str = "1h", symbol: str = "BTCUSDT"):
        self.interval = interval
        self.symbol = symbol
        self.reload()
        # Derleme ve iş parçacıklarının ilk açılışı kullanıcının ilk aramasına yansımasın diye.
        n = len(self.t)
        self.search(n - 100, n, True)
        self.search(n - 100, n, False)

    def reload(self):
        df = load(self.interval, self.symbol)
        self.t = (df.open_time.to_numpy() // 1000).astype(np.int64)
        self.o, self.h, self.l, self.c = (np.ascontiguousarray(df[k].to_numpy(np.float64))
                                          for k in ("open", "high", "low", "close"))
        self.avg_range = (pd.Series((self.h - self.l) / self.c)
                          .rolling(AVG_WINDOW[self.interval], min_periods=24).mean().to_numpy())
        self.step = int(np.median(np.diff(self.t[-100:])))
        t, h, l, c = self._resample(LEVEL_SEC[self.interval])
        lv = context.zigzag_levels(t, h, l, c, LEVEL_SEC[self.interval], LEVEL_MULT)
        self.levels = (lv.price.to_numpy(), lv.kind.to_numpy(), lv.formed.to_numpy(np.int64),
                       lv.confirmed.to_numpy(np.int64))
        self.htf = []
        for name, sec in HTF[self.interval]:
            t, h, l, c = self._resample(sec)
            piv = context.zigzag_levels(t, h, l, c, sec, PIVOT_MULT)
            self.htf.append((name, t + sec, context.trend_labels(pd.Series(c)).to_numpy(),
                             piv.confirmed.to_numpy(np.int64), piv.kind.to_numpy()))

    def _resample(self, sec: int):
        """Mumları sec saniyelik mumlara çevirir: (açılış zamanları, yüksek, düşük, kapanış). Son mum yarım olabilir;
        kullananlar sadece kapanış zamanı (açılış + sec) geçmiş mumlara bakar."""
        bars = (pd.DataFrame({"h": self.h, "l": self.l, "c": self.c}, index=pd.to_datetime(self.t, unit="s"))
                .resample(f"{sec}s", origin=WEEK_ORIGIN if sec == 7 * 86400 else "epoch")
                .agg({"h": "max", "l": "min", "c": "last"}).dropna())
        return (bars.index.values.astype("datetime64[s]").astype(np.int64),
                bars.h.to_numpy(), bars.l.to_numpy(), bars.c.to_numpy())

    def htf_state(self, e: int) -> list:
        """Yapı e. mumda bittiğinde üst zaman dilimlerinin durumu: [ad, trend, son bacağın yönü].
        Sadece kapanmış mumlar ve onaylanmış dönüşler kullanılır."""
        t_end = self.t[e - 1] + self.step
        out = []
        for name, closes, trend, confirmed, kind in self.htf:
            k = int(np.searchsorted(closes, t_end, side="right")) - 1
            p = int(np.searchsorted(confirmed, t_end, side="right")) - 1
            tr = trend[k] if k >= 0 else None
            out.append([name, tr if isinstance(tr, str) else None,      # ortalama oluşmadan önce NaN gelir
                        None if p < 0 else ("yukarı" if kind[p] == "dip" else "aşağı")])
        return out

    def _cycle_json(self, e: int) -> list:
        """Yapı e. mumda bittiğinde döngüdeki yeri: [halving'den geçen gün, dönem adı]."""
        day = float(cycle_day(self.t[e - 1] + self.step))
        return [int(day), cycle_phase(day)]

    def allow_fn(self, q_end: int, cycle: bool, same_trend: bool):
        """Filtre (search'ün allow'u): adayın bittiği an, sorgunun bittiği anla aynı döngü döneminde mi
        (halving'den geçen gün farkı <= CYCLE_WINDOW) ve üst zaman dilimi durumu aynı mı (arayüzdeki
        "Trend: aynı" ile aynı kural: trend aynı, trend yatay değilse son bacak da aynı)."""
        if not (cycle or same_trend):
            return None
        q_day = cycle_day(self.t[q_end - 1] + self.step)
        q_htf = self.htf_state(q_end)

        def allow(starts, lengths):
            t_end = self.t[starts + lengths - 1] + self.step
            ok = np.ones(len(t_end), bool)
            if cycle:
                ok &= np.abs(cycle_day(t_end) - q_day) <= CYCLE_WINDOW
            if same_trend:
                for (_, q_tr, q_leg), (_, closes, trend, confirmed, kind) in zip(q_htf, self.htf):
                    if q_tr is None or q_leg is None:
                        continue                                  # sorgunun durumu bilinmiyor
                    k = np.searchsorted(closes, t_end, side="right") - 1
                    ok &= (k >= 0) & (trend[np.maximum(k, 0)] == q_tr)
                    if q_tr != "yatay":
                        p = np.searchsorted(confirmed, t_end, side="right") - 1
                        ok &= (p >= 0) & ((kind[np.maximum(p, 0)] == "dip") == (q_leg == "yukarı"))
            return ok
        return allow

    def level_lines(self, s: int, e: int) -> list:
        """[s, e) yapısı bittiğinde bilinen üst zaman dilimi tepe/dipleri (sonradan oluşanlar yok), yapının
        çevresindekiler: [fiyat, tür, oluştuğu an, o zamana kadar kırılmış mı]."""
        price, kind, formed, confirmed = self.levels
        t_end = self.t[e - 1] + self.step
        hi, lo = self.h[s:e].max(), self.l[s:e].min()
        band = LEVEL_BAND * (hi - lo)
        sel = np.flatnonzero((confirmed <= t_end) & (formed >= t_end - LEVEL_DAYS[self.interval] * 86400)
                             & (price >= lo - band) & (price <= hi + band))
        out = []
        for i in sel:
            a = int(np.searchsorted(self.t, confirmed[i]))
            if kind[i] == "tepe":
                broken = a < e and self.h[a:e].max() > price[i]
            else:
                broken = a < e and self.l[a:e].min() < price[i]
            out.append([float(price[i]), str(kind[i]), int(formed[i]), bool(broken)])
        return out

    def status(self) -> dict:
        return {"interval": self.interval, "count": len(self.t), "version": API_VERSION,
                "first": int(self.t[0]), "last": int(self.t[-1]) + self.step}

    def pivots(self, now: int) -> list[tuple[int, str]]:
        """now'dan önceki onaylanmış tepe ve dipler (zigzag). Gelecek veriye bakmaz."""
        a = max(0, now - PIVOT_LOOKBACK)
        h, l = self.h[a:now], self.l[a:now]
        thr = PIVOT_MULT * self.avg_range[now - 1]
        piv, direction, hi, lo = [], 0, 0, 0
        for i in range(1, len(h)):
            if direction >= 0 and h[i] >= h[hi]:
                hi = i
            if direction <= 0 and l[i] <= l[lo]:
                lo = i
            if direction == 0:
                if lo < hi and h[hi] >= l[lo] * (1 + thr):
                    piv.append((a + lo, "dip")); direction = 1
                elif hi < lo and l[lo] <= h[hi] * (1 - thr):
                    piv.append((a + hi, "tepe")); direction = -1
            # Onay, uç noktadan sonraki bir mumla gelmeli: tek mumun içinde tepe mi dip mi önce
            # oluştu bilinemez (yoksa büyük tek bir mum hem tepe hem dip sayılır).
            elif direction == 1 and hi < i and l[i] <= h[hi] * (1 - thr):
                piv.append((a + hi, "tepe")); direction = -1; lo = i
            elif direction == -1 and lo < i and h[i] >= l[lo] * (1 + thr):
                piv.append((a + lo, "dip")); direction = 1; hi = i
        return piv

    def groups(self, now: int, piv) -> list[dict]:
        """Son tepe/diplerden başlayan, uzunluğu MIN_LEN..MAX_LEN olan yapılar (en yeniden eskiye)."""
        out = []
        for idx, kind in reversed(piv):
            length = now - idx
            if length < MIN_LEN:
                continue
            if length > MAX_LEN or len(out) == MAX_GROUPS:
                break
            out.append({"start_idx": idx, "length": length, "kind": kind})
        return out

    def context(self, starts, lengths) -> np.ndarray:
        """Yapıların bağlam özellikleri (_context): (n, 3)."""
        out = np.empty((len(starts), 3))
        _context(self.h, self.l, self.c, np.asarray(starts, np.int64), np.asarray(lengths, np.int64),
                 CTX_PRE, out)
        return out

    def _score(self, starts, lengths, q, q_rel, use_dtw):
        out = np.empty(len(starts))
        _score(self.h, self.l, self.c, self.avg_range, starts, lengths, q, q_rel,
               W, BAND, use_dtw, SIZE_TOL, out)
        return out

    def search(self, q_start: int, q_end: int, use_dtw: bool, exact: bool = False,
               end_limit: int | None = None, allow=None):
        """Sorgudan önce bitmiş adayları puanlar: (dist, start, length), elenenler hariç.

        exact=True: her aday denenir (matcher.search ile aynı, foot=0).
        Aksi halde iki aşamalı: seyrek ön tarama + en iyilerin komşuları.
        end_limit: adaylar en geç bu indekste bitmeli (simülasyonda, adayın "sonrası" da karar
        anından önce bitsin diye: end_limit = şu an - ufuk).
        allow: (başlangıçlar, uzunluklar) -> bool dizisi; False olan adaylar elenir (filtre). Ön taramada
        uygulanır ki komşuları denenecek en iyi adaylar filtreden geçenler arasından seçilsin.
        """
        L = q_end - q_start
        last_end = q_start if end_limit is None else min(q_start, end_limit)
        q = np.empty((K, 3))
        _features(self.h, self.l, self.c, q_start, L, q)
        q_rel = _normalize(q, L, self.avg_range[q_end - 1])
        coarse = not exact
        div = STRIDE_DIV[use_dtw]

        starts, lengths = [], []
        for scale in SCALES:
            m = int(round(L * scale))
            n_starts = last_end - m + 1
            if n_starts <= 0:
                continue
            st = np.arange(0, n_starts, max(1, m // div) if coarse else 1)
            starts.append(st)
            lengths.append(np.full(len(st), m))
        if not starts:                              # sorgudan önce yeterli geçmiş yok
            return np.empty(0), np.empty(0, np.int64), np.empty(0, np.int64)
        s, m = np.concatenate(starts), np.concatenate(lengths)
        d = self._score(s, m, q, q_rel, use_dtw)
        if allow is not None:
            d[~allow(s, m)] = np.inf

        if coarse:
            # En iyi ön tarama adaylarının atlanan komşularını da dene.
            ns, nm = [], []
            for k in np.argsort(d)[:REFINE_TOP]:
                step = max(1, m[k] // div)
                if step == 1 or not np.isfinite(d[k]):
                    continue
                nb = np.arange(max(0, s[k] - step + 1), min(last_end - m[k], s[k] + step - 1) + 1)
                nb = nb[nb != s[k]]
                ns.append(nb)
                nm.append(np.full(len(nb), m[k]))
            if ns:
                s2, m2 = np.concatenate(ns), np.concatenate(nm)
                d2 = self._score(s2, m2, q, q_rel, use_dtw)
                if allow is not None:
                    d2[~allow(s2, m2)] = np.inf
                s, m = np.concatenate([s, s2]), np.concatenate([m, m2])
                d = np.concatenate([d, d2])

        keep = np.isfinite(d)
        return d[keep], s[keep], m[keep]

    def _bars(self, a: int, b: int, factor: int, seg: int) -> list:
        """[a, b) mumlarını factor'lük gruplar halinde birleştirir: [zaman, a, y, d, k, bölüm]."""
        if b <= a:
            return []
        first = np.arange(a, b, factor)
        last = np.minimum(first + factor, b) - 1
        hi = np.maximum.reduceat(self.h[a:b], first - a)
        lo = np.minimum.reduceat(self.l[a:b], first - a)
        return [[t, o, h, l, c, seg] for t, o, h, l, c in zip(
            self.t[first].tolist(), self.o[first].tolist(), hi.tolist(), lo.tolist(), self.c[last].tolist())]

    def _window(self, s: int, m: int, now: int, pre: float, max_bars: int, min_after: int = 0,
                min_before: int = 0) -> list:
        """Eşleşmeyi öncesi (pre*m, en az min_before mum) ve sonrası (AFTER_MAX*m, en az min_after mum)
        ile döner; 'şu an'ın ötesine geçmez. max_bars'tan uzunsa mumlar birleştirilir."""
        a = max(0, s - max(int(pre * m), min_before))
        e = min(now, s + m + max(AFTER_MAX * m, min_after))
        factor = max(1, -(-(e - a) // max_bars))
        return (self._bars(a, s, factor, 0) + self._bars(s, s + m, factor, 1)
                + self._bars(s + m, e, factor, 2))

    def detail(self, start_ts: int, m: int, now_ts: int | None) -> list:
        """Karşılaştırma penceresi için bir eşleşmenin ayrıntılı (birleştirilmemiş) mumları."""
        now = len(self.t) if now_ts is None else int(np.searchsorted(self.t, now_ts))
        s = int(np.searchsorted(self.t, start_ts))
        if not 0 <= s < now or m <= 0:
            raise ValueError("Geçersiz eşleşme")
        return self._window(s, m, now, pre=1.0, max_bars=DETAIL_BARS, min_after=DETAIL_MIN_AFTER,
                            min_before=DETAIL_MIN_BEFORE)

    def find(self, now_ts: int | None, use_dtw: bool, sel: tuple[int, int] | None = None,
             by_context: bool = False, cycle: bool = False, same_trend: bool = False) -> dict:
        """'Şu an' için arama. sel=(başlangıç, bitiş) verilirse otomatik gruplar yerine sadece
        kullanıcının seçtiği [başlangıç, bitiş) aralığına benzeyen olaylar aranır.
        by_context: şekle göre en iyi POOL olay, şekil + bağlam puanıyla yeniden sıralanır.
        cycle / same_trend: sadece aynı döngü döneminde / aynı üst zaman dilimi durumunda biten olaylar."""
        t0 = time.perf_counter()
        now = len(self.t) if now_ts is None else int(np.searchsorted(self.t, now_ts))
        now = max(MAX_LEN * 3, min(now, len(self.t)))
        piv = self.pivots(now)
        if sel:
            a, b = (int(np.searchsorted(self.t, x)) for x in sel)
            b = min(b, now)
            if not MIN_SEL <= b - a <= MAX_SEL:
                raise ValueError(f"Seçim {MIN_SEL}-{MAX_SEL} mum arasında olmalı (şu an {max(0, b - a)} mum).")
            groups = [{"start_idx": a, "end_idx": b, "length": b - a, "kind": "secim"}]
        else:
            groups = [{**g, "end_idx": now} for g in self.groups(now, piv)]
        for g in groups:
            d, s, m = self.search(g["start_idx"], g["end_idx"], use_dtw,
                                  allow=self.allow_fn(g["end_idx"], cycle, same_trend))
            pool = distinct_events(d, s, m, POOL)           # şekle göre sıralı
            pool = pool[100 * (1 - d[pool] / 2) >= MIN_SIM]
            q_ctx = self.context([g["start_idx"]], [g["length"]])[0]
            ctx = self.context(s[pool], m[pool])
            dc = context_distance(q_ctx, ctx)
            order = np.argsort(d[pool] + CTX_WEIGHT * dc, kind="stable") if by_context else np.arange(len(pool))
            order = order[:TOP]
            picks = pool[order]
            # Benzerlik: kapanış şekilleri için korelasyona denk (%100 = birebir aynı şekil).
            sim = np.clip(100 * (1 - d[picks] / 2), 0, 100)
            ctx_sim = context_similarity(dc[order])
            g["levels"] = self.level_lines(g["start_idx"], g["end_idx"])
            g["htf"] = self.htf_state(g["end_idx"])
            g["cycle"] = self._cycle_json(g["end_idx"])
            g["start"] = int(self.t[g.pop("start_idx")])
            g["end"] = int(self.t[g.pop("end_idx") - 1]) + self.step
            g["context"] = _ctx_json(q_ctx)
            g["results"] = [{"rank": r + 1, "start": int(self.t[s[k]]), "length": int(m[k]),
                             "similarity": round(float(sim[r]), 1),
                             "context": round(float(ctx_sim[r])), "ctx": _ctx_json(ctx[order[r]]),
                             "levels": self.level_lines(int(s[k]), int(s[k] + m[k])),
                             "htf": self.htf_state(int(s[k] + m[k])),
                             "cycle": self._cycle_json(int(s[k] + m[k])),
                             "bars": self._window(int(s[k]), int(m[k]), now, pre=0.5, max_bars=CARD_BARS)}
                            for r, k in enumerate(picks)]
        a = max(0, now - MAIN_BARS)
        return {
            "interval": self.interval,
            "now": int(self.t[now - 1]) + self.step,
            "method": "dtw" if use_dtw else "euclid",
            "rank": "context" if by_context else "shape",
            "version": API_VERSION,
            "filter": "cycle_trend" if cycle and same_trend else "cycle" if cycle else "trend" if same_trend else "none",
            "level_tf": LEVEL_NAME[self.interval],
            "main": self._bars(a, now, 1, 0),
            "pivots": [[int(self.t[i]), kind, float(self.h[i] if kind == "tepe" else self.l[i])]
                       for i, kind in piv if i >= a],
            "groups": groups,
            "elapsed": round(time.perf_counter() - t0, 2),
        }
