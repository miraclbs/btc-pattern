"""rank_sim.py havuzlarından sıralama yöntemlerini karşılaştırır: hangisi sonrasını daha iyi tahmin ediyor?

Her karar anında her yöntem TOP_K analog seçer; analogların sonrasından bir tahmin çıkar ve aynı anda
gerçekte ne olduğuyla karşılaştırılır. Yöntemler:
  dtw, euclid        : şekle göre ilk TOP_K (arayüzdeki "Şekil" sıralaması)
  ... +bağlam λ      : havuz, şekil mesafesi + λ x bağlam farkı ile yeniden sıralanır ("Şekil + bağlam")
  ... sadece bağlam  : havuz sadece bağlam farkıyla sıralanır (teşhis için)
  rastgele           : sorguyla aynı uzunlukta rastgele geçmiş pencereler (kontrol grubu: bilgi yok)
  rastgele aynı boy  : aynı uzunlukta ve döneme göre boyu en yakın rastgele pencereler (kontrol grubu:
                       şekil yok, sadece boy). Bir yöntem bunu geçemiyorsa sinyali şekilden değil boydan geliyordur.

Ölçütler, karar anları arasında sıra korelasyonu (0 = bilgi yok, büyük = iyi):
  IC 8s, IC 1g : analogların ortalama ileri getirisi (8 saat / 1 gün sonra, normal mum boyu cinsinden)
                 ile gerçek getiri
  iki taraf    : analogların sonrasında "yapının hem tepesi hem dibi alındı" oranı ile gerçekte alınıp
                 alınmadığı (sonrası: yapı uzunluğunun 3 katı, en fazla HORIZON mum)
  işlem        : simulate.py'deki 12 stop/hedef/yön ayarında tahmini R ile gerçek R (ortalama). Masrafsız:
                 masraf karar anına göre değişip tahmine de gerçeğe de aynı eklendiği için sahte korelasyon katar.
Köşeli parantez: aylık blok bootstrap ile %90 aralık. "dtw'ye göre": eşleştirilmiş farkın %90 aralığı.

Önceden konan karar kuralı: birincil ölçüt IC 1g. Bir yöntem dev'de dtw'yi geçerse (farkın %90 aralığı
sıfırın üstünde) val'de bir kez denenir; orada da geçerse arayüzde varsayılan olur.

Kullanım: .venv/Scripts/python analyze_rank.py [dev|val]
"""
import sys

import numpy as np
import pandas as pd
from numba import njit

import engine
from engine import _features, _normalize
from simulate import HORIZON, INTERVAL, RESULTS, STOP_MULTS, TARGETS, _outcomes

TOP_K = 50
LAMBDAS = (0.03, 0.1, 0.3, 1.0)
FWD = {"IC 8s": 32, "IC 1g": 96}          # ileri getiri ufku (15dk mum)
N_RANDOM = 400                            # "rastgele aynı boy" için karar anı başına denenen pencere
N_BOOT = 1000
rng = np.random.default_rng(0)


@njit(cache=True)
def _rel_size(h, l, c, avg_range, starts, lengths, k):
    """Yapıların döneme göre boyu (engine._normalize ile aynı ölçü)."""
    out = np.empty(len(starts))
    x = np.empty((k, 3))
    for i in range(len(starts)):
        s, m = starts[i], lengths[i]
        _features(h, l, c, s, m, x)
        out[i] = _normalize(x, m, avg_range[s + m - 1])
    return out


def rel_size(eng, starts, lengths) -> np.ndarray:
    return _rel_size(eng.h, eng.l, eng.c, eng.avg_range, starts.astype(np.int64), lengths.astype(np.int64), engine.K)


@njit(cache=True)
def _both_sides(h, l, starts, lengths, after_max):
    """Yapı bittikten sonra (3 x uzunluk, en fazla after_max mum) hem tepesi hem dibi geçildi mi."""
    out = np.zeros(len(starts), np.bool_)
    for i in range(len(starts)):
        s, m = starts[i], lengths[i]
        hi, lo = -np.inf, np.inf
        for t in range(s, s + m):
            hi = max(hi, h[t])
            lo = min(lo, l[t])
        up, dn = False, False
        for t in range(s + m, min(s + m + min(3 * m, after_max), len(h))):
            up = up or h[t] > hi
            dn = dn or l[t] < lo
        out[i] = up and dn
    return out


def outcomes(eng, starts, lengths) -> dict:
    """Yapı bittikten sonra olanlar: ileri getiriler, iki taraf, 12 işlem sonucu (masrafsız R)."""
    starts, lengths = starts.astype(np.int64), lengths.astype(np.int64)
    ends = starts + lengths
    o, avg = eng.o, eng.avg_range
    f = {name: (eng.c[ends + H - 1] - o[ends]) / (o[ends] * avg[ends - 1]) for name, H in FWD.items()}
    f["iki taraf"] = _both_sides(eng.h, eng.l, starts, lengths, HORIZON).astype(float)
    R, _, _ = _outcomes(o, eng.h, eng.l, eng.c, avg, ends, STOP_MULTS, TARGETS, HORIZON)
    f["R"] = R.reshape(len(ends), -1)
    return f


def colcorr(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Sütun sütun Pearson korelasyonu (sıraya çevrilmiş verilerde Spearman'a denk)."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    return (X * Y).sum(0) / np.sqrt((X * X).sum(0) * (Y * Y).sum(0))


def main(period: str):
    eng = engine.Engine(INTERVAL)
    z = np.load(RESULTS / f"pool_{INTERVAL}_{period}.npz")
    now, q_start = z["now"], z["q_start"]
    ok = np.all([(z[f"{k}_start"] >= 0).sum(1) >= TOP_K for k in ("dtw", "euclid")], axis=0)
    now, q_start = now[ok], q_start[ok]
    n, L = len(now), now - q_start
    print(f"{INTERVAL} {period}: {n} karar anı, analog sayısı {TOP_K}, havuz {z['dtw_start'].shape[1]}\n")

    # Rastgele kontrol grupları: sorguyla aynı uzunlukta, sonrası karar anından önce biten pencereler.
    # "Aynı boy": N_RANDOM pencere içinden döneme göre boyu sorgununkine en yakın TOP_K tanesi.
    lo = engine.CTX_PRE * L
    hi = now - HORIZON - L
    cand = (lo[:, None] + rng.random((n, N_RANDOM)) * (hi - lo)[:, None]).astype(np.int64)
    r_len = np.repeat(L[:, None], TOP_K, 1)
    cand_rel = rel_size(eng, cand.ravel(), np.repeat(L, N_RANDOM)).reshape(n, N_RANDOM)
    q_rel = rel_size(eng, q_start, L)
    near = np.argsort(np.abs(np.log(cand_rel / q_rel[:, None])), axis=1)[:, :TOP_K]
    r_start = cand[:, :TOP_K]
    rs_start = np.take_along_axis(cand, near, 1)

    # Bütün adayların (havuzlar + rastgele) sonuçları bir kez hesaplanır.
    starts = np.concatenate([z["dtw_start"][ok].ravel(), z["euclid_start"][ok].ravel(),
                             r_start.ravel(), rs_start.ravel()])
    lens = np.concatenate([z["dtw_length"][ok].ravel(), z["euclid_length"][ok].ravel(),
                           r_len.ravel(), r_len.ravel()])
    valid = starts >= 0
    keys = np.where(valid, starts.astype(np.int64) * 1000 + lens, -1)
    uniq, inv = np.unique(keys[valid], return_inverse=True)
    u_s, u_m = uniq // 1000, uniq % 1000
    print(f"{len(uniq)} farklı aday için sonuçlar hesaplanıyor...", flush=True)
    feat = outcomes(eng, u_s, u_m)
    u_ctx = eng.context(u_s, u_m)
    gidx = np.full(len(keys), -1, np.int64)
    gidx[valid] = inv
    P = z["dtw_start"].shape[1]
    pool_idx = {"dtw": gidx[:n * P].reshape(n, P), "euclid": gidx[n * P:2 * n * P].reshape(n, P)}
    rand_idx = gidx[2 * n * P:2 * n * P + n * TOP_K].reshape(n, TOP_K)
    rand_size_idx = gidx[2 * n * P + n * TOP_K:].reshape(n, TOP_K)

    # Gerçekte olan (karar anından sonra).
    act = outcomes(eng, q_start, L)
    q_ctx = eng.context(q_start, L)

    # Yöntemler: her biri karar anı başına TOP_K aday seçer.
    variants = {"rastgele": rand_idx, "rastgele aynı boy": rand_size_idx}
    info = {"rastgele": (np.nan, np.nan), "rastgele aynı boy": (np.nan, np.nan)}
    for name in ("dtw", "euclid"):
        idx = pool_idx[name]
        dist = np.where(idx >= 0, z[f"{name}_dist"][ok], np.inf)
        dc = np.where(idx >= 0, engine.context_distance(q_ctx[:, None, :], u_ctx[np.maximum(idx, 0)]), np.inf)
        scores = {name: dist, **{f"{name} +bağlam {lam:g}": dist + lam * dc for lam in LAMBDAS},
                  f"{name} sadece bağlam": dc}
        for vname, sc in scores.items():
            sel = np.argsort(sc, axis=1, kind="stable")[:, :TOP_K]
            variants[vname] = np.take_along_axis(idx, sel, 1)
            d_sel, dc_sel = np.take_along_axis(dist, sel, 1), np.take_along_axis(dc, sel, 1)
            info[vname] = (np.mean(100 * (1 - d_sel / 2)), np.mean(engine.context_similarity(dc_sel)))

    # Tahmin / gerçek çiftleri, sıraya çevrilmiş halde: her yöntem için 3 ölçüt + 12 işlem ayarı.
    names = list(variants)
    est_cols, act_cols = [], []
    for v in names:
        sel = variants[v]
        for k in (*FWD, "iki taraf"):
            est_cols.append(feat[k][sel].mean(1))
            act_cols.append(act[k])
        R = feat["R"][sel].mean(1)
        for j in range(R.shape[1]):
            est_cols.append(R[:, j])
            act_cols.append(act["R"][:, j])
    rank = lambda cols: np.column_stack([pd.Series(c).rank().to_numpy() for c in cols])
    E, A = rank(est_cols), rank(act_cols)
    per = 3 + feat["R"].shape[1]

    def metrics(rows=slice(None)) -> np.ndarray:
        """(yöntem, [IC 8s, IC 1g, iki taraf, işlem])"""
        r = colcorr(E[rows], A[rows]).reshape(len(names), per)
        return np.column_stack([r[:, :3], r[:, 3:].mean(1)])

    point = metrics()
    months = pd.to_datetime(eng.t[now], unit="s").to_period("M").astype(str).to_numpy()
    groups = [np.flatnonzero(months == mth) for mth in np.unique(months)]
    boot = np.empty((N_BOOT, *point.shape))
    for b in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        boot[b] = metrics(np.concatenate([groups[i] for i in pick]))
    lo_ci, hi_ci = np.percentile(boot, [5, 95], axis=0)
    base = names.index("dtw")
    diff = boot[:, :, 1] - boot[:, base:base + 1, 1]
    d_lo, d_hi = np.percentile(diff, [5, 95], axis=0)

    rows = []
    for i, v in enumerate(names):
        ci = lambda j: f"{point[i, j]:+.3f} [{lo_ci[i, j]:+.3f}, {hi_ci[i, j]:+.3f}]"
        shape_pct, ctx_pct = info[v]
        rows.append({"yöntem": v, "şekil %": "—" if np.isnan(shape_pct) else f"{shape_pct:.1f}",
                     "bağlam %": "—" if np.isnan(ctx_pct) else f"{ctx_pct:.0f}",
                     "IC 8s": ci(0), "IC 1g": ci(1),
                     "dtw'ye göre (IC 1g)": "—" if i == base else f"[{d_lo[i]:+.3f}, {d_hi[i]:+.3f}]",
                     "iki taraf": ci(2), "işlem": ci(3)})
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False))

    # "İki taraf da alınır" tezi: analoglar ne kadar çok "iki taraf" diyorsa gerçekte de öyle mi?
    print(f"\niki taraf, temel oran: %{100 * act['iki taraf'].mean():.0f}. Analogların tahminine göre beşe bölünce gerçekleşen:")
    for v in ("dtw", f"dtw +bağlam {engine.CTX_WEIGHT:g}", "rastgele", "rastgele aynı boy"):
        est = feat["iki taraf"][variants[v]].mean(1)
        q = pd.qcut(est, 5, labels=False, duplicates="drop")
        cal = pd.DataFrame({"t": est, "g": act["iki taraf"]}).groupby(q).mean()
        print(f"  {v:<18}" + "  ".join(f"%{100 * a:.0f}->%{100 * b:.0f}" for a, b in zip(cal.t, cal.g)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
