'use strict';

const LC = LightweightCharts;
const $ = (s) => document.querySelector(s);

const C = {
  up: '#22c55e', down: '#e5e7eb',
  upDim: 'rgba(34,197,94,.28)', downDim: 'rgba(229,231,235,.2)',     // öncesi
  upPost: 'rgba(34,197,94,.6)', downPost: 'rgba(229,231,235,.5)',    // sonrası: asıl incelenen kısım
  band: 'rgba(129,140,248,.10)', accent: '#818cf8', pivot: '#64748b',
};
const TF = {
  '15m': { chip: '15dk', name: '15 dakikalık', sec: 900 },
  '1h': { chip: '1S', name: 'saatlik', sec: 3600 },
  '4h': { chip: '4S', name: '4 saatlik', sec: 14400 },
  '1d': { chip: '1G', name: 'günlük', sec: 86400 },
};
const AFTER_OPTIONS = [0, 1, 3, 5, 10];   // "sonrası": yapı uzunluğunun katı (engine.AFTER_MAX = 10)
const COUNT_OPTIONS = [10, 20, 50];       // gösterilecek sonuç sayısı (engine.TOP = 50)
const MIN_SEL = 10;   // engine.MIN_SEL / MAX_SEL ile aynı
const MAX_SEL = 600;

// Grafik kütüphanesi saatleri UTC gösterir; yerel saatle görünsün diye kaydırıyoruz.
const OFFSET = -new Date().getTimezoneOffset() * 60;
const local = (t) => t + OFFSET;

const fmtDay = new Intl.DateTimeFormat('tr-TR', { day: 'numeric', month: 'short', year: 'numeric' });
const fmtHour = new Intl.DateTimeFormat('tr-TR', { hour: '2-digit', minute: '2-digit' });
const fmtShort = new Intl.DateTimeFormat('tr-TR', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
const fmtShortDay = new Intl.DateTimeFormat('tr-TR', { day: 'numeric', month: 'short', year: 'numeric' });
const fmtDayMonth = new Intl.DateTimeFormat('tr-TR', { day: 'numeric', month: 'short' });
const daily = () => state.tf === '1d';
const day = (t) => fmtDay.format(new Date(t * 1000));
const hour = (t) => fmtHour.format(new Date(t * 1000));
const short = (t) => (daily() ? fmtShortDay : fmtShort).format(new Date(t * 1000));
const when = (t) => (daily() ? day(t) : `${day(t)} ${hour(t)}`);
const num = (x, d = 1) => x.toLocaleString('tr-TR', { minimumFractionDigits: d, maximumFractionDigits: d });
const kindText = (g) => ({ tepe: 'Tepeden', dip: 'Dipten', secim: 'Seçim' })[g.kind];
/** Benzerlik yüzdesine göre renk tonu: lo ve altı kırmızı, lo + span ve üstü yeşil. */
const hueOf = (v, lo, span) => Math.round(30 + 115 * Math.max(0, Math.min(1, (v - lo) / span)));

/** Bağlam (engine._context, yapının boyu cinsinden): [nereden geldi, üstteki likidite, alttaki likidite]. */
function ctxText(c) {
  if (!c || c.includes(null)) return '';
  const side = (v, name) => (v >= 0 ? `${name} ${num(v)}×` : `${name} alınmış`);
  return `önce ${c[0] < 0 ? '↓' : '↑'}${num(Math.abs(c[0]))}× · ${side(c[1], 'üstte')} · ${side(c[2], 'altta')}`;
}
/** Üst zaman dilimi durumu (engine.htf_state): [ad, trend, son bacak]. Örn. "4S ↑ düzeltme". */
function htfText([name, trend, leg]) {
  if (!trend || !leg) return `${name} ?`;
  const arrow = { yukarı: '↑', aşağı: '↓', yatay: '→' }[trend];
  const phase = trend === 'yatay' ? (leg === 'yukarı' ? 'yükselen bacak' : 'düşen bacak')
    : (trend === leg ? 'trend yönünde' : 'düzeltme');
  return `${name} ${arrow} ${phase}`;
}
function htfTitle([name, trend, leg]) {
  if (!trend || !leg) return `${name}: yeterli geçmiş yok`;
  const word = { yukarı: 'yükseliş', aşağı: 'düşüş', yatay: 'yatay' }[trend];
  return `${name}: ${trend === 'yatay' ? 'trend yok (yatay)' : `${word} trendi`}, `
    + `son dönüşten beri fiyat ${leg === 'yukarı' ? 'yükseliyor' : 'düşüyor'}`;
}
/** Etiket kutucukları; ref verilirse (seçili yapının durumu) aynı olanlar vurgulanır. */
function htfChips(list, ref, cls = '') {
  return (list || []).map((x, i) => {
    const same = ref && ref[i] && x[1] === ref[i][1] && x[2] === ref[i][2];
    return `<span class="${cls}${same ? ' same' : ''}" title="${htfTitle(x)}">${htfText(x)}</span>`;
  }).join('');
}
const CTX_HELP ='Önce: yapıdan önceki 2 kat uzunlukta fiyat ne kadar düşmüş (↓) ya da yükselmiş (↑). '
  + 'Üstte / altta: o dönemin en yüksek / en düşük noktası yapının tepesinin / dibinin ne kadar ötesinde '
  + '(alınmamış likidite); "alınmış": yapı o seviyeyi geçmiş. Hepsi yapının boyu cinsinden.';
/** n mumun kabaca ne kadar zaman ettiği: "~11 gün", "~18 saat". */
function span(n) {
  const hours = (n * TF[state.tf].sec) / 3600;
  return hours >= 48 ? `~${Math.round(hours / 24)} gün` : `~${Math.max(1, Math.round(hours))} saat`;
}

function loadPrefs() {
  try { return JSON.parse(localStorage.getItem('btc-pattern') || '{}'); } catch { return {}; }
}
function savePrefs() {
  try {
    localStorage.setItem('btc-pattern', JSON.stringify({
      tf: state.tf, mode: state.mode, method: state.method, rank: state.rank, after: state.after, count: state.count,
      levels: state.levels,
      now: $('#nowInput').value,
    }));
  } catch { /* gizli pencere vb. */ }
}

const prefs = loadPrefs();
const state = {
  tf: TF[prefs.tf] ? prefs.tf : '1h',
  mode: prefs.mode || 'live',
  method: prefs.method || 'dtw',
  rank: prefs.rank === 'context' ? 'context' : 'shape',
  after: AFTER_OPTIONS.includes(prefs.after) ? prefs.after : (prefs.after === false ? 0 : 3),
  count: COUNT_OPTIONS.includes(prefs.count) ? prefs.count : 50,
  levels: prefs.levels !== false,   // karşılaştırmada üst zaman dilimi tepe/dipleri
  drawing: false,                   // karşılaştırmada tıklayınca yatay çizgi

  status: null,
  data: null,
  group: 0,
  modal: -1,
  busy: false,
  selection: null,   // {start, end}: kullanıcının seçtiği aralık (unix saniye, end hariç)
  selecting: false,
};
let mainChart = null;
let gridCharts = [];
let modalCharts = [];
let drag = null;
const detailCache = new Map();   // karşılaştırma penceresi için çekilen ayrıntılı mumlar
const userLines = new Map();     // kullanıcının çizdiği yatay çizgiler: grafik anahtarı -> fiyatlar

/* ---------- Grafik yardımcıları ---------- */

function makeChart(el, { axes = false, interactive = false, logo = false } = {}) {
  const line = { visible: interactive, labelVisible: interactive, color: '#475569', style: 2 };
  return LC.createChart(el, {
    autoSize: true,
    layout: {
      background: { type: 'solid', color: 'transparent' }, textColor: '#8b93a7',
      fontFamily: 'Inter, system-ui, sans-serif', fontSize: 11, attributionLogo: logo,
    },
    localization: { locale: 'tr-TR' },
    grid: { vertLines: { visible: false }, horzLines: { color: axes ? 'rgba(148,163,184,.07)' : 'rgba(0,0,0,0)' } },
    rightPriceScale: { visible: axes, borderVisible: false, scaleMargins: { top: 0.08, bottom: 0.08 } },
    timeScale: { visible: axes, borderVisible: false, timeVisible: !daily(), secondsVisible: false },
    crosshair: { vertLine: line, horzLine: line },
    handleScroll: interactive,
    handleScale: interactive,
  });
}

/** Küçük grafiklerde mumlar okunabilsin diye ardışık mumları birleştirir (bölüm sınırlarını korur). */
function aggregate(bars, maxBars) {
  const f = Math.ceil(bars.length / maxBars);
  if (f <= 1) return bars;
  const out = [];
  for (let i = 0; i < bars.length;) {
    const seg = bars[i][5];
    let j = i, hi = -Infinity, lo = Infinity;
    for (; j < bars.length && j < i + f && bars[j][5] === seg; j++) {
      hi = Math.max(hi, bars[j][2]);
      lo = Math.min(lo, bars[j][3]);
    }
    out.push([bars[i][0], bars[i][1], hi, lo, bars[j - 1][4], seg]);
    i = j;
  }
  return out;
}

/** Yapının bittiği andan sonra state.after katı kadarını bırakır (bölüm 2 = sonrası). */
function trimAfter(bars, start, length) {
  const limit = start + (1 + state.after) * length * TF[state.tf].sec;
  return bars.filter((b) => b[5] !== 2 || (state.after > 0 && b[0] < limit));
}

/** bars: [zaman, açılış, yüksek, düşük, kapanış, bölüm]; bölüm 0 = öncesi, 1 = yapı, 2 = sonrası. */
function drawCandles(chart, bars, { maxBars = 0 } = {}) {
  const rows = maxBars ? aggregate(bars, maxBars) : bars;
  // Yapının arkasındaki şerit: sadece yapı mumlarında değeri olan, ölçeği 0-1'e sabit bir alan.
  const band = chart.addAreaSeries({
    priceScaleId: 'band', lineVisible: false, lineColor: 'rgba(0,0,0,0)',
    topColor: C.band, bottomColor: C.band,
    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
    autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 1 } }),
  });
  chart.priceScale('band').applyOptions({ visible: false, scaleMargins: { top: 0, bottom: 0 } });
  band.setData(rows.map((b) => (b[5] === 1 ? { time: local(b[0]), value: 1 } : { time: local(b[0]) })));
  const series = chart.addCandlestickSeries({
    priceLineVisible: false, lastValueVisible: false, borderVisible: false,
    priceFormat: { type: 'price', precision: 0, minMove: 1 },
  });
  series.setData(rows.map(([t, o, h, l, c, seg]) => {
    const up = c >= o;
    const color = seg === 1 ? (up ? C.up : C.down) : seg === 2 ? (up ? C.upPost : C.downPost) : (up ? C.upDim : C.downDim);
    return { time: local(t), open: o, high: h, low: l, close: c, color, wickColor: color, borderColor: color };
  }));
  chart.timeScale().fitContent();
  return series;
}

/** Üst zaman dilimi tepe/dipleri: oluştuğu andan (ya da grafiğin başından) sona kadar kesikli çizgi.
 *  levels: [fiyat, tür, oluştuğu an, kırılmış mı]. Ölçeği etkilemez. */
function drawLevels(chart, bars, levels, tf) {
  if (!state.levels || !levels || !bars.length) return;
  const last = bars[bars.length - 1][0];
  levels.forEach(([price, kind, formed, broken]) => {
    // Çizgi, grafikteki bir mumdan başlasın (birleştirilmiş mumlarda boşluk açılmasın).
    const start = (bars.find((b) => b[0] >= formed) || bars[bars.length - 1])[0];
    const alpha = broken ? 0.4 : 0.85;
    const line = chart.addLineSeries({
      color: kind === 'tepe' ? `rgba(249,168,212,${alpha})` : `rgba(134,239,172,${alpha})`,
      lineWidth: 1, lineStyle: broken ? LC.LineStyle.Dotted : LC.LineStyle.Dashed,
      priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: false,
      title: `${tf} ${kind} ${fmtDayMonth.format(new Date(formed * 1000))}`,
      autoscaleInfoProvider: () => null,
    });
    line.setData(start === last ? [{ time: local(last), value: price }]
      : [{ time: local(start), value: price }, { time: local(last), value: price }]);
  });
}

/** Çizim açıkken tıklanan fiyata yatay çizgi ekler; var olan çizginin üstüne tıklanırsa siler. */
function bindDrawing(chart, series, key) {
  const lines = new Map();   // fiyat -> IPriceLine
  const add = (p) => lines.set(p, series.createPriceLine({
    price: p, color: C.accent, lineWidth: 1, lineStyle: LC.LineStyle.Solid, axisLabelVisible: true, title: '',
  }));
  (userLines.get(key) || []).forEach(add);
  chart.subscribeClick((param) => {
    if (!state.drawing || !param.point) return;
    const y = param.point.y;
    const hit = [...lines.keys()].find((p) => Math.abs(series.priceToCoordinate(p) - y) < 6);
    if (hit !== undefined) {
      series.removePriceLine(lines.get(hit));
      lines.delete(hit);
    } else {
      const p = series.coordinateToPrice(y);
      if (p !== null) add(Math.round(p));
    }
    userLines.set(key, [...lines.keys()]);
  });
}

function dispose(list) {
  list.forEach((c) => c.remove());
  return [];
}

/** Bir mumun yapı içindeki yeri: 0 = öncesi, 1 = yapı, 2 = sonrası. */
const segOf = (t, g) => (t < g.start ? 0 : t < g.end ? 1 : 2);
const barStep = () => state.data.main[1][0] - state.data.main[0][0];

/* ---------- Çizim ---------- */

function render({ keepView = false } = {}) {
  const d = state.data;
  $('#nowLabel').textContent = `${state.mode === 'live' ? 'Canlı' : 'Geçmiş tarih'} · ${TF[d.interval].name}`
    + ` · şu an ${when(d.now)}`;
  $('#elapsed').textContent = `${num(d.elapsed)} sn · ${d.method === 'dtw' ? 'DTW' : 'Birebir'}`
    + (d.rank === 'context' ? ' + bağlam' : '');
  renderGroups();
  renderMain({ keepView });
  renderGrid();
}

function renderGroups() {
  const el = $('#groups');
  el.innerHTML = '';
  state.data.groups.forEach((g, i) => {
    const b = document.createElement('button');
    b.className = 'group' + (i === state.group ? ' on' : '');
    b.setAttribute('role', 'tab');
    b.setAttribute('aria-selected', i === state.group);
    const key = state.selection ? '' : `<kbd>${i + 1}</kbd>`;
    b.innerHTML = `<span class="k">${key}<span class="tag-${g.kind}">${kindText(g)}</span></span>`
      + `<b>${g.length} mum</b><small>${short(g.start)} itibarıyla</small>`;
    b.onclick = () => selectGroup(i);
    el.append(b);
  });
  if (state.selection) {
    const b = document.createElement('button');
    b.className = 'group clear';
    b.textContent = '✕ Seçimi kaldır';
    b.title = 'Otomatik tepe/dip gruplarına dön';
    b.onclick = clearSelection;
    el.append(b);
  }
}

function renderMain({ keepView = false } = {}) {
  const d = state.data;
  const g = d.groups[state.group];
  const view = keepView && mainChart ? mainChart.timeScale().getVisibleRange() : null;
  if (mainChart) mainChart.remove();
  mainChart = makeChart($('#mainChart'), { axes: true, interactive: true, logo: true });
  const bars = d.main.map((b) => [b[0], b[1], b[2], b[3], b[4], g ? segOf(b[0], g) : 0]);
  const series = drawCandles(mainChart, bars);
  const start = g && g.kind !== 'secim' ? g.start : null;
  series.setMarkers(d.pivots.map(([t, kind]) => ({
    time: local(t),
    position: kind === 'tepe' ? 'aboveBar' : 'belowBar',
    shape: kind === 'tepe' ? 'arrowDown' : 'arrowUp',
    color: t === start ? C.accent : C.pivot,
    text: t === start ? (kind === 'tepe' ? 'Tepe' : 'Dip') : '',
  })));
  if (view) {
    mainChart.timeScale().setVisibleRange(view);
  } else if (g) {
    // Seçili yapı ve biraz öncesi görünsün.
    const i0 = bars.findIndex((b) => b[0] >= g.start);
    const pad = Math.max(30, Math.round((bars.length - i0) * 0.6));
    mainChart.timeScale().setVisibleLogicalRange({ from: Math.max(0, i0 - pad), to: bars.length + 3 });
  }
}

/** Gösterilen sonuçlar: seçili grubun ilk state.count tanesi. */
const shownResults = () => state.data.groups[state.group].results.slice(0, state.count);

let cardObserver = null;

function renderGrid() {
  gridCharts = dispose(gridCharts);
  if (cardObserver) cardObserver.disconnect();
  const grid = $('#grid');
  grid.innerHTML = '';
  const g = state.data.groups[state.group];
  if (!g) {
    $('#queryHtf').innerHTML = '';
    $('#groupTitle').textContent = 'Sonuç yok';
    grid.innerHTML = '<div class="empty">Bu an için uygun bir tepe/dip bulunamadı. Başka bir tarih dene ya da grafikte bir alan seç.</div>';
    return;
  }
  const results = shownResults();
  $('#queryHtf').innerHTML = `<b>${g.kind === 'secim' ? 'Seçtiğin yapı' : 'Şu anki yapı'}:</b>${htfChips(g.htf, null, 'query')}`
    + '<small>Kartlarda seninkiyle aynı olan durum yeşil</small>';
  $('#groupTitle').textContent = g.kind === 'secim'
    ? `Seçtiğin ${g.length} mumluk yapıya en benzer ${results.length} olay`
    : `${kindText(g)} başlayan ${g.length} mumluk yapıya en benzer ${results.length} olay`;
  if (!results.length) {
    grid.innerHTML = '<div class="empty">Bu yapıdan önce karşılaştırılacak yeterli geçmiş yok.</div>';
    return;
  }
  // 50 kartın grafiği birden çizilmesin: kart ekrana yaklaşınca çizilir.
  cardObserver = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (!en.isIntersecting) return;
      cardObserver.unobserve(en.target);
      const r = results[Number(en.target.dataset.i)];
      const chart = makeChart(en.target.querySelector('.mini'));
      drawCandles(chart, trimAfter(r.bars, r.start, r.length), { maxBars: 80 });
      gridCharts.push(chart);
    });
  }, { rootMargin: '400px 0px' });
  results.forEach((r, i) => {
    const card = document.createElement('article');
    card.className = 'card';
    card.tabIndex = 0;
    card.dataset.i = i;
    if (i < 10) card.style.animationDelay = `${i * 25}ms`;
    else card.style.animation = 'none';     // uzun listede beliriş animasyonu sadece ilk satırlarda
    card.title = `Bu yapı, seçili yapının ${num(r.length / g.length)} katı sürede oluşmuş`;
    const sub = daily() ? `${r.length} mum` : `${hour(r.start)} · ${r.length} mum`;
    const ctx = ctxText(r.ctx);
    card.innerHTML = `<header><span class="rank">${r.rank}</span>`
      + `<div class="when"><b>${day(r.start)}</b><small>${sub}</small></div>`
      + `<div class="scores"><span class="sim" style="--h:${hueOf(r.similarity, 85, 13)}" title="Şekil benzerliği">%${num(r.similarity)}</span>`
      + `<small class="ctx" style="--h:${hueOf(r.context, 40, 50)}" title="${ctx ? `Bu olay: ${ctx}\n\n${CTX_HELP}` : CTX_HELP}">`
      + `bağlam %${r.context}</small></div></header>`
      + `<div class="htf">${htfChips(r.htf, g.htf)}</div><div class="mini"></div>`;
    card.onclick = () => openModal(i);
    card.onkeydown = (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); openModal(i); }
    };
    grid.append(card);
    cardObserver.observe(card);
  });
}

function renderSkeleton() {
  gridCharts = dispose(gridCharts);
  $('#grid').innerHTML = Array.from({ length: 10 },
    () => '<div class="card skeleton"><div class="bar"></div><div class="block"></div></div>').join('');
}

function selectGroup(i) {
  if (!state.data || i === state.group || i >= state.data.groups.length) return;
  state.group = i;
  renderGroups();
  renderMain();
  renderGrid();
}

/* ---------- Alan seçimi ---------- */

function startSelecting() {
  if (!state.data || state.busy) return;
  state.selecting = true;
  $('#mainWrap').classList.add('selecting');
  $('#selHint').hidden = false;
  $('#selectBtn').classList.add('on');
}

function stopSelecting() {
  state.selecting = false;
  drag = null;
  $('#mainWrap').classList.remove('selecting');
  $('#selHint').hidden = true;
  $('#selBox').hidden = true;
  $('#selectBtn').classList.remove('on');
}

function barIndexAt(clientX) {
  const x = clientX - $('#mainChart').getBoundingClientRect().left;
  const logical = mainChart.timeScale().coordinateToLogical(x);
  if (logical === null) return null;
  return Math.max(0, Math.min(state.data.main.length - 1, Math.round(logical)));
}

function drawSelBox(i0, i1) {
  const ts = mainChart.timeScale();
  const a = Math.min(i0, i1), b = Math.max(i0, i1);
  const x0 = ts.logicalToCoordinate(a - 0.5), x1 = ts.logicalToCoordinate(b + 0.5);
  const box = $('#selBox');
  box.hidden = false;
  box.style.left = `${x0}px`;
  box.style.width = `${Math.max(2, x1 - x0)}px`;
  const n = b - a + 1;
  $('#selLabel').textContent = `${n} mum`;
  box.style.background = n < MIN_SEL || n > MAX_SEL ? 'rgba(248,113,113,.16)' : '';
}

function bindSelection() {
  const wrap = $('#mainWrap');
  wrap.addEventListener('pointerdown', (e) => {
    if (!state.selecting || e.button !== 0) return;
    const i = barIndexAt(e.clientX);
    if (i === null) return;
    drag = { i0: i, i1: i };
    try { wrap.setPointerCapture(e.pointerId); } catch { /* fare grafikten çıkınca takip edilemez, sorun değil */ }
    drawSelBox(i, i);
    e.preventDefault();
  });
  wrap.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const i = barIndexAt(e.clientX);
    if (i !== null) { drag.i1 = i; drawSelBox(drag.i0, i); }
  });
  wrap.addEventListener('pointerup', () => {
    if (!drag) return;
    const a = Math.min(drag.i0, drag.i1), b = Math.max(drag.i0, drag.i1);
    stopSelecting();
    const n = b - a + 1;
    if (n < MIN_SEL || n > MAX_SEL) {
      toast(`Seçim ${MIN_SEL}–${MAX_SEL} mum arasında olmalı (seçtiğin: ${n}).`, 'error');
      return;
    }
    const main = state.data.main;
    state.selection = { start: main[a][0], end: main[b][0] + barStep() };
    state.group = 0;
    runSearch({ keepView: true });
  });
}

function clearSelection() {
  if (!state.selection) return;
  state.selection = null;
  state.group = 0;
  runSearch();
}

/* ---------- Karşılaştırma penceresi ---------- */

function openModal(i) {
  state.modal = i;
  $('#modal').hidden = false;
  renderModal();
}

function setDrawing(on) {
  state.drawing = on;
  $('#drawBtn').classList.toggle('on', on);
  $('#modal').classList.toggle('drawing', on);
}

function toggleLevels() {
  state.levels = !state.levels;
  $('#levelsBtn').classList.toggle('on', state.levels);
  savePrefs();
  if (state.modal >= 0) renderModal();
}

function closeModal() {
  state.modal = -1;
  setDrawing(false);
  modalToken++;
  $('#modal').hidden = true;
  modalCharts = dispose(modalCharts);
}

function stepModal(delta) {
  const n = shownResults().length;
  const i = state.modal + delta;
  if (i >= 0 && i < n) { state.modal = i; renderModal(); }
}

let modalToken = 0;   // pencere yeniden çizilirse yarım kalan eski çizim iptal olsun diye

async function renderModal() {
  const token = ++modalToken;
  modalCharts = dispose(modalCharts);
  const d = state.data;
  const g = d.groups[state.group];
  const r = g.results[state.modal];
  const sec = TF[d.interval].sec;
  setSeg('#modalAfterSeg', String(state.after));
  $('#prevBtn').disabled = state.modal === 0;
  $('#nextBtn').disabled = state.modal === shownResults().length - 1;
  $('#modalTitle').textContent = `#${r.rank} · ${when(r.start)}`;
  $('#modalSub').textContent = `%${num(r.similarity)} benzer · geçmişteki yapı ${r.length} mum, `
    + `${g.kind === 'secim' ? 'seçtiğin' : 'şu anki'} ${g.length} mum`
    + (state.after ? ` · sonrası ${state.after}× = ${state.after * r.length} mum (${span(state.after * r.length)});`
      + ' daha ilerisi için grafiği sağa kaydır' : '');
  $('#modalQueryLabel').textContent = g.kind === 'secim' ? `Seçtiğin yapı · ${day(g.start)}` : 'Şu an';
  $('#modalPastLabel').textContent = `Geçmiş · ${day(r.start)}`;
  $('#modalQueryHtf').innerHTML = htfChips(g.htf, null, 'query');
  $('#modalPastHtf').innerHTML = htfChips(r.htf, g.htf);
  $('#modalQueryCtx').textContent = ctxText(g.context);
  $('#modalPastCtx').textContent = ctxText(r.ctx) && `${ctxText(r.ctx)} · bağlam uyumu %${r.context}`;
  $('#modalQueryCtx').title = $('#modalPastCtx').title = CTX_HELP;

  // Seçim "şu an"dan önce bitiyorsa, sonrasında ne olduğu da (soluk) gösterilir. Öncesi ana grafikteki
  // kadardır (seviyelere bakmak için sola kaydırılabilir); ilk görünüm yapının yarısı kadar öncesinden.
  const from = g.start - Math.floor(g.length / 2) * sec;
  const to = g.end + state.after * g.length * sec;
  const qBars = d.main.filter((b) => b[0] < to)
    .map((b) => [b[0], b[1], b[2], b[3], b[4], segOf(b[0], g)]);
  const q = makeChart($('#modalQuery'), { axes: true, interactive: true });
  const qSeries = drawCandles(q, qBars);
  drawLevels(q, qBars, g.levels, d.level_tf);
  bindDrawing(q, qSeries, `q:${d.interval}:${g.start}:${g.end}`);
  q.timeScale().setVisibleLogicalRange({
    from: Math.max(0, qBars.findIndex((b) => b[0] >= from)), to: qBars.length + 1,
  });
  modalCharts = [q];

  // Geçmiş eşleşme: ayrıntılı mumlar sunucudan (sonrası 10 katına kadar, birleştirilmeden).
  const key = `${d.interval}:${r.start}:${r.length}:${d.now}`;
  const el = $('#modalMatch');
  let bars = detailCache.get(key);
  if (!bars) {
    el.classList.add('loading');
    el.textContent = 'Yükleniyor…';
    try {
      bars = (await api(`/api/detail?interval=${d.interval}&start=${r.start}&length=${r.length}&now=${d.now}`)).bars;
      detailCache.set(key, bars);
    } catch (e) {
      bars = r.bars;
      toast(`Ayrıntılı veri alınamadı: ${e.message}`, 'error');
    }
    if (token !== modalToken) return;
  }
  el.classList.remove('loading');
  el.textContent = '';
  const rows = state.after ? bars : bars.filter((b) => b[5] !== 2);
  const m = makeChart(el, { axes: true, interactive: true });
  const mSeries = drawCandles(m, rows);
  drawLevels(m, rows, r.levels, d.level_tf);
  bindDrawing(m, mSeries, `m:${d.interval}:${r.start}:${r.length}`);
  // İlk görünüm: yapının yarısı kadar öncesi + yapı + seçilen kat kadar sonrası. Gerisi sağda.
  const vFrom = r.start - Math.floor(r.length / 2) * sec;
  const vTo = r.start + (1 + state.after) * r.length * sec;
  const i0 = Math.max(0, rows.findIndex((b) => b[0] >= vFrom));
  const i1 = rows.findIndex((b) => b[0] >= vTo);
  m.timeScale().setVisibleLogicalRange({ from: i0, to: (i1 < 0 ? rows.length : i1) + 1 });
  modalCharts.push(m);
}

/* ---------- Sunucu ---------- */

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { msg = (await res.json()).detail || msg; } catch { /* yanıt JSON değil */ }
    throw new Error(msg);
  }
  return res.json();
}

function renderStatus(st) {
  state.status = st;
  $('#status').textContent = `${st.count.toLocaleString('tr-TR')} ${TF[st.interval].name} mum · ${short(st.last)} itibarıyla`;
}

function setBusy(on, text = 'Ara') {
  state.busy = on;
  const b = $('#searchBtn');
  b.disabled = on;
  b.classList.toggle('busy', on);
  b.querySelector('.label').textContent = on ? text : 'Ara';
}

let toastTimer;
function toast(msg, kind = '') {
  const t = $('#toast');
  t.textContent = msg;
  t.className = `toast ${kind}`;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 6000);
}

async function runSearch({ keepView = false } = {}) {
  if (state.busy) return;
  const params = new URLSearchParams({ interval: state.tf, method: state.method, rank: state.rank });
  if (state.mode === 'past') {
    const v = $('#nowInput').value;
    if (!v) { toast('Önce bir tarih seç.', 'error'); return; }
    params.set('now', Math.floor(new Date(v).getTime() / 1000));
  }
  if (state.selection) {
    params.set('start', state.selection.start);
    params.set('end', state.selection.end);
  }
  renderSkeleton();
  try {
    if (state.mode === 'live') {
      setBusy(true, 'Veri güncelleniyor…');
      try {
        renderStatus(await api(`/api/update?interval=${state.tf}`, { method: 'POST' }));
      } catch (e) {
        toast(`Veri güncellenemedi, mevcut veriyle aranıyor. (${e.message})`, 'error');
      }
    }
    setBusy(true, 'Aranıyor…');
    state.data = await api(`/api/search?${params}`);
    state.group = Math.min(state.group, Math.max(0, state.data.groups.length - 1));
    render({ keepView });
  } catch (e) {
    toast(`Arama başarısız: ${e.message}`, 'error');
    if (state.selection) state.selection = null;
    if (state.data) renderGrid(); else $('#grid').innerHTML = '';
  } finally {
    setBusy(false);
  }
}

/* ---------- Kontroller ---------- */

function toInputValue(t) {
  const d = new Date(t * 1000);
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
}

function setSeg(sel, v) {
  document.querySelectorAll(`${sel} button`).forEach((b) => b.classList.toggle('on', b.dataset.v === v));
}

function setMode(v) {
  state.mode = v;
  setSeg('#modeSeg', v);
  const input = $('#nowInput');
  input.hidden = v !== 'past';
  if (v === 'past' && !input.value && state.status) input.value = toInputValue(state.status.last);
  savePrefs();
}

function setCount(v) {
  if (state.count === v) return;
  state.count = v;
  setSeg('#countSeg', String(v));
  savePrefs();
  if (state.data) renderGrid();
}

function setAfter(v) {
  if (state.after === v) return;
  state.after = v;
  setSeg('#afterSeg', String(v));
  savePrefs();
  if (state.data) renderGrid();
  if (state.modal >= 0) renderModal();
}

async function setTf(v) {
  if (state.busy) return;
  state.tf = v;
  state.selection = null;
  state.group = 0;
  setSeg('#tfSeg', v);
  $('#tfChip').textContent = `BTC/USDT · ${TF[v].chip}`;
  savePrefs();
  if (mainChart) { mainChart.remove(); mainChart = null; }
  try {
    renderStatus(await api(`/api/status?interval=${v}`));
  } catch (e) {
    toast(`Veri yüklenemedi: ${e.message}`, 'error');
    return;
  }
  runSearch();
}

function bind() {
  document.querySelectorAll('#tfSeg button').forEach((b) => {
    b.onclick = () => { if (state.tf !== b.dataset.v) setTf(b.dataset.v); };
  });
  document.querySelectorAll('#modeSeg button').forEach((b) => {
    b.onclick = () => {
      if (state.mode === b.dataset.v) return;
      state.selection = null;
      setMode(b.dataset.v);
      if (b.dataset.v === 'past') $('#nowInput').focus(); else runSearch();
    };
  });
  document.querySelectorAll('#methodSeg button').forEach((b) => {
    b.onclick = () => {
      if (state.method === b.dataset.v) return;
      state.method = b.dataset.v;
      setSeg('#methodSeg', state.method);
      savePrefs();
      if (state.data) runSearch({ keepView: true });
    };
  });
  document.querySelectorAll('#rankSeg button').forEach((b) => {
    b.onclick = () => {
      if (state.rank === b.dataset.v) return;
      state.rank = b.dataset.v;
      setSeg('#rankSeg', state.rank);
      savePrefs();
      if (state.data) runSearch({ keepView: true });
    };
  });
  document.querySelectorAll('.after-seg button').forEach((b) => {
    b.onclick = () => setAfter(Number(b.dataset.v));
  });
  document.querySelectorAll('#countSeg button').forEach((b) => {
    b.onclick = () => setCount(Number(b.dataset.v));
  });
  $('#nowInput').onchange = () => { state.selection = null; savePrefs(); };
  $('#nowInput').onkeydown = (e) => { if (e.key === 'Enter') runSearch(); };
  $('#searchBtn').onclick = () => runSearch();
  $('#selectBtn').onclick = () => (state.selecting ? stopSelecting() : startSelecting());
  $('#closeBtn').onclick = closeModal;
  $('#levelsBtn').onclick = toggleLevels;
  $('#drawBtn').onclick = () => setDrawing(!state.drawing);
  $('#prevBtn').onclick = () => stepModal(-1);
  $('#nextBtn').onclick = () => stepModal(1);
  $('#modal').onclick = (e) => { if (e.target.id === 'modal') closeModal(); };
  bindSelection();

  document.addEventListener('keydown', (e) => {
    if (!$('#modal').hidden) {
      if (e.key === 'Escape') closeModal();
      else if (e.key === 'ArrowLeft') stepModal(-1);
      else if (e.key === 'ArrowRight') stepModal(1);
      else if (e.key === 'd' || e.key === 'D') setDrawing(!state.drawing);
      return;
    }
    if (e.target.matches('input, textarea')) return;
    if (e.key === 'Escape' && state.selecting) stopSelecting();
    else if (e.key === 's' || e.key === 'S') (state.selecting ? stopSelecting() : startSelecting());
    else if (/^[1-5]$/.test(e.key)) selectGroup(Number(e.key) - 1);
    else if (e.key === 'Enter' && !e.target.closest('.card, button')) runSearch();
  });
}

async function init() {
  bind();
  setSeg('#tfSeg', state.tf);
  $('#tfChip').textContent = `BTC/USDT · ${TF[state.tf].chip}`;
  setSeg('#methodSeg', state.method);
  setSeg('#rankSeg', state.rank);
  $('#levelsBtn').classList.toggle('on', state.levels);
  setSeg('#afterSeg', String(state.after));
  setSeg('#countSeg', String(state.count));
  if (prefs.now) $('#nowInput').value = prefs.now;
  try {
    renderStatus(await api(`/api/status?interval=${state.tf}`));
  } catch (e) {
    toast(`Sunucuya ulaşılamadı: ${e.message}`, 'error');
    return;
  }
  setMode(state.mode);
  runSearch();
}

init();
