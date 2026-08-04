'use strict';
/* ============ Lesson: Train a CNN ============
 * The CNN lesson with the toy numbers replaced by a real convnet learning a
 * real (locally generated) image dataset, with a held-out validation split.
 * One timeline cycle = one SGD step.
 */
const CNNTrainModule = (() => {

  const VW = 1280, VH = 640;
  const cfg = { ds: 'shapes', F1: 6, F2: 12, K: 3, lr: 0.05, batch: 16 };
  let dsSeed = 3, netSeed = 17;

  let ds = null, net = null, iter = null, Wsnap = null;
  let lossHist = [], accHist = [];      // accHist entries: {step, tr, va}
  let stepNo = 0, epoch = 0, order = [], orderPos = 0;
  let cache = {};                        // rendered map bitmaps for sample 0
  let steps = [];
  const EVAL_EVERY = 10;

  // ---------------- setup ----------------
  function loadData() {
    ds = CNet.loadImages(cfg.ds, cfg.ds === 'digits' ? 420 : 320, dsSeed);
    resetNet();
  }

  function resetNet() {
    net = CNet.makeNet({ F1: cfg.F1, F2: cfg.F2, K: cfg.K, nOut: ds.K }, netSeed);
    lossHist = []; accHist = [];
    stepNo = 0; epoch = 0;
    order = shuffled(ds.nTrain); orderPos = 0;
    evaluate();
    prepareIteration();
  }

  function shuffled(n) {
    const rnd = U.mulberry32(netSeed + epoch * 7919 + 1);
    const a = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  function nextIdx() {
    const out = [];
    for (let k = 0; k < cfg.batch; k++) {
      if (orderPos >= order.length) { epoch++; order = shuffled(order.length); orderPos = 0; }
      out.push(order[orderPos++]);
    }
    return out;
  }

  function evaluate() {
    const tr = CNet.evaluate(net, ds.train.X, ds.train.y, 150);
    const va = CNet.evaluate(net, ds.val.X, ds.val.y);
    accHist.push({ step: stepNo, tr: tr.acc, va: va.acc, trL: tr.loss, vaL: va.loss });
    if (accHist.length > 260) accHist.shift();
  }

  function prepareIteration() {
    const idx = nextIdx();
    const r = CNet.trainBatch(net, ds.train.X, ds.train.y, idx, cfg.lr, false);
    // per-sample deltas for the backward animation (sample 0)
    const g0 = CNet.zeroGrads(net);
    const d0 = CNet.backward(net, r.caches[0], ds.train.y[idx[0]], g0);
    Wsnap = { W1: net.W1.slice(), W2: net.W2.slice(), Wd: net.Wd.slice() };
    iter = { idx, ...r, d0, gMax: maxAbs(r.g.W1) };
    lossHist.push(r.meanLoss);
    if (lossHist.length > 400) lossHist.shift();
    buildCache();
    buildSteps();
  }

  function maxAbs(a) { let m = 1e-9; for (const v of a) m = Math.max(m, Math.abs(v)); return m; }

  function onLoop() {
    CNet.sgdStep(net, iter.g, cfg.lr);
    stepNo++;
    if (stepNo % EVAL_EVERY === 0) evaluate();
    prepareIteration();
  }

  function turbo(n) {
    for (let k = 0; k < n; k++) {
      const idx = nextIdx();
      const r = CNet.trainBatch(net, ds.train.X, ds.train.y, idx, cfg.lr, true);
      lossHist.push(r.meanLoss);
      stepNo++;
      if (stepNo % EVAL_EVERY === 0) evaluate();
    }
    while (lossHist.length > 400) lossHist.shift();
    prepareIteration();
  }

  function buildSteps() {
    steps = [
      { type: 'batch',  dur: 2.0 },
      { type: 'conv1',  dur: 2.8 },
      { type: 'pool1',  dur: 1.8 },
      { type: 'conv2',  dur: 2.4 },
      { type: 'pool2',  dur: 1.6 },
      { type: 'dense',  dur: 2.0 },
      { type: 'loss',   dur: 2.0 },
      { type: 'bwd',    dur: 3.0 },
      { type: 'update', dur: 2.2 },
    ];
  }

  // ---------------- bitmap cache ----------------
  function tile(data, off, W, H, ramp) {
    const cv = document.createElement('canvas');
    cv.width = W; cv.height = H;
    const c = cv.getContext('2d');
    const img = c.createImageData(W, H);
    let mx = 1e-6;
    for (let i = 0; i < W * H; i++) mx = Math.max(mx, Math.abs(data[off + i]));
    for (let i = 0; i < W * H; i++) {
      const v = data[off + i];
      const col = ramp(v, mx);
      const n = parseInt(col.slice(1), 16);
      img.data[i * 4] = (n >> 16) & 255;
      img.data[i * 4 + 1] = (n >> 8) & 255;
      img.data[i * 4 + 2] = n & 255;
      img.data[i * 4 + 3] = 255;
    }
    c.putImageData(img, 0, 0);
    return cv;
  }
  const seqRamp = (v, mx) => U.seqColor(U.clamp(v / mx, 0, 1));
  const gray = (v) => U.seqColor(U.clamp(v, 0, 1));

  function buildCache() {
    const c0 = iter.caches[0];
    const S = CNet.SIDE;
    cache = {
      input: tile(c0.x.data, 0, S, S, (v) => gray(v)),
      a1: Array.from({ length: cfg.F1 }, (_, f) => tile(c0.a1.data, f * S * S, S, S, seqRamp)),
      p1: Array.from({ length: cfg.F1 }, (_, f) => tile(c0.p1.data, f * (S / 2) * (S / 2), S / 2, S / 2, seqRamp)),
      a2: Array.from({ length: cfg.F2 }, (_, f) => tile(c0.a2.data, f * (S / 2) * (S / 2), S / 2, S / 2, seqRamp)),
      p2: Array.from({ length: cfg.F2 }, (_, f) => tile(c0.p2.data, f * (S / 4) * (S / 4), S / 4, S / 4, seqRamp)),
      batch: iter.idx.map(i => tile(ds.train.X[i], 0, S, S, (v) => gray(v))),
    };
  }

  // ---------------- layout ----------------
  const P = {
    inp:  { x: 44,  y: 96,  s: 92 },
    f1:   { x: 168, y: 96,  cell: 27, cols: 3 },
    a1:   { x: 288, y: 96,  cell: 42, cols: 3 },
    p1:   { x: 448, y: 96,  cell: 30, cols: 3 },
    a2:   { x: 572, y: 96,  cell: 26, cols: 4 },
    p2:   { x: 700, y: 96,  cell: 18, cols: 4 },
    flat: { x: 788, y: 96,  w: 16, h: 150 },
    bars: { x: 852, y: 100, w: 168 },
    card: { x: 1058, y: 96, w: 182 },
    filt: { x: 44,  y: 344, w: 214, h: 168 },
    bstr: { x: 282, y: 344, w: 300, h: 168 },
    lch:  { x: 612, y: 344, w: 292, h: 128 },
    ach:  { x: 936, y: 344, w: 304, h: 128 },
    tiles:{ x: 612, y: 500 },
  };

  // which pipeline stages are revealed at this step
  const ORDER = ['batch', 'conv1', 'pool1', 'conv2', 'pool2', 'dense', 'loss', 'bwd', 'update'];
  function reveal(si, t, stage) {
    const cur = ORDER.indexOf(steps[si].type);
    const want = ORDER.indexOf(stage);
    if (cur > want) return 1;
    if (cur === want) return U.clamp(t / 0.8, 0, 1);
    return 0;
  }
  // backward wavefront: 1 → 0 across the pipeline, right to left
  function bwdReach(si, t) {
    return steps[si].type === 'bwd' ? U.clamp(t / 0.85, 0, 1) : (steps[si].type === 'update' ? 1 : -1);
  }

  // ---------------- render ----------------
  function render(ctx, si, t) {
    const st = steps[si];
    drawPipeline(ctx, si, t);
    drawFilters(ctx, si, t);
    drawBatchStrip(ctx, si, t);
    drawCharts(ctx);
    drawTiles(ctx);

    txt(ctx, 44, 40, `training a CNN — ${ds.name}`, { size: 15, weight: 650, color: '#fff' });
    txt(ctx, 44, 60, `${ds.nTrain} train / ${ds.nVal} val · 1×16×16 → conv${cfg.F1} → pool → conv${cfg.F2} → pool → dense${ds.K} · ${CNet.paramCount(net).toLocaleString('en-US')} params`,
        { size: 11 });
    txt(ctx, 44, 76, `${ds.blurb} — generated locally`, { size: 10, color: '#63615c' });
  }

  function grid(ctx, canvases, box, n, label, alpha, ringIdx) {
    const cols = box.cols, cell = box.cell, gap = 4;
    for (let i = 0; i < n; i++) {
      const cx = box.x + (i % cols) * (cell + gap);
      const cy = box.y + Math.floor(i / cols) * (cell + gap);
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.imageSmoothingEnabled = false;
      if (canvases[i]) ctx.drawImage(canvases[i], cx, cy, cell, cell);
      ctx.strokeStyle = ringIdx === i ? '#eda100' : 'rgba(255,255,255,0.12)';
      ctx.lineWidth = ringIdx === i ? 1.6 : 1;
      ctx.strokeRect(cx - 0.5, cy - 0.5, cell + 1, cell + 1);
      ctx.restore();
    }
    const rows = Math.ceil(n / cols);
    txt(ctx, box.x, box.y - 10, label, { size: 9.5, weight: 600, color: alpha > 0.5 ? '#c3c2b7' : '#4f4e4b' });
    return box.y + rows * (cell + gap);
  }

  function drawPipeline(ctx, si, t) {
    const st = steps[si];
    const S = CNet.SIDE;
    const c0 = iter.caches[0];
    const label0 = ds.train.y[iter.idx[0]];

    // input
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(cache.input, P.inp.x, P.inp.y, P.inp.s, P.inp.s);
    ctx.strokeStyle = 'rgba(255,255,255,0.18)';
    ctx.strokeRect(P.inp.x - 0.5, P.inp.y - 0.5, P.inp.s + 1, P.inp.s + 1);
    ctx.restore();
    txt(ctx, P.inp.x, P.inp.y - 10, 'input 1×16×16', { size: 9.5, weight: 600, color: '#c3c2b7' });
    txt(ctx, P.inp.x, P.inp.y + P.inp.s + 14, `true: ${ds.classes[label0]}`,
        { size: 11, weight: 600, color: U.CAT[label0 % U.CAT.length] });

    // conv1 filters (live — they change during 'update')
    const aC1 = reveal(si, t, 'conv1');
    drawFilterGrid(ctx, P.f1, net.W1, Wsnap.W1, iter.g.W1, cfg.F1, 1, si, t, aC1);
    txt(ctx, P.f1.x, P.f1.y - 10, `conv1: ${cfg.F1} filters`, { size: 9.5, weight: 600, color: aC1 > .5 ? '#c3c2b7' : '#4f4e4b' });

    grid(ctx, cache.a1, P.a1, cfg.F1, `ReLU maps ${cfg.F1}×16×16`, aC1);
    const aP1 = reveal(si, t, 'pool1');
    grid(ctx, cache.p1, P.p1, cfg.F1, `pool → ${cfg.F1}×8×8`, aP1);
    const aC2 = reveal(si, t, 'conv2');
    grid(ctx, cache.a2, P.a2, cfg.F2, `conv2+ReLU ${cfg.F2}×8×8`, aC2);
    const aP2 = reveal(si, t, 'pool2');
    grid(ctx, cache.p2, P.p2, cfg.F2, `pool → ${cfg.F2}×4×4`, aP2);

    // flatten column
    const aD = reveal(si, t, 'dense');
    const F = P.flat, flat = c0.flat;
    ctx.save();
    ctx.globalAlpha = aD;
    const mx = Math.max(1e-6, ...flat);
    const rowH = F.h / flat.length;
    for (let i = 0; i < flat.length; i++) {
      ctx.fillStyle = U.seqColor(U.clamp(flat[i] / mx, 0, 1));
      ctx.fillRect(F.x, F.y + i * rowH, F.w, Math.max(rowH, 0.8));
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.strokeRect(F.x - 0.5, F.y - 0.5, F.w + 1, F.h + 1);
    ctx.restore();
    txt(ctx, F.x - 4, F.y - 10, `flatten ${flat.length}`, { size: 9.5, weight: 600, color: aD > .5 ? '#c3c2b7' : '#4f4e4b' });

    // class bars
    const B = P.bars;
    txt(ctx, B.x, B.y - 14, 'dense → softmax', { size: 9.5, weight: 600, color: aD > .5 ? '#c3c2b7' : '#4f4e4b' });
    let pred = 0;
    for (let k = 1; k < c0.probs.length; k++) if (c0.probs[k] > c0.probs[pred]) pred = k;
    for (let k = 0; k < ds.K; k++) {
      const y = B.y + k * 24;
      const p = c0.probs[k] * aD;
      const isT = k === label0;
      ctx.save();
      ctx.globalAlpha = aD;
      ctx.fillStyle = isT ? U.CAT[k % U.CAT.length] : '#2e3a46';
      ctx.fillRect(B.x + 44, y, Math.max(1.5, (B.w - 60) * p), 15);
      ctx.restore();
      txt(ctx, B.x + 38, y + 8, ds.classes[k], { align: 'right', mono: true, size: 10,
          color: isT ? '#fff' : '#898781' });
      if (aD > 0.6)
        txt(ctx, B.x + 50 + (B.w - 60) * p, y + 8, (c0.probs[k] * 100).toFixed(0) + '%',
            { mono: true, size: 9.5 });
    }

    // verdict card
    const showLoss = ['loss', 'bwd', 'update'].includes(st.type);
    if (showLoss) {
      const C = P.card;
      const right = pred === label0;
      ctx.save();
      ctx.strokeStyle = right ? '#0ca30c' : '#e34948';
      ctx.fillStyle = right ? 'rgba(12,163,12,0.08)' : 'rgba(227,73,72,0.08)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(C.x, C.y, C.w, 96, 9); else ctx.rect(C.x, C.y, C.w, 96);
      ctx.fill(); ctx.stroke();
      ctx.restore();
      txt(ctx, C.x + 12, C.y + 20, right ? '✓ correct' : '✗ wrong',
          { size: 12, weight: 650, color: right ? '#0ca30c' : '#e34948' });
      txt(ctx, C.x + 12, C.y + 42, `predicted  ${ds.classes[pred]}`, { mono: true, size: 11, color: '#c3c2b7' });
      txt(ctx, C.x + 12, C.y + 58, `true       ${ds.classes[label0]}`, { mono: true, size: 11, color: '#c3c2b7' });
      txt(ctx, C.x + 12, C.y + 78, `loss ${iter.losses[0].toFixed(3)}`, { mono: true, size: 11.5, color: '#fff', weight: 600 });
    }

    // backward wavefront across the pipeline
    const br = bwdReach(si, t);
    if (st.type === 'bwd') {
      const x0 = P.card.x, x1 = P.inp.x;
      const wx = U.lerp(x0, x1, br);
      ctx.save();
      ctx.strokeStyle = U.withAlpha('#eda100', 0.75);
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(wx, P.inp.y - 22); ctx.lineTo(wx, P.inp.y + 176); ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
      txt(ctx, wx - 6, P.inp.y - 30, '∂L/∂· flowing back', { align: 'right', size: 10, color: '#eda100', weight: 600 });
    }

    // connecting arrows
    const arrows = [[P.inp.x + P.inp.s, P.f1.x], [P.f1.x + 92, P.a1.x], [P.a1.x + 134, P.p1.x],
                    [P.p1.x + 98, P.a2.x], [P.a2.x + 113, P.p2.x], [P.p2.x + 81, P.flat.x],
                    [P.flat.x + P.flat.w, P.bars.x + 40]];
    arrows.forEach(([a, b]) => arrow(ctx, a + 5, P.inp.y + 60, b - 5, P.inp.y + 60));
  }

  // conv1 filters, drawn from the weights themselves so they animate as they learn
  function drawFilterGrid(ctx, box, W, Wold, G, F, Cin, si, t, alpha) {
    const st = steps[si];
    const K = cfg.K, cell = box.cell, px = cell / K, gap = 4;
    const upd = st.type === 'update' ? U.easeInOut(U.clamp(t / 0.8, 0, 1)) : (st.type === 'bwd' ? 0 : 1);
    let mx = 1e-6;
    for (const v of W) mx = Math.max(mx, Math.abs(v));
    for (let f = 0; f < F; f++) {
      const cx = box.x + (f % box.cols) * (cell + gap);
      const cy = box.y + Math.floor(f / box.cols) * (cell + gap);
      ctx.save();
      ctx.globalAlpha = alpha;
      for (let ky = 0; ky < K; ky++)
        for (let kx = 0; kx < K; kx++) {
          const i = (f * Cin * K + ky) * K + kx;
          const v = st.type === 'update' ? U.lerp(Wold[i], Wold[i] - cfg.lr * G[i], upd) : W[i];
          ctx.fillStyle = U.divColor(v, mx);
          ctx.fillRect(cx + kx * px, cy + ky * px, px, px);
        }
      // gradient glow while backprop reaches conv1
      const br = bwdReach(si, t);
      if (st.type === 'bwd' && br > 0.75) {
        let gm = 0;
        for (let k = 0; k < K * K; k++) gm = Math.max(gm, Math.abs(G[f * Cin * K * K + k]));
        ctx.strokeStyle = U.withAlpha('#eda100', 0.4 + 0.6 * gm / Math.max(1e-9, iter.gMax));
        ctx.lineWidth = 2;
      } else {
        ctx.strokeStyle = 'rgba(255,255,255,0.14)';
        ctx.lineWidth = 1;
      }
      ctx.strokeRect(cx - 0.5, cy - 0.5, cell + 1, cell + 1);
      ctx.restore();
    }
  }

  // enlarged "what conv1 learned" panel
  function drawFilters(ctx, si, t) {
    const B = P.filt;
    panel(ctx, B.x, B.y, B.w, B.h);
    txt(ctx, B.x, B.y - 10, 'what conv1 has learned (updates live)',
        { size: 10.5, weight: 600, color: '#c3c2b7' });
    const cols = 3, cell = 58, gap = 10;
    const box = { x: B.x + 14, y: B.y + 16, cell, cols };
    drawFilterGrid(ctx, box, net.W1, Wsnap.W1, iter.g.W1, cfg.F1, 1, si, t, 1);
    txt(ctx, B.x + 14, B.y + B.h - 12,
        cfg.F1 <= 6 ? 'blue = negative weight · red = positive' : '', { size: 9.5 });
  }

  function drawBatchStrip(ctx, si, t) {
    const B = P.bstr;
    panel(ctx, B.x, B.y, B.w, B.h);
    txt(ctx, B.x, B.y - 10, `this minibatch — ${cfg.batch} images`,
        { size: 10.5, weight: 600, color: '#c3c2b7' });
    const st = steps[si];
    const showPred = ['loss', 'bwd', 'update'].includes(st.type);
    const cell = 30, gap = 5, cols = 8;
    const rev = st.type === 'batch' ? U.clamp(t / 0.7, 0, 1) * cfg.batch : cfg.batch;
    for (let k = 0; k < Math.min(cfg.batch, 16); k++) {
      if (k >= rev) continue;
      const cx = B.x + 14 + (k % cols) * (cell + gap);
      const cy = B.y + 20 + Math.floor(k / cols) * (cell + gap + 12);
      ctx.save();
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(cache.batch[k], cx, cy, cell, cell);
      const lab = ds.train.y[iter.idx[k]];
      let col = 'rgba(255,255,255,0.14)', lw = 1;
      if (showPred) {
        const p = iter.caches[k].probs;
        let am = 0;
        for (let c2 = 1; c2 < p.length; c2++) if (p[c2] > p[am]) am = c2;
        col = am === lab ? '#0ca30c' : '#e34948'; lw = 1.8;
      }
      if (k === 0) { col = '#eda100'; lw = 2; }
      ctx.strokeStyle = col; ctx.lineWidth = lw;
      ctx.strokeRect(cx - 0.5, cy - 0.5, cell + 1, cell + 1);
      ctx.restore();
      txt(ctx, cx + cell / 2, cy + cell + 8, ds.classes[lab],
          { align: 'center', size: 8.5, mono: true, color: '#63615c' });
    }
    if (showPred)
      txt(ctx, B.x + 14, B.y + B.h - 10,
          `mean batch loss ${iter.meanLoss.toFixed(3)} · green = right, red = wrong`,
          { size: 9.5, color: '#eda100' });
  }

  // ---------------- charts ----------------
  function drawCharts(ctx) {
    // loss
    const L = P.lch;
    panel(ctx, L.x, L.y, L.w, L.h);
    txt(ctx, L.x, L.y - 10, 'batch loss', { size: 10.5, weight: 600, color: '#c3c2b7' });
    if (lossHist.length > 1) {
      const hi = Math.max(...lossHist), lo = 0;
      const px = i => L.x + 8 + i / (lossHist.length - 1) * (L.w - 16);
      const py = v => L.y + L.h - 12 - (v - lo) / Math.max(1e-9, hi - lo) * (L.h - 26);
      ctx.save();
      ctx.strokeStyle = '#3987e5'; ctx.lineWidth = 1.8; ctx.lineJoin = 'round';
      ctx.beginPath();
      lossHist.forEach((v, i) => i ? ctx.lineTo(px(i), py(v)) : ctx.moveTo(px(i), py(v)));
      ctx.stroke();
      ctx.restore();
      txt(ctx, L.x + 7, L.y + 12, hi.toFixed(2), { mono: true, size: 9 });
      txt(ctx, L.x + 7, L.y + L.h - 12, '0', { mono: true, size: 9 });
    }

    // accuracy: train vs val — the overfitting view
    const A = P.ach;
    panel(ctx, A.x, A.y, A.w, A.h);
    txt(ctx, A.x, A.y - 10, 'accuracy — train vs held-out validation',
        { size: 10.5, weight: 600, color: '#c3c2b7' });
    if (accHist.length > 1) {
      const px = i => A.x + 8 + i / (accHist.length - 1) * (A.w - 16);
      const py = v => A.y + A.h - 14 - v * (A.h - 28);
      const line = (key, col, dash) => {
        ctx.save();
        ctx.strokeStyle = col; ctx.lineWidth = 1.8; ctx.lineJoin = 'round';
        if (dash) ctx.setLineDash([4, 3]);
        ctx.beginPath();
        accHist.forEach((h, i) => i ? ctx.lineTo(px(i), py(h[key])) : ctx.moveTo(px(i), py(h[key])));
        ctx.stroke();
        ctx.restore();
      };
      // 100% guide
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.10)';
      ctx.beginPath(); ctx.moveTo(A.x + 8, py(1)); ctx.lineTo(A.x + A.w - 8, py(1)); ctx.stroke();
      ctx.restore();
      line('tr', '#3987e5', false);
      line('va', '#eda100', true);
      const last = accHist[accHist.length - 1];
      txt(ctx, A.x + A.w - 8, py(last.tr) - 9, `train ${(last.tr * 100).toFixed(0)}%`,
          { align: 'right', mono: true, size: 9.5, color: '#3987e5' });
      txt(ctx, A.x + A.w - 8, py(last.va) + 12, `val ${(last.va * 100).toFixed(0)}%`,
          { align: 'right', mono: true, size: 9.5, color: '#eda100' });
      txt(ctx, A.x + 7, py(1) - 7, '100%', { mono: true, size: 9 });
    } else {
      txt(ctx, A.x + A.w / 2, A.y + A.h / 2, 'training…', { align: 'center', size: 11 });
    }
  }

  function drawTiles(ctx) {
    const last = accHist[accHist.length - 1] || { tr: 0, va: 0 };
    const gap2 = last.tr - last.va;
    const tiles = [
      ['step', String(stepNo)],
      ['epoch', String(epoch)],
      ['val accuracy', (last.va * 100).toFixed(1) + '%'],
      ['train − val gap', (gap2 * 100).toFixed(1) + ' pts'],
    ];
    tiles.forEach(([k, v], i) => {
      const tx = P.tiles.x + i * 160, ty = P.tiles.y;
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.09)';
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(tx, ty, 150, 42, 7); else ctx.rect(tx, ty, 150, 42);
      ctx.stroke();
      ctx.restore();
      txt(ctx, tx + 10, ty + 13, k, { size: 9.5 });
      const col = k === 'train − val gap' && gap2 > 0.15 ? '#eda100' : '#fff';
      txt(ctx, tx + 10, ty + 29, v, { mono: true, size: 13, color: col, weight: 600 });
    });
  }

  // ---------------- helpers ----------------
  function panel(ctx, x, y, w, h) {
    ctx.save();
    ctx.fillStyle = '#141414';
    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, 8); else ctx.rect(x, y, w, h);
    ctx.fill(); ctx.stroke();
    ctx.restore();
  }
  function arrow(ctx, x0, y, x1, y1) {
    ctx.save();
    ctx.strokeStyle = ctx.fillStyle = 'rgba(137,135,129,0.5)';
    ctx.lineWidth = 1.1;
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1 - 4, y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x1, y); ctx.lineTo(x1 - 5, y - 3.2); ctx.lineTo(x1 - 5, y + 3.2);
    ctx.closePath(); ctx.fill();
    ctx.restore();
  }
  function txt(ctx, x, y, s, o = {}) {
    ctx.save();
    ctx.fillStyle = o.color || '#898781';
    ctx.font = `${o.weight || 500} ${o.size || 11.5}px ${o.mono ? 'ui-monospace, Menlo, monospace' : 'system-ui, sans-serif'}`;
    ctx.textAlign = o.align || 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(s, x, y);
    ctx.restore();
  }

  // ---------------- captions ----------------
  function caption(si, t) {
    const st = steps[si];
    const c0 = iter.caches[0];
    const label0 = ds.train.y[iter.idx[0]];
    switch (st.type) {
      case 'batch':
        return `Step <b>${stepNo}</b> — a minibatch of <b>${cfg.batch}</b> images. The orange one is the sample we follow through the layers.`;
      case 'conv1':
        return `<b>conv1</b>: ${cfg.F1} filters of ${cfg.K}×${cfg.K} slide over the image. Each produces its own feature map — bright where that filter matched.`;
      case 'pool1':
        return `<b>max-pool 2×2</b>: keep the strongest response in each little square. 16×16 → 8×8, and small shifts stop mattering.`;
      case 'conv2':
        return `<b>conv2</b>: ${cfg.F2} filters that look at all ${cfg.F1} previous maps at once — so these respond to <i>combinations</i> of layer-1 features.`;
      case 'pool2':
        return `Pool again → ${cfg.F2}×4×4. Space keeps shrinking while the number of feature channels grows.`;
      case 'dense': {
        let pred = 0;
        for (let k = 1; k < c0.probs.length; k++) if (c0.probs[k] > c0.probs[pred]) pred = k;
        return `Flatten to ${c0.flat.length} numbers → one dense layer → softmax. It says <b>${ds.classes[pred]}</b> (${(c0.probs[pred] * 100).toFixed(0)}%).`;
      }
      case 'loss':
        return `Cross-entropy compares that distribution with the truth (<b>${ds.classes[label0]}</b>). Averaged over the batch: <b>${iter.meanLoss.toFixed(3)}</b>`;
      case 'bwd':
        return `<b>Backprop</b> — the gradient flows right to left, through the dense layer, back through pooling (only to the pixels that won), and into every filter.`;
      case 'update':
        return `<b>SGD</b>: every filter shifts by −${cfg.lr}·∂L/∂w. Watch the conv1 filters below-left change shape — that is the network learning what to look for.`;
    }
    return '';
  }

  // ---------------- detail ----------------
  function buildDetail(el, si) {
    const st = steps[si];
    const c0 = iter.caches[0];
    const label0 = ds.train.y[iter.idx[0]];
    const last = accHist[accHist.length - 1] || { tr: 0, va: 0 };
    let h = '';
    const shapeRow = (a, b, c) =>
      `<tr><td>${a}</td><td>${b}</td><td>${c}</td></tr>`;
    if (st.type === 'bwd' || st.type === 'update') {
      h = `<div class="dp-title">gradients — same rule as the MLP, one per weight</div><div class="dp-eq">` +
          `∂L/∂logits = p − onehot(${ds.classes[label0]}) = [${c0.probs.map((p, k) => (p - (k === label0 ? 1 : 0)).toFixed(2)).join(', ')}]<br>` +
          `→ dense → un-pool (gradient goes only to the max pixel) → ReLU mask → <b>every filter</b><br>` +
          `largest conv1 filter gradient: <b>${maxAbs(iter.g.W1).toExponential(2)}</b> · ` +
          `update size = lr × that = <b>${(cfg.lr * maxAbs(iter.g.W1)).toExponential(2)}</b>` +
          `</div><div class="dp-note">A convolution shares one small filter across every position, so each filter's gradient is the sum of its contribution at <b>all ${CNet.SIDE * CNet.SIDE}</b> positions. That sharing is why a CNN needs so few parameters.</div>`;
    } else {
      h = `<div class="dp-title">shapes through the network · ${CNet.paramCount(net).toLocaleString('en-US')} parameters</div>` +
          `<table class="dp-table"><thead><tr><th>stage</th><th>output</th><th>params</th></tr></thead><tbody>` +
          shapeRow('input', '1 × 16 × 16', '—') +
          shapeRow(`conv1 (${cfg.K}×${cfg.K}) + ReLU`, `${cfg.F1} × 16 × 16`, (cfg.F1 * (cfg.K * cfg.K + 1)).toLocaleString('en-US')) +
          shapeRow('maxpool 2×2', `${cfg.F1} × 8 × 8`, '—') +
          shapeRow(`conv2 (${cfg.K}×${cfg.K}) + ReLU`, `${cfg.F2} × 8 × 8`, (cfg.F2 * (cfg.F1 * cfg.K * cfg.K + 1)).toLocaleString('en-US')) +
          shapeRow('maxpool 2×2', `${cfg.F2} × 4 × 4`, '—') +
          shapeRow('flatten', `${cfg.F2 * 16}`, '—') +
          shapeRow('dense + softmax', `${ds.K}`, (ds.K * (cfg.F2 * 16 + 1)).toLocaleString('en-US')) +
          `</tbody></table>` +
          `<div class="dp-note">Train ${(last.tr * 100).toFixed(1)}% · validation ${(last.va * 100).toFixed(1)}% — ` +
          `the validation images are <b>never trained on</b>. If train climbs while val stalls, the network is memorising rather than generalising.</div>`;
    }
    el.innerHTML = h;
  }
  function updateDetail() {}

  // ---------------- controls ----------------
  function init(controlsEl, legendEl) {
    controlsEl.innerHTML = '';
    const g0 = grp(controlsEl, 'Dataset');
    const row = document.createElement('div');
    row.className = 'ctl-row';
    row.innerHTML = `<label>images</label><select class="ctl-select">` +
      Object.keys(CNet.DATASETS).map(k => `<option value="${k}">${CNet.DATASETS[k].name}</option>`).join('') +
      `</select>`;
    row.querySelector('select').value = cfg.ds;
    row.querySelector('select').onchange = e => { cfg.ds = e.target.value; loadData(); refresh(); };
    g0.appendChild(row);
    const note = document.createElement('div');
    note.className = 'shape-note';
    note.dataset.id = 'cstat';
    g0.appendChild(note);

    const g1 = grp(controlsEl, 'Architecture');
    list(g1, 'conv1 filters', () => cfg.F1, v => cfg.F1 = v, [4, 6, 8], true);
    list(g1, 'conv2 filters', () => cfg.F2, v => cfg.F2 = v, [8, 12, 16], true);
    list(g1, 'kernel', () => cfg.K, v => cfg.K = v, [3, 5], true);

    const g2 = grp(controlsEl, 'Optimization');
    list(g2, 'learning rate', () => cfg.lr, v => cfg.lr = v, [0.01, 0.02, 0.05, 0.1, 0.2], false);
    list(g2, 'batch size', () => cfg.batch, v => cfg.batch = v, [8, 16, 32], false);

    [['⚡ Train 50 steps', 50], ['⚡⚡ Train 300 steps', 300]].forEach(([label, n]) => {
      const b = document.createElement('button');
      b.className = 'btn';
      b.textContent = label;
      b.onclick = () => { b.textContent = 'training…'; setTimeout(() => { turbo(n); b.textContent = label; App.resetTimeline(); }, 20); };
      controlsEl.appendChild(b);
    });
    const bR = document.createElement('button');
    bR.className = 'btn';
    bR.textContent = '⟲ Reset weights';
    bR.onclick = () => { netSeed = (netSeed * 16807 + 29) % 2147483647; resetNet(); App.resetTimeline(); };
    controlsEl.appendChild(bR);

    legendEl.innerHTML =
      `<div class="legend-row"><span class="legend-swatch" style="background:#3987e5"></span><span>train accuracy</span></div>` +
      `<div class="legend-row"><span class="legend-swatch" style="background:#eda100"></span><span>validation accuracy</span></div>` +
      `<div class="legend-row" style="margin-top:8px"><span>filter weight</span></div>` +
      (() => { const s = []; for (let i = 0; i <= 10; i++) s.push(U.divColor(U.lerp(-1, 1, i / 10), 1));
        return `<div class="legend-ramp" style="background:linear-gradient(90deg,${s.join(',')})"></div>` +
               `<div class="legend-cap"><span>−</span><span>+</span></div>`; })();

    status();
  }

  function status() {
    const el = document.querySelector('[data-id=cstat]');
    if (el) el.innerHTML =
      `<b>${ds.nTrain}</b> train · <b>${ds.nVal}</b> validation<br>` +
      `${ds.K} classes, 16×16 grayscale<br>` +
      `<span style="color:#63615c">${ds.blurb}; generated locally</span>`;
  }
  function refresh() {
    init(document.getElementById('controls'), document.getElementById('legend'));
    App.resetTimeline();
  }

  function grp(parent, title) {
    const g = document.createElement('div');
    g.className = 'ctl-group';
    g.innerHTML = `<h3>${title}</h3>`;
    parent.appendChild(g);
    return g;
  }
  function list(g, label, get, set, vals, doReset) {
    const row = document.createElement('div');
    row.className = 'ctl-row';
    row.innerHTML = `<label>${label}</label>` +
      `<span class="ctl-num"><button data-d="-1">−</button><span class="val">${get()}</span><button data-d="1">+</button></span>`;
    const val = row.querySelector('.val');
    row.querySelectorAll('button').forEach(b => b.onclick = () => {
      const i = vals.indexOf(get());
      const v = vals[U.clamp(i + Number(b.dataset.d), 0, vals.length - 1)];
      if (v === get()) return;
      set(v); val.textContent = get();
      if (doReset) resetNet(); else prepareIteration();
      App.resetTimeline();
    });
    g.appendChild(row);
  }

  loadData();

  return {
    id: 'cnntrain',
    title: 'Train a CNN',
    desc: 'A real convolutional network learning real images, with a held-out validation split. Watch the conv1 filters change shape as it figures out what to look for.',
    VW, VH, loop: true,
    get steps() { return steps; },
    init, regen: () => resetNet(), render, caption, buildDetail, updateDetail, onLoop
  };
})();
