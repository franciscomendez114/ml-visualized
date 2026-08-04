'use strict';
/* ============ Train an MLP — real data, real SGD, animated ============
 * One iteration = one cycle of the timeline:
 *   batch → fwd(sample 0..2) → rest → loss → backward → update → (loop)
 * All numbers come from ML (js/data.js): genuine forward/backward/SGD.
 */
const TrainModule = (() => {

  const VW = 1280, VH = 640;
  const cfg = { ds: 'moons', nHid: 8, lr: 0.1, batch: 8, act: 'relu' };
  let dsSeed = 5, netSeed = 11;

  let ds = null, net = null;
  let iter = null, Wold = null;
  let hist = [], steps = [], stepNo = 0, epoch = 0, seen = 0;
  let order = [], orderPos = 0;
  let bmap = null, bmapDirty = true;     // decision-boundary raster
  const NSHOW = 3, GRID = 40;

  // ---------------- setup ----------------
  function loadData(key) {
    ds = ML.loadBuiltin(key, dsSeed);
    resetNet();
  }
  function setData(newDs) { ds = newDs; resetNet(); }

  function resetNet() {
    net = ML.makeNet([ds.nIn, cfg.nHid, ds.nOut], cfg.act, ds.task, netSeed);
    hist = []; stepNo = 0; epoch = 0; seen = 0;
    order = shuffled(ds.X.length);
    orderPos = 0;
    bmapDirty = true;
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
    seen += out.length;
    return out;
  }

  // compute the whole iteration up-front; the update is applied on loop commit
  function prepareIteration() {
    const idx = nextIdx();
    const r = ML.trainBatch(net, ds.X, ds.y, idx, cfg.lr, false);   // no apply yet
    // per-sample deltas for the backward animation (sample 0)
    const g0 = ML.zeroGrads(net);
    const deltas = ML.backward(net, r.caches[0], ds.y[idx[0]], g0);
    Wold = snapshot();
    iter = { idx, ...r, deltas, gradMax: maxAbsGrad(r.grads) };
    hist.push(r.meanLoss);
    if (hist.length > 400) hist.shift();
    buildSteps();
  }

  function snapshot() {
    return net.L.map(l => ({ W: l.W.map(r => r.slice()), b: l.b.slice() }));
  }
  function maxAbsGrad(g) {
    let m = 1e-9;
    for (const gl of g) {
      for (const row of gl.W) for (const v of row) m = Math.max(m, Math.abs(v));
      for (const v of gl.b) m = Math.max(m, Math.abs(v));
    }
    return m;
  }
  function maxAbsW() {
    let m = 1e-9;
    for (const l of net.L) for (const row of l.W) for (const v of row) m = Math.max(m, Math.abs(v));
    return m;
  }

  function buildSteps() {
    const nShow = Math.min(NSHOW, cfg.batch);
    steps = [{ type: 'batch', dur: 2.0 }];
    for (let s = 0; s < nShow; s++) steps.push({ type: 'fwd', s, dur: 2.2 });
    if (cfg.batch > nShow) steps.push({ type: 'rest', dur: 1.7 });
    steps.push({ type: 'loss', dur: 2.0 },
               { type: 'bwd', dur: 3.0 },
               { type: 'update', dur: 2.2 });
  }

  // committed by main.js when the timeline wraps
  function onLoop() {
    ML.sgdStep(net, iter.grads, cfg.lr);
    stepNo++;
    bmapDirty = true;
    prepareIteration();
  }

  // run many iterations with no animation
  function turbo(n) {
    for (let k = 0; k < n; k++) {
      const idx = nextIdx();
      const r = ML.trainBatch(net, ds.X, ds.y, idx, cfg.lr, true);
      hist.push(r.meanLoss);
      stepNo++;
    }
    while (hist.length > 400) hist.shift();
    bmapDirty = true;
    prepareIteration();
  }

  // ---------------- layout ----------------
  const P = {
    data: { x: 44, y: 108, w: 232, h: 232 },
    net:  { x: 330, y: 96, w: 580, h: 300 },
    loss: { x: 968, y: 130, w: 268, h: 150 },
    tbl:  { x: 330, y: 432 }
  };

  function nodePos(l, j) {
    const sizes = [ds.nIn, cfg.nHid, ds.nOut];
    const xs = [P.net.x + 40, P.net.x + P.net.w / 2, P.net.x + P.net.w - 40];
    const n = sizes[l];
    const gap = Math.min(38, (P.net.h - 30) / Math.max(1, n - 1) || P.net.h);
    return [xs[l], P.net.y + P.net.h / 2 - gap * (n - 1) / 2 + j * gap];
  }
  const R = () => Math.min(15, Math.max(9, 120 / Math.max(cfg.nHid, ds.nIn)));

  // ---------------- render ----------------
  function render(ctx, si, t) {
    const st = steps[si];
    drawDataPanel(ctx, si, t);
    drawNet(ctx, si, t);
    drawLossPanel(ctx, si, t);
    drawBatchTable(ctx, si, t);
    drawHeader(ctx, si, t);
  }

  function drawHeader(ctx, si, t) {
    txt(ctx, 44, 46, `training an MLP — ${ds.name}`, { size: 15, weight: 650, color: '#fff' });
    txt(ctx, 44, 66, `${ds.X.length} samples · ${ds.nIn}→${cfg.nHid}→${ds.nOut} · ${ML.ACTS[cfg.act].name} · SGD lr=${cfg.lr} · batch ${cfg.batch}`, { size: 11.5 });
    txt(ctx, 44, 84, ds.synthetic ? 'synthetic data, generated locally' : `loaded: ${ds.name}`,
        { size: 10.5, color: '#63615c' });
  }

  // --- data panel: decision boundary (2-feature clf) or pred-vs-actual (reg) ---
  function drawDataPanel(ctx, si, t) {
    const b = P.data;
    ctx.save();
    ctx.fillStyle = '#161616';
    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(b.x, b.y, b.w, b.h, 8); else ctx.rect(b.x, b.y, b.w, b.h);
    ctx.fill(); ctx.stroke();
    ctx.restore();

    if (ds.task === 'clf' && ds.nIn === 2) drawBoundary(ctx, b, si, t);
    else drawPredScatter(ctx, b, si, t);
  }

  const dbound = () => {
    let lo = 0, hi = 0;
    for (const r of ds.X) for (const v of r) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const pad = (hi - lo) * 0.12;
    return [lo - pad, hi + pad];
  };

  function drawBoundary(ctx, b, si, t) {
    const [lo, hi] = dbound();
    if (bmapDirty || !bmap) {
      if (!bmap) {
        bmap = document.createElement('canvas');
        bmap.width = GRID; bmap.height = GRID;
      }
      const g2 = bmap.getContext('2d');
      const img = g2.createImageData(GRID, GRID);
      for (let iy = 0; iy < GRID; iy++)
        for (let ix = 0; ix < GRID; ix++) {
          const x1 = lo + (hi - lo) * (ix + 0.5) / GRID;
          const x2 = hi - (hi - lo) * (iy + 0.5) / GRID;
          const p = ML.forward(net, [x1, x2]).out;
          let am = 0;
          for (let k = 1; k < p.length; k++) if (p[k] > p[am]) am = k;
          const conf = U.clamp((p[am] - 1 / p.length) / (1 - 1 / p.length), 0, 1);
          const col = U.mix('#161616', U.CAT[am % U.CAT.length], 0.18 + 0.42 * conf);
          const n = parseInt(col.slice(1), 16);
          const o = (iy * GRID + ix) * 4;
          img.data[o] = (n >> 16) & 255; img.data[o + 1] = (n >> 8) & 255;
          img.data[o + 2] = n & 255; img.data[o + 3] = 255;
        }
      g2.putImageData(img, 0, 0);
      bmapDirty = false;
    }
    ctx.save();
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(b.x, b.y, b.w, b.h, 8); else ctx.rect(b.x, b.y, b.w, b.h);
    ctx.clip();
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(bmap, b.x, b.y, b.w, b.h);
    // data points
    const sx = v => b.x + (v - lo) / (hi - lo) * b.w;
    const sy = v => b.y + b.h - (v - lo) / (hi - lo) * b.h;
    const inBatch = new Set(iter.idx);
    for (let i = 0; i < ds.X.length; i++) {
      const hot = inBatch.has(i);
      ctx.beginPath();
      ctx.arc(sx(ds.X[i][0]), sy(ds.X[i][1]), hot ? 3.6 : 2.1, 0, 7);
      ctx.fillStyle = U.CAT[ds.y[i] % U.CAT.length];
      ctx.globalAlpha = hot ? 1 : 0.62;
      ctx.fill();
      if (hot) {
        ctx.globalAlpha = 1;
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.4; ctx.stroke();
      }
    }
    ctx.restore();
    txt(ctx, b.x, b.y - 10, 'decision boundary (live)', { size: 10.5, weight: 600, color: '#c3c2b7' });
    txt(ctx, b.x + b.w, b.y - 10, `${ds.feats[0]} · ${ds.feats[1]}`, { size: 10, align: 'right' });
    txt(ctx, b.x, b.y + b.h + 13, 'white ring = this batch', { size: 10 });
  }

  function drawPredScatter(ctx, b, si, t) {
    const pts = [];
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < ds.X.length; i += Math.max(1, Math.floor(ds.X.length / 120))) {
      const p = ML.forward(net, ds.X[i]).out[0];
      const yv = ds.y[i];
      pts.push([yv, p, i]);
      lo = Math.min(lo, yv, p); hi = Math.max(hi, yv, p);
    }
    const pad = (hi - lo) * 0.1 || 1;
    lo -= pad; hi += pad;
    const sx = v => b.x + 8 + (v - lo) / (hi - lo) * (b.w - 16);
    const sy = v => b.y + b.h - 8 - (v - lo) / (hi - lo) * (b.h - 16);
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.18)';
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(sx(lo), sy(lo)); ctx.lineTo(sx(hi), sy(hi)); ctx.stroke();
    ctx.setLineDash([]);
    const inBatch = new Set(iter.idx);
    for (const [yv, p, i] of pts) {
      ctx.beginPath();
      ctx.arc(sx(yv), sy(p), inBatch.has(i) ? 3.4 : 2, 0, 7);
      ctx.fillStyle = inBatch.has(i) ? '#eda100' : '#3987e5';
      ctx.globalAlpha = inBatch.has(i) ? 1 : 0.55;
      ctx.fill();
    }
    ctx.restore();
    txt(ctx, b.x, b.y - 10, 'predicted vs actual', { size: 10.5, weight: 600, color: '#c3c2b7' });
    txt(ctx, b.x + b.w / 2, b.y + b.h + 13, `actual ${ds.targetName} →  (dashed = perfect)`, { size: 10, align: 'center' });
  }

  // --- the network ---
  function fwdFront(si, t) {         // 0..1 across layers during a fwd step
    const st = steps[si];
    return st.type === 'fwd' ? U.clamp(t / 0.82, 0, 1) : null;
  }
  function bwdFront(si, t) {
    const st = steps[si];
    return st.type === 'bwd' ? U.clamp(t / 0.85, 0, 1) : null;
  }

  // weights to display (interpolated during the update step)
  function dispW(si, t, l, i, j) {
    const st = steps[si];
    if (st.type === 'update') {
      const k = U.easeInOut(U.clamp(t / 0.8, 0, 1));
      return Wold[l].W[i][j] - k * cfg.lr * iter.grads[l].W[i][j];
    }
    return net.L[l].W[i][j];
  }

  function drawNet(ctx, si, t) {
    const st = steps[si];
    const sizes = [ds.nIn, cfg.nHid, ds.nOut];
    const wMax = Math.max(maxAbsW(), 0.2);
    const ff = fwdFront(si, t), bf = bwdFront(si, t);
    const sIdx = st.type === 'fwd' ? st.s : 0;
    const cache = iter.caches[sIdx];

    // edges
    for (let l = 0; l < 2; l++) {
      for (let i = 0; i < sizes[l + 1]; i++) {
        for (let j = 0; j < sizes[l]; j++) {
          const w = dispW(si, t, l, i, j);
          const [x0, y0] = nodePos(l, j), [x1, y1] = nodePos(l + 1, i);
          let col = U.divColor(w, wMax), alpha = 0.18 + 0.5 * Math.abs(w) / wMax, lw = 0.6 + 1.7 * Math.abs(w) / wMax;

          if (ff !== null && ff * 2 > l) { alpha = Math.min(1, alpha + 0.35); }
          if (bf !== null) {
            const reach = (1 - bf) * 2;    // wavefront travels right → left
            if (reach <= l + 1) {
              const g = iter.grads[l].W[i][j];
              col = U.divColor(-g, iter.gradMax);   // color = direction of the update
              alpha = 0.35 + 0.65 * Math.abs(g) / iter.gradMax;
              lw = 0.8 + 2.6 * Math.abs(g) / iter.gradMax;
            }
          }
          if (st.type === 'update') {
            const g = iter.grads[l].W[i][j];
            alpha = Math.min(1, alpha + 0.3 * Math.abs(g) / iter.gradMax);
          }
          ctx.save();
          ctx.globalAlpha = alpha;
          ctx.strokeStyle = col; ctx.lineWidth = lw;
          ctx.beginPath();
          ctx.moveTo(x0 + R(), y0); ctx.lineTo(x1 - R(), y1);
          ctx.stroke();
          ctx.restore();
        }
      }
    }

    // traveling pulses
    if (ff !== null) {
      const seg = ff * 2;
      const l = U.clamp(Math.floor(seg), 0, 1);
      const u = seg - l;
      for (let i = 0; i < sizes[l + 1]; i++)
        for (let j = 0; j < sizes[l]; j++) {
          const [x0, y0] = nodePos(l, j), [x1, y1] = nodePos(l + 1, i);
          const act = cache.a[l][j];
          if (Math.abs(act) < 0.05) continue;
          ctx.save();
          ctx.globalAlpha = 0.8;
          ctx.fillStyle = '#ffffff';
          ctx.beginPath();
          ctx.arc(U.lerp(x0 + R(), x1 - R(), u), U.lerp(y0, y1, u), 2.2, 0, 7);
          ctx.fill();
          ctx.restore();
        }
    }
    if (bf !== null) {
      const seg = (1 - bf) * 2;
      const l = U.clamp(Math.floor(seg), 0, 1);
      const u = 1 - (seg - l);
      for (let i = 0; i < sizes[l + 1]; i++)
        for (let j = 0; j < sizes[l]; j++) {
          const [x0, y0] = nodePos(l, j), [x1, y1] = nodePos(l + 1, i);
          ctx.save();
          ctx.globalAlpha = 0.85;
          ctx.fillStyle = '#eda100';
          ctx.beginPath();
          ctx.arc(U.lerp(x1 - R(), x0 + R(), u), U.lerp(y1, y0, u), 2.4, 0, 7);
          ctx.fill();
          ctx.restore();
        }
    }

    // nodes
    const r = R();
    for (let l = 0; l < 3; l++) {
      for (let j = 0; j < sizes[l]; j++) {
        const [x, y] = nodePos(l, j);
        const reached = ff === null ? (st.type !== 'batch') : ff * 2 >= l - 0.02;
        const v = cache.a[l] ? cache.a[l][j] : 0;
        const absA = Math.max(1, ...cache.a[l].map(Math.abs));
        let fill = '#20242b';
        if (reached && st.type !== 'batch') fill = U.divColor(v, absA);
        ctx.save();
        ctx.beginPath(); ctx.arc(x, y, r, 0, 7);
        ctx.fillStyle = fill; ctx.fill();
        // backward: ring shows |δ| for this node's layer
        let ring = 'rgba(255,255,255,0.16)', rw = 1.1;
        if (bf !== null && l > 0) {
          const reach = (1 - bf) * 2;
          if (reach <= l) {
            const dv = iter.deltas[l - 1][j];
            const dm = Math.max(...iter.deltas[l - 1].map(Math.abs), 1e-9);
            ring = U.withAlpha('#eda100', 0.35 + 0.65 * Math.abs(dv) / dm);
            rw = 1.4 + 1.8 * Math.abs(dv) / dm;
          }
        }
        ctx.strokeStyle = ring; ctx.lineWidth = rw; ctx.stroke();
        if (reached && st.type !== 'batch' && r >= 11) {
          ctx.fillStyle = U.inkFor(fill);
          ctx.font = `600 ${r <= 12 ? 8.5 : 9.5}px ui-monospace, Menlo, monospace`;
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(fmt(v), x, y + 0.5);
        }
        ctx.restore();
      }
    }

    // layer captions
    const labs = [`input · ${ds.nIn}`, `hidden · ${cfg.nHid} ${ML.ACTS[cfg.act].name}`,
                  ds.task === 'clf' ? `output · ${ds.nOut} softmax` : 'output · 1 linear'];
    for (let l = 0; l < 3; l++)
      txt(ctx, nodePos(l, 0)[0], P.net.y + P.net.h + 6, labs[l],
          { align: 'center', mono: true, size: 10 });

    // input feature names
    for (let j = 0; j < ds.nIn && ds.nIn <= 6; j++) {
      const [x, y] = nodePos(0, j);
      txt(ctx, x - r - 7, y, ds.feats[j], { align: 'right', size: 9.5 });
    }
    // output labels
    for (let j = 0; j < ds.nOut; j++) {
      const [x, y] = nodePos(2, j);
      txt(ctx, x + r + 7, y, ds.task === 'clf' ? (ds.classes ? ds.classes[j] : 'c' + j) : ds.targetName,
          { size: 9.5 });
    }

    // loss node to the right of the output
    if (['loss', 'bwd', 'update'].includes(st.type) || st.type === 'rest') drawLossNode(ctx, si, t);
  }

  function drawLossNode(ctx, si, t) {
    const st = steps[si];
    const [ox, oy] = nodePos(2, 0);
    const x = P.net.x + P.net.w + 22, y = P.net.y + P.net.h / 2;
    const active = st.type === 'loss' || st.type === 'bwd';
    ctx.save();
    ctx.strokeStyle = active ? '#eda100' : 'rgba(255,255,255,0.2)';
    ctx.fillStyle = '#20242b';
    ctx.lineWidth = active ? 1.8 : 1.1;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x - 30, y - 22, 60, 44, 9); else ctx.rect(x - 30, y - 22, 60, 44);
    ctx.fill(); ctx.stroke();
    ctx.restore();
    txt(ctx, x, y - 8, 'loss', { align: 'center', size: 10, weight: 600, color: active ? '#eda100' : '#898781' });
    txt(ctx, x, y + 8, iter.meanLoss.toFixed(3), { align: 'center', mono: true, size: 11.5, color: '#fff' });
    if (st.type === 'bwd') {
      const a = 0.4 + 0.6 * Math.abs(Math.sin(t * 9));
      txt(ctx, x, y + 34, '∂L/∂·', { align: 'center', mono: true, size: 10, color: U.withAlpha('#eda100', a) });
    }
  }

  // --- loss curve ---
  function drawLossPanel(ctx, si, t) {
    const b = P.loss;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(b.x, b.y, b.w, b.h, 8); else ctx.rect(b.x, b.y, b.w, b.h);
    ctx.stroke();
    ctx.restore();
    txt(ctx, b.x, b.y - 10, 'batch loss', { size: 10.5, weight: 600, color: '#c3c2b7' });

    if (hist.length > 1) {
      const hi = Math.max(...hist), lo = Math.min(...hist, 0);
      const sx = i => b.x + 6 + (i / (hist.length - 1)) * (b.w - 12);
      const sy = v => b.y + b.h - 8 - ((v - lo) / Math.max(1e-9, hi - lo)) * (b.h - 20);
      ctx.save();
      ctx.strokeStyle = '#3987e5'; ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      hist.forEach((v, i) => i ? ctx.lineTo(sx(i), sy(v)) : ctx.moveTo(sx(i), sy(v)));
      ctx.stroke();
      ctx.fillStyle = '#3987e5';
      ctx.beginPath(); ctx.arc(sx(hist.length - 1), sy(hist[hist.length - 1]), 3, 0, 7); ctx.fill();
      ctx.restore();
      txt(ctx, b.x + 6, b.y + 12, hi.toFixed(2), { mono: true, size: 9.5 });
      txt(ctx, b.x + 6, b.y + b.h - 8, lo.toFixed(2), { mono: true, size: 9.5 });
    } else {
      txt(ctx, b.x + b.w / 2, b.y + b.h / 2, 'training…', { align: 'center', size: 11 });
    }

    // metric tiles
    const my = b.y + b.h + 26;
    const tiles = ds.task === 'clf'
      ? [['step', String(stepNo)], ['epoch', String(epoch)],
         ['loss', iter.meanLoss.toFixed(3)], ['train acc', (ML.accuracy(net, ds.X, ds.y) * 100).toFixed(1) + '%']]
      : [['step', String(stepNo)], ['epoch', String(epoch)],
         ['loss', iter.meanLoss.toFixed(3)], ['rmse', (Math.sqrt(2 * ML.meanLoss(net, ds.X, ds.y)) * (ds.sdY || 1)).toFixed(2)]];
    tiles.forEach(([k, v], i) => {
      const tx = b.x + (i % 2) * 136, ty = my + Math.floor(i / 2) * 46;
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.09)';
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(tx, ty, 126, 38, 7); else ctx.rect(tx, ty, 126, 38);
      ctx.stroke();
      ctx.restore();
      txt(ctx, tx + 10, ty + 12, k, { size: 9.5 });
      txt(ctx, tx + 10, ty + 26, v, { mono: true, size: 13, color: '#fff', weight: 600 });
    });
  }

  // --- batch table ---
  function drawBatchTable(ctx, si, t) {
    const st = steps[si];
    const b = P.tbl;
    const nShow = Math.min(NSHOW, cfg.batch);
    const colW = 62, rowH = 20;
    const shownRows = Math.min(cfg.batch, 6);

    let revealed = 0;
    if (st.type === 'batch') revealed = U.clamp(t / 0.7, 0, 1) * cfg.batch;
    else revealed = cfg.batch;

    // header
    const cols = [...ds.feats.slice(0, 4).map(f => f), 'target', 'pred', 'loss'];
    cols.forEach((c, i) =>
      txt(ctx, b.x + 30 + i * colW, b.y - 4, c, { size: 9.5, weight: 600, color: '#63615c', align: 'center' }));

    for (let k = 0; k < shownRows; k++) {
      const y = b.y + 8 + k * rowH;
      if (k >= revealed) continue;
      const i = iter.idx[k];
      const active = st.type === 'fwd' && st.s === k;
      if (active) {
        ctx.save();
        ctx.fillStyle = 'rgba(237,161,0,0.10)';
        ctx.strokeStyle = U.withAlpha('#eda100', 0.7);
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(b.x - 4, y - 2, 30 + cols.length * colW, rowH - 2, 5);
        else ctx.rect(b.x - 4, y - 2, 30 + cols.length * colW, rowH - 2);
        ctx.fill(); ctx.stroke();
        ctx.restore();
      }
      txt(ctx, b.x + 6, y + 8, '#' + k, { size: 9.5, mono: true, color: active ? '#eda100' : '#63615c' });
      // features (raw units where we have them)
      const raw = ds.Xraw[i];
      for (let f = 0; f < Math.min(4, ds.nIn); f++)
        txt(ctx, b.x + 30 + f * colW, y + 8, fmt(raw[f]), { mono: true, size: 10, align: 'center', color: '#c3c2b7' });
      const cT = 30 + Math.min(4, ds.nIn) * colW;
      // target
      const tgt = ds.task === 'clf'
        ? (ds.classes ? ds.classes[ds.y[i]] : String(ds.y[i]))
        : fmt(ds.yraw[i]);
      txt(ctx, b.x + cT, y + 8, tgt, { mono: true, size: 10, align: 'center',
        color: ds.task === 'clf' ? U.CAT[ds.y[i] % U.CAT.length] : '#c3c2b7' });
      // prediction appears only once this sample has actually been forwarded
      let done;
      if (st.type === 'fwd') done = k < st.s || (k === st.s && t > 0.8);
      else if (st.type === 'rest') done = k < nShow || t > 0.5;
      else done = ['loss', 'bwd', 'update'].includes(st.type);
      if (done) {
        const p = iter.caches[k].out;
        let ptxt, pcol;
        if (ds.task === 'clf') {
          let am = 0;
          for (let c2 = 1; c2 < p.length; c2++) if (p[c2] > p[am]) am = c2;
          ptxt = `${ds.classes ? ds.classes[am] : am} ${(p[am] * 100).toFixed(0)}%`;
          pcol = am === ds.y[i] ? '#0ca30c' : '#e34948';
        } else {
          ptxt = fmt(p[0] * (ds.sdY || 1) + (ds.muY || 0));
          pcol = '#c3c2b7';
        }
        txt(ctx, b.x + cT + colW, y + 8, ptxt, { mono: true, size: 10, align: 'center', color: pcol });
        if (['loss', 'bwd', 'update'].includes(st.type))
          txt(ctx, b.x + cT + 2 * colW, y + 8, iter.losses[k].toFixed(3),
              { mono: true, size: 10, align: 'center', color: '#898781' });
      } else {
        txt(ctx, b.x + cT + colW, y + 8, '…', { mono: true, size: 11, align: 'center', color: '#63615c' });
      }
    }
    if (cfg.batch > shownRows) {
      const y = b.y + 8 + shownRows * rowH;
      txt(ctx, b.x + 30, y + 8, `… + ${cfg.batch - shownRows} more samples in this batch (computed, not drawn)`,
          { size: 10, color: '#63615c' });
    }
    // mean loss chip
    if (['loss', 'bwd', 'update'].includes(st.type)) {
      const y = b.y + 8 + (shownRows + (cfg.batch > shownRows ? 1 : 0)) * rowH + 6;
      const a = st.type === 'loss' ? U.easeOut(U.clamp(t / 0.5, 0, 1)) : 1;
      ctx.save();
      ctx.globalAlpha = a;
      txt(ctx, b.x + 30, y + 8, `mean batch loss = ${iter.meanLoss.toFixed(4)}`,
          { mono: true, size: 11.5, color: '#eda100', weight: 600 });
      ctx.restore();
    }
  }

  const fmt = v => Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2);

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
    switch (st.type) {
      case 'batch':
        return `Step <b>${stepNo}</b> — draw a minibatch of <b>${cfg.batch}</b> samples (highlighted in the data panel). Features are standardized before they enter the net.`;
      case 'fwd': {
        const i = iter.idx[st.s];
        const tgt = ds.task === 'clf' ? (ds.classes ? ds.classes[ds.y[i]] : ds.y[i]) : fmt(ds.yraw[i]);
        return `<b>Forward pass, sample #${st.s}</b> — real values flow left to right: each neuron computes z = Σ w·x + b, then ${ML.ACTS[cfg.act].name}. Target: <b>${tgt}</b>`;
      }
      case 'rest':
        return `The other <b>${cfg.batch - Math.min(NSHOW, cfg.batch)}</b> samples run the same way — computed in full, just not animated. Gradients are averaged over all ${cfg.batch}.`;
      case 'loss':
        return `<b>Loss</b> = ${ds.task === 'clf' ? 'cross-entropy' : 'mean squared error'} between prediction and target, averaged over the batch → <b>${iter.meanLoss.toFixed(4)}</b>`;
      case 'bwd':
        return `<b>Backpropagation</b> — gradients flow <b>right to left</b>: ∂L/∂z at the output, then chain-ruled back through W₂, the hidden layer, and W₁. Edge color = which way that weight will move.`;
      case 'update':
        return `<b>SGD update</b>: w ← w − ${cfg.lr}·∂L/∂w. Every edge shifts a little; the decision boundary re-draws. Then the next batch — press ▶ to keep training.`;
    }
    return '';
  }

  // ---------------- detail panel ----------------
  function chips(v, absM, fmtF = x => x.toFixed(2)) {
    const arr = Array.isArray(v) ? v : [v];
    const m = absM || Math.max(1e-6, ...arr.map(Math.abs));
    return `<span style="display:inline-flex;gap:2px;vertical-align:middle">` +
      arr.map(x => {
        const bg = U.divColor(x, m);
        return `<span style="min-width:38px;height:19px;padding:0 3px;border-radius:4px;background:${bg};color:${U.inkFor(bg)};font:600 9.5px var(--mono);display:flex;align-items:center;justify-content:center">${fmtF(x)}</span>`;
      }).join('') + `</span>`;
  }

  function buildDetail(el, si) {
    const st = steps[si];
    let h = '';
    const s = st.type === 'fwd' ? st.s : 0;
    const i = iter.idx[s];
    const c = iter.caches[s];

    if (st.type === 'batch') {
      h = `<div class="dp-title">minibatch — ${cfg.batch} of ${ds.X.length} samples · epoch ${epoch}</div>` +
          `<div class="dp-eq">indices: <b>${iter.idx.join(', ')}</b><br>` +
          `<span style="color:var(--ink-muted)">features standardized: x̂ = (x − μ)/σ, computed over the whole dataset</span></div>`;
    } else if (st.type === 'fwd') {
      h = `<div class="dp-title">forward pass — sample #${s} (dataset row ${i})</div><div class="dp-eq">` +
          `x&nbsp; = ${chips(c.a[0], null)}<br>` +
          `z₁ = W₁x + b₁ = ${chips(c.z[0], null)}<br>` +
          `a₁ = ${ML.ACTS[cfg.act].name}(z₁) = ${chips(c.a[1], null)}<br>` +
          `z₂ = W₂a₁ + b₂ = ${chips(c.z[1], null)}<br>` +
          (ds.task === 'clf'
            ? `p&nbsp; = softmax(z₂) = ${chips(c.out, 1, x => x.toFixed(2))} → target <b>${ds.classes ? ds.classes[ds.y[i]] : ds.y[i]}</b>, loss <b>${iter.losses[s].toFixed(3)}</b>`
            : `ŷ&nbsp; = ${fmt(c.out[0] * (ds.sdY || 1) + (ds.muY || 0))} vs target <b>${fmt(ds.yraw[i])}</b>, loss <b>${iter.losses[s].toFixed(3)}</b>`) +
          `</div>`;
    } else if (st.type === 'rest') {
      h = `<div class="dp-title">the rest of the batch</div><div class="dp-eq">` +
          `losses: ${iter.losses.map(l => l.toFixed(2)).join(' · ')}<br>` +
          `<span style="color:var(--ink-muted)">each sample gets its own forward pass and its own gradient; SGD averages them.</span></div>`;
    } else if (st.type === 'loss') {
      h = `<div class="dp-title">batch loss</div><div class="dp-eq">` +
          (ds.task === 'clf'
            ? `L = −(1/B)·Σ log p[correct class]<br>`
            : `L = (1/B)·Σ ½(ŷ − y)²<br>`) +
          `&nbsp;&nbsp;= (${iter.losses.map(l => l.toFixed(2)).join(' + ')}) / ${cfg.batch}<br>` +
          `&nbsp;&nbsp;= <b>${iter.meanLoss.toFixed(4)}</b></div>`;
    } else if (st.type === 'bwd') {
      const d2 = iter.deltas[1], d1 = iter.deltas[0];
      h = `<div class="dp-title">backprop — chain rule, sample #0</div><div class="dp-eq">` +
          (ds.task === 'clf'
            ? `∂L/∂z₂ = p − onehot(y) = ${chips(d2, null)}<br>`
            : `∂L/∂z₂ = ŷ − y = ${chips(d2, null)}<br>`) +
          `∂L/∂W₂ = ∂L/∂z₂ ⊗ a₁ &nbsp;<span style="color:var(--ink-muted)">(max |g| = ${maxAbsGrad([iter.grads[1]]).toFixed(3)})</span><br>` +
          `∂L/∂a₁ = W₂ᵀ·∂L/∂z₂ &nbsp;→&nbsp; ∂L/∂z₁ = ∂L/∂a₁ ⊙ ${ML.ACTS[cfg.act].name}′(z₁) = ${chips(d1, null)}<br>` +
          `∂L/∂W₁ = ∂L/∂z₁ ⊗ x &nbsp;<span style="color:var(--ink-muted)">(max |g| = ${maxAbsGrad([iter.grads[0]]).toFixed(3)})</span></div>`;
    } else if (st.type === 'update') {
      const rows = [];
      for (let l = 0; l < 2 && rows.length < 4; l++)
        for (let a2 = 0; a2 < net.L[l].nout && rows.length < 4; a2++)
          for (let b2 = 0; b2 < net.L[l].nin && rows.length < 4; b2++) {
            const w = Wold[l].W[a2][b2], g = iter.grads[l].W[a2][b2];
            rows.push(`W${l + 1}[${a2}][${b2}]:&nbsp; ${w.toFixed(4)} − ${cfg.lr}·(${g.toFixed(4)}) = <b>${(w - cfg.lr * g).toFixed(4)}</b>`);
          }
      h = `<div class="dp-title">gradient descent — w ← w − lr·∂L/∂w &nbsp;(lr = ${cfg.lr})</div>` +
          `<div class="dp-eq" style="line-height:1.9;font-size:12px">${rows.join('<br>')}<br>` +
          `<span style="color:var(--ink-muted)">…and every other weight, the same way. Loss so far: ${hist[0].toFixed(3)} → ${iter.meanLoss.toFixed(3)}</span></div>`;
    }
    el.innerHTML = h;
  }
  function updateDetail() {}

  // ---------------- controls ----------------
  function init(controlsEl, legendEl) {
    controlsEl.innerHTML = '';

    const g0 = grp(controlsEl, 'Dataset');
    const sel = document.createElement('div');
    sel.className = 'ctl-row';
    sel.innerHTML = `<label>data</label><select class="ctl-select">` +
      Object.keys(ML.DATASETS).map(k =>
        `<option value="${k}">${ML.DATASETS[k].name}</option>`).join('') + `</select>`;
    const selEl = sel.querySelector('select');
    selEl.value = cfg.ds;
    selEl.onchange = e => { cfg.ds = e.target.value; loadData(cfg.ds); refresh(); };
    g0.appendChild(sel);

    const file = document.createElement('label');
    file.className = 'btn';
    file.style.display = 'block';
    file.style.textAlign = 'center';
    file.innerHTML = `↑ Load CSV<input type="file" accept=".csv,.txt" hidden>`;
    file.querySelector('input').onchange = ev => {
      const f = ev.target.files[0];
      if (!f) return;
      const rd = new FileReader();
      rd.onload = () => {
        try {
          setData(ML.parseCSV(String(rd.result), f.name));
          cfg.ds = 'csv';
          refresh();
          status(`loaded ${ds.X.length} rows · ${ds.nIn} features · ${ds.task === 'clf' ? ds.nOut + ' classes' : 'regression'}`);
        } catch (err) { status('could not parse: ' + err.message, true); }
      };
      rd.readAsText(f);
    };
    g0.appendChild(file);
    const stat = document.createElement('div');
    stat.className = 'shape-note';
    stat.dataset.id = 'tstatus';
    g0.appendChild(stat);

    const g1 = grp(controlsEl, 'Model');
    list(g1, 'hidden', () => cfg.nHid, v => cfg.nHid = v, [4, 6, 8, 12, 16], true);
    const ar = document.createElement('div');
    ar.className = 'ctl-row';
    ar.innerHTML = `<label>activation</label><select class="ctl-select">
      <option value="relu">ReLU</option><option value="tanh">tanh</option><option value="sigmoid">sigmoid</option></select>`;
    ar.querySelector('select').value = cfg.act;
    ar.querySelector('select').onchange = e => { cfg.act = e.target.value; resetNet(); refresh(); };
    g1.appendChild(ar);

    const g2 = grp(controlsEl, 'Optimization');
    list(g2, 'learning rate', () => cfg.lr, v => cfg.lr = v, [0.01, 0.03, 0.1, 0.3, 1.0], false);
    list(g2, 'batch size', () => cfg.batch, v => cfg.batch = v, [4, 8, 16, 32], false);

    const bTurbo = document.createElement('button');
    bTurbo.className = 'btn';
    bTurbo.textContent = '⚡ Train 200 steps';
    bTurbo.onclick = () => { turbo(200); App.resetTimeline(); };
    controlsEl.appendChild(bTurbo);

    const bTurbo2 = document.createElement('button');
    bTurbo2.className = 'btn';
    bTurbo2.textContent = '⚡⚡ Train 2000 steps';
    bTurbo2.onclick = () => { turbo(2000); App.resetTimeline(); };
    controlsEl.appendChild(bTurbo2);

    const bReset = document.createElement('button');
    bReset.className = 'btn';
    bReset.textContent = '⟲ Reset weights';
    bReset.onclick = () => { netSeed = (netSeed * 16807 + 23) % 2147483647; resetNet(); App.resetTimeline(); };
    controlsEl.appendChild(bReset);

    legendEl.innerHTML =
      `<div class="legend-row"><span class="legend-swatch" style="background:#eda100"></span><span>gradient / backward pass</span></div>` +
      `<div class="legend-row" style="margin-top:6px"><span>weight or activation</span></div>` +
      (() => { const s2 = []; for (let k = 0; k <= 10; k++) s2.push(U.divColor(U.lerp(-1, 1, k / 10), 1));
        return `<div class="legend-ramp" style="background:linear-gradient(90deg,${s2.join(',')})"></div>` +
               `<div class="legend-cap"><span>−</span><span>+</span></div>`; })() +
      (ds.task === 'clf' && ds.classes
        ? `<div class="legend-row" style="margin-top:10px"><span style="font-size:10.5px;color:#898781">classes:</span></div>` +
          ds.classes.map((c, k) =>
            `<div class="legend-row"><span class="legend-swatch" style="background:${U.CAT[k % U.CAT.length]}"></span><span>${c}</span></div>`).join('')
        : '');

    status(ds.synthetic
      ? `${ds.name} — <b>synthetic</b>, ${ds.X.length} rows<br>generated locally (no network)`
      : `${ds.name} — ${ds.X.length} rows · ${ds.nIn} features`);
  }

  function status(msg, err) {
    const el = document.querySelector('[data-id=tstatus]');
    if (el) { el.innerHTML = msg; el.style.color = err ? '#e34948' : ''; }
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
      set(v);
      val.textContent = get();
      if (doReset) resetNet(); else { buildSteps(); prepareIteration(); }
      App.resetTimeline();
    });
    g.appendChild(row);
  }

  loadData(cfg.ds);

  return {
    id: 'train',
    title: 'Train an MLP',
    desc: 'A real MLP trained with real SGD on a real dataset — watch a minibatch flow forward, the loss form, gradients flow backward, and the weights (and decision boundary) update. Press ▶ to keep training.',
    VW, VH, loop: true,
    get steps() { return steps; },
    init, regen: () => resetNet(), render, caption, buildDetail, updateDetail, onLoop
  };
})();
