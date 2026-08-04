'use strict';
/* ============ Lesson 3: gradient descent ============
 * The idea behind all training, stripped of neural networks:
 * stand on a surface, feel which way is downhill, take a step. Repeat.
 */
const GDModule = (() => {

  const VW = 1280, VH = 600;
  const cfg = { land: 'bowl1d', lr: 0.1 };

  const LAND = {
    bowl1d: {
      dim: 1, name: 'a simple bowl', tex: 'f(x) = x²',
      f: x => x * x, df: x => 2 * x,
      dom: [-3.2, 3.2], start: 2.6,
      note: 'One minimum, right in the middle. Any starting point rolls to it.'
    },
    bumpy1d: {
      dim: 1, name: 'bumpy — local minima', tex: 'f(x) = 0.3x² + sin(2x)',
      f: x => 0.3 * x * x + Math.sin(2 * x),
      df: x => 0.6 * x + 2 * Math.cos(2 * x),
      dom: [-5, 5], start: 3.6,
      note: 'Several valleys. Where you start decides which one you fall into.'
    },
    bowl2d: {
      dim: 2, name: 'round bowl (2 weights)', tex: 'f(x,y) = x² + y²',
      f: (x, y) => x * x + y * y, g: (x, y) => [2 * x, 2 * y],
      dom: [-3, 3], start: [2.3, 1.9],
      note: 'Two knobs to tune at once. The gradient points straight at the bottom.'
    },
    ravine2d: {
      dim: 2, name: 'narrow ravine (2 weights)', tex: 'f(x,y) = 0.15x² + 3y²',
      f: (x, y) => 0.15 * x * x + 3 * y * y, g: (x, y) => [0.3 * x, 6 * y],
      dom: [-3, 3], start: [2.6, 1.5],
      note: 'Steep across, shallow along. Plain gradient descent zig-zags here.'
    },
  };

  const LRS = [0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.9, 1.05];

  const steps = [
    { type: 'look',  dur: 1.7 },
    { type: 'slope', dur: 2.4 },
    { type: 'step',  dur: 2.2 },
  ];

  let pos, next, grad, path = [], fhist = [], nsteps = 0, diverged = false;
  let cmap = null, cmapKey = '';

  const L = () => LAND[cfg.land];
  const is2D = () => L().dim === 2;

  // ---------------- math ----------------
  const fAt = p => is2D() ? L().f(p[0], p[1]) : L().f(p);
  const gAt = p => is2D() ? L().g(p[0], p[1]) : L().df(p);

  function reset(keepLand) {
    const l = L();
    pos = is2D() ? l.start.slice() : l.start;
    path = [clone(pos)];
    fhist = [fAt(pos)];
    nsteps = 0; diverged = false;
    prepare();
  }
  const clone = p => Array.isArray(p) ? p.slice() : p;

  function prepare() {
    grad = gAt(pos);
    next = is2D()
      ? [pos[0] - cfg.lr * grad[0], pos[1] - cfg.lr * grad[1]]
      : pos - cfg.lr * grad;
    if (!isFinite(fAt(next)) || magnitude(next) > span() * 6) diverged = true;
  }
  const magnitude = p => Array.isArray(p) ? Math.hypot(p[0], p[1]) : Math.abs(p);
  const span = () => L().dom[1] - L().dom[0];

  function onLoop() {
    if (diverged) return;             // frozen until reset
    pos = next;
    path.push(clone(pos));
    fhist.push(fAt(pos));
    if (path.length > 300) path.shift();
    if (fhist.length > 300) fhist.shift();
    nsteps++;
    prepare();
  }

  function turbo(n) {
    for (let k = 0; k < n && !diverged; k++) onLoop();
  }

  // ---------------- layout ----------------
  const PLOT = { x: 48, y: 78, w: 776, h: 432 };
  const CURVE = { x: 872, y: 108, w: 364, h: 168 };

  const sx = v => PLOT.x + 40 + (v - L().dom[0]) / span() * (PLOT.w - 64);
  const sy2 = v => PLOT.y + 26 + (L().dom[1] - v) / span() * (PLOT.h - 62);   // 2D: y axis
  function fRange() {
    const [lo, hi] = L().dom;
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i <= 80; i++) {
      const v = L().f(lo + (hi - lo) * i / 80);
      mn = Math.min(mn, v); mx = Math.max(mx, v);
    }
    const pad = (mx - mn) * 0.12;
    return [mn - pad, mx + pad];
  }
  const syCurve = (v, r) => PLOT.y + PLOT.h - 34 - (v - r[0]) / (r[1] - r[0]) * (PLOT.h - 70);

  // ---------------- render ----------------
  function render(ctx, si, t) {
    panel(ctx, PLOT.x, PLOT.y, PLOT.w, PLOT.h);
    if (is2D()) draw2D(ctx, si, t); else draw1D(ctx, si, t);
    drawSidebar(ctx, si, t);

    txt(ctx, 48, 34, 'gradient descent — how a model actually improves', { size: 15, weight: 650, color: '#fff' });
    txt(ctx, 48, 56, `${L().name} · ${L().tex} · learning rate ${cfg.lr}`, { size: 11.5, mono: true });
  }

  // interpolated ball position during the 'step' phase
  function ballPos(si, t) {
    if (steps[si].type !== 'step' || diverged) return pos;
    const k = U.easeInOut(U.clamp(t / 0.85, 0, 1));
    return is2D()
      ? [U.lerp(pos[0], next[0], k), U.lerp(pos[1], next[1], k)]
      : U.lerp(pos, next, k);
  }

  // ---- 1-D: a curve you can see the slope of ----
  function draw1D(ctx, si, t) {
    const r = fRange(), [lo, hi] = L().dom;
    const st = steps[si];

    // axes
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.12)'; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(sx(lo), syCurve(0, r)); ctx.lineTo(sx(hi), syCurve(0, r));
    ctx.stroke();
    ctx.restore();

    // the curve
    ctx.save();
    ctx.strokeStyle = '#3987e5'; ctx.lineWidth = 2.4; ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = 0; i <= 240; i++) {
      const x = lo + (hi - lo) * i / 240;
      const y = syCurve(L().f(x), r);
      i ? ctx.lineTo(sx(x), y) : ctx.moveTo(sx(x), y);
    }
    ctx.stroke();
    ctx.restore();

    // where we've been
    ctx.save();
    for (let i = 0; i < path.length; i++) {
      const x = path[i];
      ctx.globalAlpha = 0.18 + 0.5 * (i / Math.max(1, path.length - 1));
      ctx.fillStyle = '#c3c2b7';
      ctx.beginPath(); ctx.arc(sx(x), syCurve(L().f(x), r), 2.4, 0, 7); ctx.fill();
    }
    ctx.restore();

    const bx = ballPos(si, t);
    const by = L().f(bx);

    // tangent line = the slope the ball feels
    if (st.type === 'slope' || st.type === 'step') {
      const m = L().df(pos);
      const a = st.type === 'slope' ? U.easeOut(U.clamp(t / 0.5, 0, 1)) : 1;
      const dx = span() * 0.16;
      ctx.save();
      ctx.globalAlpha = a;
      ctx.strokeStyle = '#eda100'; ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sx(pos - dx), syCurve(L().f(pos) - m * dx, r));
      ctx.lineTo(sx(pos + dx), syCurve(L().f(pos) + m * dx, r));
      ctx.stroke();
      ctx.restore();
      txt(ctx, sx(pos) + 14, syCurve(L().f(pos), r) - 26,
          `slope = ${fmt(m)}`, { mono: true, size: 11, color: '#eda100', weight: 600 });
      // downhill arrow along the x axis
      if (Math.abs(m) > 1e-6) {
        const dir = -Math.sign(m);
        const y0 = syCurve(r[0], r) + 16;
        arrow(ctx, sx(pos), y0, sx(pos) + dir * 54, y0, '#eda100');
        txt(ctx, sx(pos) + dir * 30, y0 - 12, 'downhill', { align: 'center', size: 9.5, color: '#eda100' });
      }
    }

    // the step itself
    if (st.type === 'step' && !diverged) {
      ctx.save();
      ctx.strokeStyle = U.withAlpha('#eda100', 0.5);
      ctx.setLineDash([4, 3]); ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(sx(pos), syCurve(L().f(pos), r));
      ctx.lineTo(sx(next), syCurve(L().f(next), r));
      ctx.stroke();
      ctx.restore();
    }

    // ball — pinned to the edge (with a marker) once it has flown off the chart
    const bxD = U.clamp(bx, lo, hi);
    const byD = U.clamp(L().f(bxD), r[0], r[1]);
    const off = bxD !== bx;
    ball(ctx, sx(bxD), syCurve(byD, r));
    if (off) {
      const dir = Math.sign(bx);
      arrow(ctx, sx(bxD), syCurve(byD, r) - 30, sx(bxD) + dir * 34, syCurve(byD, r) - 30, '#e34948');
      txt(ctx, sx(bxD) - dir * 8, syCurve(byD, r) - 44, 'off the chart →'.replace('→', dir > 0 ? '→' : '←'),
          { align: dir > 0 ? 'right' : 'left', size: 10, color: '#e34948', weight: 600 });
    }
    txt(ctx, sx(bxD), syCurve(byD, r) - 22, `x = ${fmt(bx)}`,
        { align: 'center', mono: true, size: 11, color: off ? '#e34948' : '#fff', weight: 600 });

    txt(ctx, PLOT.x + PLOT.w - 12, PLOT.y + PLOT.h - 12, 'click the curve to move the ball',
        { align: 'right', size: 10 });
    txt(ctx, sx(lo) - 8, syCurve(0, r), 'f', { align: 'right', size: 11, mono: true });
  }

  // ---- 2-D: a contour map, two weights at once ----
  function draw2D(ctx, si, t) {
    const [lo, hi] = L().dom;
    const key = cfg.land;
    if (!cmap || cmapKey !== key) {
      const G = 60;
      cmap = document.createElement('canvas');
      cmap.width = cmap.height = G;
      const c2 = cmap.getContext('2d');
      const img = c2.createImageData(G, G);
      let mx = 0;
      for (let iy = 0; iy < G; iy++)
        for (let ix = 0; ix < G; ix++)
          mx = Math.max(mx, L().f(lo + (hi - lo) * ix / (G - 1), hi - (hi - lo) * iy / (G - 1)));
      for (let iy = 0; iy < G; iy++)
        for (let ix = 0; ix < G; ix++) {
          const vx = lo + (hi - lo) * ix / (G - 1), vy = hi - (hi - lo) * iy / (G - 1);
          const band = Math.floor(Math.sqrt(L().f(vx, vy) / mx) * 9) / 9;   // banded = readable contours
          const col = U.seqColor(0.06 + 0.84 * band);
          const n = parseInt(col.slice(1), 16);
          const o = (iy * G + ix) * 4;
          img.data[o] = (n >> 16) & 255; img.data[o + 1] = (n >> 8) & 255;
          img.data[o + 2] = n & 255; img.data[o + 3] = 255;
        }
      c2.putImageData(img, 0, 0);
      cmapKey = key;
    }
    const px = PLOT.x + 40, py = PLOT.y + 26;
    const pw = PLOT.w - 64, ph = PLOT.h - 62;
    ctx.save();
    ctx.beginPath(); ctx.rect(px, py, pw, ph); ctx.clip();
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(cmap, px, py, pw, ph);
    ctx.restore();
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.strokeRect(px - 0.5, py - 0.5, pw + 1, ph + 1);
    ctx.restore();

    // path
    if (path.length > 1) {
      ctx.save();
      ctx.strokeStyle = U.withAlpha('#ffffff', 0.75); ctx.lineWidth = 1.6;
      ctx.beginPath();
      path.forEach((p, i) => i ? ctx.lineTo(sx(p[0]), sy2(p[1])) : ctx.moveTo(sx(p[0]), sy2(p[1])));
      ctx.stroke();
      ctx.fillStyle = 'rgba(255,255,255,0.6)';
      for (const p of path) { ctx.beginPath(); ctx.arc(sx(p[0]), sy2(p[1]), 1.8, 0, 7); ctx.fill(); }
      ctx.restore();
    }

    const st = steps[si];
    // gradient arrow
    if (st.type === 'slope' || st.type === 'step') {
      const a = st.type === 'slope' ? U.easeOut(U.clamp(t / 0.5, 0, 1)) : 1;
      const gm = Math.hypot(grad[0], grad[1]) || 1;
      const len = 64;
      ctx.save();
      ctx.globalAlpha = a * 0.55;
      arrow(ctx, sx(pos[0]), sy2(pos[1]),
            sx(pos[0]) + grad[0] / gm * len * (pw / PLOT.w), sy2(pos[1]) - grad[1] / gm * len * (ph / PLOT.h),
            '#e34948');
      ctx.restore();
      ctx.save();
      ctx.globalAlpha = a;
      arrow(ctx, sx(pos[0]), sy2(pos[1]),
            sx(pos[0]) - grad[0] / gm * len * (pw / PLOT.w), sy2(pos[1]) + grad[1] / gm * len * (ph / PLOT.h),
            '#eda100');
      ctx.restore();
      txt(ctx, sx(pos[0]) + 10, sy2(pos[1]) - 26, 'downhill = −gradient',
          { size: 10, color: '#eda100', weight: 600 });
    }

    const b = ballPos(si, t);
    ball(ctx, sx(U.clamp(b[0], lo, hi)), sy2(U.clamp(b[1], lo, hi)));
    txt(ctx, PLOT.x + PLOT.w - 12, PLOT.y + PLOT.h - 12, 'click the map to move the ball',
        { align: 'right', size: 10 });
    txt(ctx, px, PLOT.y + PLOT.h - 12, 'dark = low f (good) · bright = high f', { size: 10 });
  }

  function ball(ctx, x, y) {
    ctx.save();
    ctx.shadowColor = 'rgba(0,0,0,0.6)'; ctx.shadowBlur = 8;
    ctx.beginPath(); ctx.arc(x, y, 8, 0, 7);
    ctx.fillStyle = diverged ? '#e34948' : '#ffffff';
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = diverged ? '#e34948' : '#eda100'; ctx.lineWidth = 2; ctx.stroke();
    ctx.restore();
  }

  // ---- right column: f-vs-step curve + numbers ----
  function drawSidebar(ctx, si, t) {
    const C = CURVE;
    panel(ctx, C.x, C.y, C.w, C.h);
    txt(ctx, C.x, C.y - 12, 'f at each step — this is the "loss curve"',
        { size: 10.5, weight: 600, color: '#c3c2b7' });
    if (fhist.length > 1) {
      const hi = Math.max(...fhist), lo = Math.min(...fhist);
      const px = i => C.x + 10 + i / (fhist.length - 1) * (C.w - 20);
      const py = v => C.y + C.h - 14 - (v - lo) / Math.max(1e-9, hi - lo) * (C.h - 30);
      ctx.save();
      ctx.strokeStyle = '#3987e5'; ctx.lineWidth = 2; ctx.lineJoin = 'round';
      ctx.beginPath();
      fhist.forEach((v, i) => i ? ctx.lineTo(px(i), py(v)) : ctx.moveTo(px(i), py(v)));
      ctx.stroke();
      ctx.fillStyle = '#3987e5';
      ctx.beginPath(); ctx.arc(px(fhist.length - 1), py(fhist[fhist.length - 1]), 3, 0, 7); ctx.fill();
      ctx.restore();
      txt(ctx, C.x + 8, C.y + 12, fmt(hi), { mono: true, size: 9.5 });
      txt(ctx, C.x + 8, C.y + C.h - 12, fmt(lo), { mono: true, size: 9.5 });
    } else {
      txt(ctx, C.x + C.w / 2, C.y + C.h / 2, 'take a step…', { align: 'center', size: 11 });
    }

    // metric tiles
    const my = C.y + C.h + 30;
    const tiles = [['steps', String(nsteps)], ['f (the value we minimise)', fmt(fAt(pos))],
                   ['learning rate', String(cfg.lr)], ['regime', regime()]];
    tiles.forEach(([k, v], i) => {
      const tx = C.x + (i % 2) * 188, ty = my + Math.floor(i / 2) * 52;
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.09)';
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(tx, ty, 176, 44, 7); else ctx.rect(tx, ty, 176, 44);
      ctx.stroke();
      ctx.restore();
      txt(ctx, tx + 10, ty + 14, k, { size: 9.5 });
      const col = k === 'regime'
        ? (v === 'diverging' ? '#e34948' : v === 'overshooting' ? '#eda100' : v === 'crawling' ? '#898781' : '#0ca30c')
        : '#fff';
      txt(ctx, tx + 10, ty + 30, v, { mono: true, size: 13, color: col, weight: 600 });
    });

    // divergence banner
    if (diverged) {
      ctx.save();
      ctx.fillStyle = 'rgba(227,73,72,0.12)';
      ctx.strokeStyle = '#e34948'; ctx.lineWidth = 1.4;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(C.x, my + 116, C.w, 52, 8); else ctx.rect(C.x, my + 116, C.w, 52);
      ctx.fill(); ctx.stroke();
      ctx.restore();
      txt(ctx, C.x + 14, my + 134, '⚠  diverged — the steps got bigger, not smaller',
          { size: 11.5, weight: 650, color: '#e34948' });
      txt(ctx, C.x + 14, my + 152, 'lower the learning rate, then press Reset',
          { size: 10.5, color: '#c3c2b7' });
    }
  }

  function regime() {
    if (diverged) return 'diverging';
    if (fhist.length < 4) return '—';
    const recent = fhist.slice(Math.max(0, fhist.length - 6));
    const first = recent[0], last = recent[recent.length - 1];
    // NB: compare against an absolute tolerance — f can legitimately be
    // negative (the bumpy landscape bottoms out below zero), and a relative
    // "last > first * 1.02" test flips meaning once f goes negative.
    const scale = Math.max(1e-9, Math.abs(first), Math.abs(last));
    if (last > first + 0.02 * scale) return 'diverging';

    // direction reversals = bouncing back and forth across the valley
    const comp = p => is2D() ? p[1] : p;
    const P = path.slice(Math.max(0, path.length - 7));
    let flips = 0;
    for (let i = 2; i < P.length; i++) {
      const d1 = comp(P[i]) - comp(P[i - 1]);
      const d0 = comp(P[i - 1]) - comp(P[i - 2]);
      if (d1 * d0 < 0) flips++;
    }
    if (flips >= 2) return 'overshooting';

    const gm = is2D() ? Math.hypot(grad[0], grad[1]) : Math.abs(grad);
    if (gm < 5e-3) return 'converged';
    if (Math.abs(last - first) < 0.005 * scale) return 'crawling';
    return 'converging';
  }

  // ---------------- pointer: click to reposition ----------------
  function onPointer(x, y, type) {
    if (type !== 'down') return;
    const px = PLOT.x + 40, py = PLOT.y + 26, pw = PLOT.w - 64, ph = PLOT.h - 62;
    if (x < px - 40 || x > px + pw || y < py || y > py + ph) return;
    const [lo, hi] = L().dom;
    const vx = lo + (x - px) / pw * (hi - lo);
    if (is2D()) {
      const vy = hi - (y - py) / ph * (hi - lo);
      pos = [U.clamp(vx, lo, hi), U.clamp(vy, lo, hi)];
    } else {
      pos = U.clamp(vx, lo, hi);
    }
    path = [clone(pos)]; fhist = [fAt(pos)];
    nsteps = 0; diverged = false;
    prepare();
    App.resetTimeline();
  }

  // ---------------- helpers ----------------
  function panel(ctx, x, y, w, h) {
    ctx.save();
    ctx.fillStyle = '#141414';
    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, 9); else ctx.rect(x, y, w, h);
    ctx.fill(); ctx.stroke();
    ctx.restore();
  }
  function arrow(ctx, x0, y0, x1, y1, col) {
    const ang = Math.atan2(y1 - y0, x1 - x0);
    ctx.save();
    ctx.strokeStyle = ctx.fillStyle = col;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.translate(x1, y1); ctx.rotate(ang);
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(-8, -4.5); ctx.lineTo(-8, 4.5);
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
  const fmt = v => {
    if (!isFinite(v)) return '∞';
    const a = Math.abs(v);
    const s = a >= 1000 ? v.toExponential(1) : a >= 10 ? v.toFixed(2) : v.toFixed(3);
    return /^-0\.?0*$/.test(s) ? s.slice(1) : s;      // no "-0.000"
  };

  // ---------------- caption ----------------
  function caption(si, t) {
    const st = steps[si];
    if (diverged)
      return `<b>Diverged.</b> With learning rate ${cfg.lr} each step overshoots further than the last, so f grows instead of shrinking. Lower the rate and press Reset.`;
    switch (st.type) {
      case 'look':
        return `We're standing at <b>${is2D() ? `(${fmt(pos[0])}, ${fmt(pos[1])})` : `x = ${fmt(pos)}`}</b>, where f = <b>${fmt(fAt(pos))}</b>. Goal: find the lowest point — but we can only feel the ground right here.`;
      case 'slope':
        return is2D()
          ? `The <b>gradient</b> points straight uphill. Go the opposite way — that's the yellow arrow.`
          : `The <b>slope</b> here is ${fmt(L().df(pos))}. Positive slope means uphill to the right, so we move <b>left</b>. That's all a gradient tells you.`;
      case 'step':
        return `Take a step: <b>new = old − ${cfg.lr} × slope</b>. The learning rate decides how far. Too small and you crawl; too big and you fly past the bottom.`;
    }
    return '';
  }

  // ---------------- detail ----------------
  function buildDetail(el, si) {
    let h = `<div class="dp-title">the update rule — this is the whole algorithm</div><div class="dp-eq">`;
    if (is2D()) {
      h += `f(x,y) = ${L().tex.replace('f(x,y) = ', '')} &nbsp; at (${fmt(pos[0])}, ${fmt(pos[1])}) → <b>${fmt(fAt(pos))}</b><br>` +
           `∇f = (${fmt(grad[0])}, ${fmt(grad[1])})<br>` +
           `x ← ${fmt(pos[0])} − ${cfg.lr}·(${fmt(grad[0])}) = <b>${fmt(next[0])}</b><br>` +
           `y ← ${fmt(pos[1])} − ${cfg.lr}·(${fmt(grad[1])}) = <b>${fmt(next[1])}</b>`;
    } else {
      h += `x&nbsp;&nbsp;&nbsp;&nbsp;= ${fmt(pos)} &nbsp;&nbsp; f(x) = <b>${fmt(fAt(pos))}</b><br>` +
           `f′(x) = ${fmt(grad)} &nbsp;<span style="color:var(--ink-muted)">(the slope under the ball)</span><br>` +
           `step&nbsp;= −lr · f′(x) = −${cfg.lr} · ${fmt(grad)} = <b>${fmt(-cfg.lr * grad)}</b><br>` +
           `x ← ${fmt(pos)} ${(-cfg.lr * grad) >= 0 ? '+' : '−'} ${fmt(Math.abs(cfg.lr * grad))} = <b>${fmt(next)}</b>`;
    }
    h += `</div><div class="dp-note">${L().note} &nbsp;—&nbsp; ` +
         `In a real network there is one of these numbers <b>per weight</b> (thousands to billions), and f is the loss. ` +
         `The "Train an MLP" lesson runs exactly this rule on every weight at once.</div>`;
    el.innerHTML = h;
  }
  function updateDetail() {}

  // ---------------- controls ----------------
  function init(controlsEl, legendEl) {
    controlsEl.innerHTML = '';
    const g0 = grp(controlsEl, 'Landscape');
    const row = document.createElement('div');
    row.className = 'ctl-row';
    row.innerHTML = `<label>shape</label><select class="ctl-select">` +
      Object.keys(LAND).map(k => `<option value="${k}">${LAND[k].name}</option>`).join('') + `</select>`;
    row.querySelector('select').value = cfg.land;
    row.querySelector('select').onchange = e => {
      cfg.land = e.target.value; cmapKey = ''; reset(); App.resetTimeline();
    };
    g0.appendChild(row);

    const g1 = grp(controlsEl, 'Step size');
    const lrRow = document.createElement('div');
    lrRow.className = 'ctl-row';
    lrRow.innerHTML = `<label>learning rate</label>` +
      `<span class="ctl-num"><button data-d="-1">−</button><span class="val">${cfg.lr}</span><button data-d="1">+</button></span>`;
    const lrVal = lrRow.querySelector('.val');
    lrRow.querySelectorAll('button').forEach(b => b.onclick = () => {
      const i = LRS.indexOf(cfg.lr);
      const v = LRS[U.clamp(i + Number(b.dataset.d), 0, LRS.length - 1)];
      if (v === cfg.lr) return;
      cfg.lr = v; lrVal.textContent = v;
      diverged = false; prepare();
      App.resetTimeline();
    });
    g1.appendChild(lrRow);

    const hint = document.createElement('div');
    hint.className = 'shape-note';
    hint.innerHTML = `try <b>0.01</b> → crawls<br>try <b>0.3</b> → bounces<br>try <b>1.05</b> → blows up`;
    g1.appendChild(hint);

    const b30 = document.createElement('button');
    b30.className = 'btn';
    b30.textContent = '⚡ Take 30 steps';
    b30.onclick = () => { turbo(30); App.resetTimeline(); };
    controlsEl.appendChild(b30);

    const bR = document.createElement('button');
    bR.className = 'btn';
    bR.textContent = '⟲ Reset ball';
    bR.onclick = () => { reset(); App.resetTimeline(); };
    controlsEl.appendChild(bR);

    const tip = document.createElement('div');
    tip.className = 'shape-note';
    tip.innerHTML = `<b>Click the plot</b> to drop the ball somewhere else — on the bumpy landscape, where you start decides which valley you end up in.`;
    controlsEl.appendChild(tip);

    legendEl.innerHTML =
      `<div class="legend-row"><span class="legend-swatch" style="background:#eda100"></span><span>downhill direction</span></div>` +
      `<div class="legend-row"><span class="legend-swatch" style="background:#e34948"></span><span>uphill (the gradient)</span></div>`;
  }

  function grp(parent, title) {
    const g = document.createElement('div');
    g.className = 'ctl-group';
    g.innerHTML = `<h3>${title}</h3>`;
    parent.appendChild(g);
    return g;
  }

  reset();

  return {
    id: 'gd',
    title: 'Gradient Descent',
    desc: 'Training, without the neural network: put a ball on a curve, feel which way is downhill, step. The learning rate decides how far you step — and whether you ever arrive.',
    VW, VH, loop: true, interactive: true,
    get steps() { return steps; },
    init, regen: () => reset(), render, caption, buildDetail, updateDetail, onLoop, onPointer
  };
})();
