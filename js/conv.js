'use strict';
/* ============ Convolution 2D module ============ */
const ConvModule = (() => {

  const VW = 1280, VH = 640;            // design space; main.js scales to fit

  const cfg = { B: 4, C: 3, H: 6, W: 6, F: 2, K: 3, S: 1, P: 1, G: 1 };
  let seed = 7;

  let X = [], Wt = [], bias = [], Out = [], absOut = 1;
  let Hp = 0, Wp = 0, Ho = 0, Wo = 0;
  let Cg = 3, Fg = 2;                   // channels / filters per group
  let steps = [];

  // phase boundaries inside a 'slide' step
  const PH = { move: 0.18, mult: 0.62, sum: 0.80, write: 1.0 };

  // ---------------- data ----------------
  function regen() {
    const rnd = U.mulberry32(seed);
    const ri = (lo, hi) => lo + Math.floor(rnd() * (hi - lo + 1));

    while (cfg.C % cfg.G || cfg.F % cfg.G) cfg.G--;   // G must divide C and F
    Cg = cfg.C / cfg.G; Fg = cfg.F / cfg.G;

    Hp = cfg.H + 2 * cfg.P; Wp = cfg.W + 2 * cfg.P;
    Ho = Math.floor((Hp - cfg.K) / cfg.S) + 1;
    Wo = Math.floor((Wp - cfg.K) / cfg.S) + 1;

    X = []; Wt = []; bias = []; Out = [];
    for (let b = 0; b < cfg.B; b++) {
      const chs = [];
      for (let c = 0; c < cfg.C; c++) {
        const g = [];
        for (let r = 0; r < cfg.H; r++) {
          const row = [];
          for (let cc = 0; cc < cfg.W; cc++) row.push(ri(0, 9));
          g.push(row);
        }
        chs.push(g);
      }
      X.push(chs);
    }
    for (let f = 0; f < cfg.F; f++) {
      const chs = [];
      let nz = false;
      for (let c = 0; c < Cg; c++) {
        const g = [];
        for (let r = 0; r < cfg.K; r++) {
          const row = [];
          for (let cc = 0; cc < cfg.K; cc++) { const w = ri(-2, 2); if (w) nz = true; row.push(w); }
          g.push(row);
        }
        chs.push(g);
      }
      if (!nz) chs[0][0][0] = 1;
      Wt.push(chs);
      bias.push(ri(-2, 2));
    }

    // padded value lookup for batch element 0
    // full output for color scaling + final states
    absOut = 1;
    for (let f = 0; f < cfg.F; f++) {
      const m = [];
      for (let or = 0; or < Ho; or++) {
        const row = [];
        for (let oc = 0; oc < Wo; oc++) {
          const v = convAt(f, or, oc).res;
          row.push(v);
          absOut = Math.max(absOut, Math.abs(v));
        }
        m.push(row);
      }
      Out.push(m);
    }
    buildSteps();
  }

  function xpad(c, r, col) { // padded coords, batch 0
    const rr = r - cfg.P, cc = col - cfg.P;
    if (rr < 0 || cc < 0 || rr >= cfg.H || cc >= cfg.W) return 0;
    return X[0][c][rr][cc];
  }

  function convAt(f, or, oc) {
    const prods = [];
    let sum = 0;
    const gBase = Math.floor(f / Fg) * Cg;  // first input channel this filter sees
    for (let c = 0; c < Cg; c++) {
      for (let kr = 0; kr < cfg.K; kr++) {
        for (let kc = 0; kc < cfg.K; kc++) {
          const xv = xpad(gBase + c, or * cfg.S + kr, oc * cfg.S + kc);
          const wv = Wt[f][c][kr][kc];
          prods.push({ xv, wv, p: xv * wv });
          sum += xv * wv;
        }
      }
    }
    return { prods, sum, res: sum + bias[f] };
  }

  function buildSteps() {
    steps = [{ type: 'intro', dur: 2.2 }];
    if (cfg.P > 0) steps.push({ type: 'pad', dur: 1.8 });
    for (let f = 0; f < cfg.F; f++) {
      steps.push({ type: 'filter', f, dur: 1.6 });
      let prev = null;
      for (let or = 0; or < Ho; or++) {
        for (let oc = 0; oc < Wo; oc++) {
          steps.push({ type: 'slide', f, or, oc, prev, dur: 2.4 });
          prev = { or, oc };
        }
      }
    }
    steps.push({ type: 'done', dur: 2.0 });
  }

  // ---------------- layout ----------------
  function layout(ctx) {
    const s = U.clamp(470 / (0.9 * (Wp + cfg.C) + 2), 16, 36);
    const inOx = 245 + cfg.C * s * 0.9;
    const cubeH = Hp * s * 0.92, cubeD = (Wp + cfg.C) * s * 0.45;
    const inOy = Math.min(105 + cubeH, VH - 120 - cubeD);
    const inIso = Iso.make(ctx, s, inOx, inOy);
    const inRight = inOx + Wp * s * 0.9;

    const ks = 22;
    const kIso = Iso.make(ctx, ks, inRight + 95 + Cg * ks * 0.9, 315);
    const kRight = kIso.ox + cfg.K * ks * 0.9;

    const os = U.clamp(340 / (0.9 * (Wo + cfg.F) + 2), 16, 30);
    const oIso = Iso.make(ctx, os, kRight + 120 + cfg.F * os * 0.9,
                          115 + Ho * os * 0.92);
    return { inIso, kIso, oIso, inRight, kRight };
  }

  // state helpers -----------------------------------------------------------
  function padAlphaAt(si, t) {
    if (cfg.P === 0) return 0;
    const padIdx = 1;
    if (si < padIdx) return 0;
    if (si === padIdx && steps[si].type === 'pad') return U.easeInOut(t);
    return 1;
  }
  function doneCells(si, f) { // # of output cells completed for filter f before step si
    const st = steps[si];
    let n = 0;
    if (st.type === 'done') return Ho * Wo;
    for (let i = 0; i < si; i++) {
      const s2 = steps[i];
      if (s2.type === 'slide' && s2.f === f) n++;
    }
    return n;
  }
  function completedFilters(si) {
    let n = 0;
    for (let f = 0; f < cfg.F; f++) if (doneCells(si, f) >= Ho * Wo) n++;
    return n;
  }

  // ---------------- render ----------------
  function render(ctx, si, tRaw) {
    const st = steps[si];
    const t = tRaw;
    const L = layout(ctx);
    const padA = padAlphaAt(si, t);

    drawBatch(ctx, si, t);
    drawOpSymbols(ctx, L);
    drawInput(ctx, L.inIso, si, t, padA);
    drawKernelArea(ctx, L.kIso, si, t);
    drawOutput(ctx, L.oIso, si, t);
    if (st.type === 'slide') drawFly(ctx, L, st, t);
  }

  function filterColor(f) { return U.CAT[f % U.CAT.length]; }

  // --- batch mini-cubes ---
  function drawBatch(ctx, si, t) {
    const s = 6.5;
    const gap = (cfg.W + cfg.C) * s * 0.9 + 26;
    for (let b = cfg.B - 1; b >= 0; b--) {
      const iso = Iso.make(ctx, s, 78 + cfg.C * s * 0.9, 96 + b * (cfg.H * s * 0.92 + (cfg.W + cfg.C) * s * 0.45 + 24));
      const active = b === 0;
      Iso.box(iso, 0, 0, 0, cfg.W, cfg.H, cfg.C, active ? '#2a4a74' : '#333842',
              { alpha: active ? 1 : 0.95 });
      if (active) Iso.wire(iso, 0, 0, 0, cfg.W, cfg.H, cfg.C, '#3987e5', 1.6, 0.9);
      ctx.save();
      ctx.fillStyle = active ? '#c3c2b7' : '#898781';
      ctx.font = '500 10.5px system-ui, sans-serif';
      const [lx, ly] = iso.p(cfg.W + 0.5, cfg.H * 0.5, cfg.C);
      ctx.fillText('b=' + b, lx + 4, ly);
      ctx.restore();
    }
    ctx.save();
    ctx.fillStyle = '#898781';
    ctx.font = '600 11px ui-monospace, Menlo, monospace';
    ctx.fillText(`input batch  (B,C,H,W) = (${cfg.B},${cfg.C},${cfg.H},${cfg.W})`, 24, 52);
    ctx.fillStyle = '#c3c2b7';
    ctx.font = '500 11px system-ui, sans-serif';
    ctx.fillText('conv runs on every element —', 24, 68);
    ctx.fillText('watching b=0', 24, 82);
    ctx.restore();
  }

  function drawOpSymbols(ctx, L) {
    ctx.save();
    ctx.fillStyle = '#898781';
    ctx.font = '300 30px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('⊛', L.kIso.ox - Cg * L.kIso.ux - 48, 325);
    ctx.fillText('=', L.kRight + 58, 325);
    ctx.restore();
  }

  // --- main input cube ---
  function currentWindow(si, t) {
    const st = steps[si];
    if (!st || st.type !== 'slide') return null;
    const tm = Math.min(t / PH.move, 1);
    const e = U.easeInOut(tm);
    const pc = st.prev ? st.prev.oc * cfg.S : st.oc * cfg.S;
    const pr = st.prev ? st.prev.or * cfg.S : st.or * cfg.S;
    return {
      col: U.lerp(pc, st.oc * cfg.S, e),
      row: U.lerp(pr, st.or * cfg.S, e),
      f: st.f, settled: t >= PH.move
    };
  }

  function drawInput(ctx, iso, si, t, padA) {
    const C = cfg.C;
    // base body (core box always, padded wire fades in)
    Iso.box(iso, cfg.P, cfg.P, 0, cfg.W, cfg.H, C, '#1f232b');
    if (cfg.P > 0 && padA > 0)
      Iso.wire(iso, 0, 0, 0, Wp, Hp, C, 'rgba(195,194,183,0.5)', 1, padA * 0.8);

    const isPad = (r, c) => r < cfg.P || c < cfg.P || r >= cfg.P + cfg.H || c >= cfg.P + cfg.W;

    // pre-padding shell: the core's own top/right value cells, fading out as
    // the padded shell fades in
    if (cfg.P > 0 && padA < 1) {
      const a = 1 - padA;
      for (let d = 0; d < C; d++) {
        const ch = C - 1 - d;
        for (let c = 0; c < cfg.W; c++)
          Iso.topCell(iso, cfg.P, 0, cfg.P + cfg.H, c, d,
                      U.shade(U.seqColor(X[0][ch][0][c] / 9), 1.25), { alpha: a });
        for (let r = 0; r < cfg.H; r++)
          Iso.rightCell(iso, cfg.P, 0, cfg.P + cfg.W, d, r, cfg.H,
                        U.shade(U.seqColor(X[0][ch][r][cfg.W - 1] / 9), 0.62), { alpha: a });
      }
    }

    // top face: row r=0 of each channel (depth d -> channel C-1-d)
    for (let d = 0; d < C; d++) {
      const ch = C - 1 - d;
      for (let c = 0; c < Wp; c++) {
        const pad = isPad(0, c);
        if (pad && padA <= 0) continue;
        const v = xpad(ch, 0, c);
        Iso.topCell(iso, 0, 0, Hp, c, d, pad ? '#2a2a28' : U.shade(U.seqColor(v / 9), 1.25),
                    { alpha: pad ? padA * 0.9 : 1 });
      }
    }
    // right face: col Wp-1 of each channel
    for (let d = 0; d < C; d++) {
      const ch = C - 1 - d;
      for (let r = 0; r < Hp; r++) {
        const pad = isPad(r, Wp - 1);
        if (pad && padA <= 0) continue;
        const v = xpad(ch, r, Wp - 1);
        Iso.rightCell(iso, 0, 0, Wp, C - 1 - d, r, Hp, pad ? '#222220' : U.shade(U.seqColor(v / 9), 0.62),
                      { alpha: pad ? padA * 0.9 : 1 });
      }
    }
    // front face: channel 0 (z = C plane)
    for (let r = 0; r < Hp; r++) {
      for (let c = 0; c < Wp; c++) {
        const pad = isPad(r, c);
        if (pad && padA <= 0) continue;
        const v = xpad(0, r, c);
        if (pad) {
          Iso.cell(iso, 0, 0, C, c, r, Hp, '#242423', '0',
                   { alpha: padA, dashed: true, stroke: 'rgba(137,135,129,0.55)', ink: '#898781' });
        } else {
          Iso.cell(iso, 0, 0, C, c, r, Hp, U.seqColor(v / 9), String(v));
        }
      }
    }

    // group boundaries along the depth axis
    if (cfg.G > 1) {
      for (let g = 0; g < cfg.G; g++)
        Iso.wire(iso, 0, 0, C - (g + 1) * Cg, Wp, Hp, Cg,
                 U.withAlpha('#c3c2b7', 0.4), 1, 0.8);
    }

    // sliding window — spans only the depth slice its filter's group sees
    const win = currentWindow(si, t);
    if (win) {
      const fc = filterColor(win.f);
      const g = Math.floor(win.f / Fg);
      const z0 = C - (g + 1) * Cg;      // channel gBase+Cg-1 sits deepest
      const zf = z0 + Cg;               // front plane of this group's slice
      const yb = Hp - win.row - cfg.K;
      Iso.wire(iso, win.col, yb, z0, cfg.K, cfg.K, Cg, fc, 2, 0.85);
      const q = [iso.p(win.col, yb, zf), iso.p(win.col + cfg.K, yb, zf),
                 iso.p(win.col + cfg.K, yb + cfg.K, zf), iso.p(win.col, yb + cfg.K, zf)];
      Iso.poly(ctx, q, U.withAlpha(fc, win.settled ? 0.20 : 0.30), fc, 2.5);
    }

    Iso.label(iso, Wp * 0.5, C, cfg.P > 0 && padA > 0.05
      ? `padded input  (${cfg.C}×${Hp}×${Wp}) — front face = channel 0`
      : `input b=0  (${cfg.C}×${cfg.H}×${cfg.W}) — front face = channel 0`, { dy: 8 });
  }

  // --- kernel area ---
  function multIndex(t) { // index of product being multiplied during mult phase
    const n = Cg * cfg.K * cfg.K;
    if (t < PH.move) return -1;
    if (t >= PH.mult) return n;
    const tm = (t - PH.move) / (PH.mult - PH.move);
    return Math.min(Math.floor(tm * n), n - 1);
  }

  function activeFilter(si) {
    const st = steps[si];
    if (st.type === 'slide' || st.type === 'filter') return st.f;
    if (st.type === 'done') return cfg.F - 1;
    return 0;
  }

  function drawKernelArea(ctx, iso, si, t) {
    const st = steps[si];
    const f = activeFilter(si);
    const fc = filterColor(f);
    const K = cfg.K;
    const gBase = Math.floor(f / Fg) * Cg;

    let ch = 0;
    if (st.type === 'slide') {
      const mi = multIndex(t);
      ch = U.clamp(Math.floor(mi / (K * K)), 0, Cg - 1);
    }

    let pop = 1;
    if (st.type === 'filter') pop = 0.75 + 0.25 * U.easeOut(t);

    ctx.save();
    ctx.translate(iso.ox, iso.oy);
    ctx.scale(pop, pop);
    ctx.translate(-iso.ox, -iso.oy);

    Iso.box(iso, 0, 0, 0, K, K, Cg, '#242830');
    for (let r = 0; r < K; r++)
      for (let c = 0; c < K; c++) {
        const v = Wt[f][ch][r][c];
        Iso.cell(iso, 0, 0, Cg, c, r, K, U.divColor(v, 2), (v > 0 ? '+' : '') + v);
      }
    for (let d = 0; d < Cg; d++)
      for (let c = 0; c < K; c++)
        Iso.topCell(iso, 0, 0, K, c, d, U.shade(U.divColor(Wt[f][Cg - 1 - d][0][c], 2), 1.2));
    for (let d = 0; d < Cg; d++)
      for (let r = 0; r < K; r++)
        Iso.rightCell(iso, 0, 0, K, Cg - 1 - d, r, K, U.shade(U.divColor(Wt[f][Cg - 1 - d][r][K - 1], 2), 0.62));
    Iso.wire(iso, 0, 0, 0, K, K, Cg, U.withAlpha(fc, 0.9), 2);
    ctx.restore();

    Iso.label(iso, K * 0.5, Cg, `filter ${f + 1}/${cfg.F}  (${Cg}×${K}×${K})`, { dy: 8, color: fc, weight: 600 });
    Iso.label(iso, K * 0.5, Cg, cfg.G > 1
      ? `group ${Math.floor(f / Fg)} · ch ${gBase}–${gBase + Cg - 1} · bias ${bias[f] >= 0 ? '+' : ''}${bias[f]}`
      : `bias ${bias[f] >= 0 ? '+' : ''}${bias[f]}   ·   showing ch ${ch}`, { dy: 24 });

    // filter chips
    ctx.save();
    ctx.textAlign = 'left';
    for (let i = 0; i < cfg.F; i++) {
      const [bx, by] = [iso.ox - Cg * iso.ux - 8, iso.oy - K * iso.h - 46];
      const y = by - 24 + i * 20;
      ctx.fillStyle = i === f ? U.withAlpha(filterColor(i), 0.25) : 'transparent';
      ctx.strokeStyle = U.withAlpha(filterColor(i), i === f ? 1 : 0.45);
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(bx, y, 74, 17, 5); else ctx.rect(bx, y, 74, 17);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = i === f ? '#ffffff' : '#898781';
      ctx.font = '600 10px ui-monospace, Menlo, monospace';
      ctx.fillText('filter ' + (i + 1), bx + 8, y + 12);
    }
    ctx.restore();
  }

  // --- output ---
  function drawOutput(ctx, iso, si, t) {
    const st = steps[si];
    const f = activeFilter(si);
    const nDoneFilters = Math.min(completedFilters(si), st.type === 'done' ? cfg.F : cfg.F - 1);

    // completed maps stacked behind
    let zi = 0;
    for (let i = 0; i < cfg.F; i++) {
      if (i === f && st.type !== 'done') continue;
      if (doneCells(si, i) < Ho * Wo && !(st.type === 'done')) continue;
      if (st.type === 'done' && i === f) continue;
      Iso.box(iso, 0, 0, zi, Wo, Ho, 1, U.shade(filterColor(i), 0.45), { alpha: 0.9 });
      Iso.wire(iso, 0, 0, zi, Wo, Ho, 1, U.withAlpha(filterColor(i), 0.6), 1);
      zi++;
    }

    // active map (front)
    const zFront = zi;
    const nd = doneCells(si, f);
    Iso.box(iso, 0, 0, zFront, Wo, Ho, 1, '#1f232b');
    let k = 0;
    for (let or = 0; or < Ho; or++) {
      for (let oc = 0; oc < Wo; oc++) {
        const idx = or * Wo + oc;
        let show = idx < nd;
        let isCurrent = false;
        if (st.type === 'slide' && st.f === f && st.or === or && st.oc === oc) {
          isCurrent = true;
          if (t >= PH.write + (1 - PH.write) * 0.65) show = true;
        }
        if (st.type === 'done') show = true;
        if (show) {
          const v = Out[f][or][oc];
          Iso.cell(iso, 0, 0, zFront + 1, oc, or, Ho, U.divColor(v, absOut), U.fmt(v));
        } else {
          Iso.cell(iso, 0, 0, zFront + 1, oc, or, Ho, '#20201f', null, { stroke: 'rgba(255,255,255,0.07)' });
        }
        if (isCurrent) {
          const q = Iso.cellQuad(iso, 0, 0, zFront + 1, oc, or, Ho);
          Iso.poly(ctx, q, null, filterColor(f), 2.4);
        }
        k++;
      }
    }
    Iso.wire(iso, 0, 0, zFront, Wo, Ho, 1, U.withAlpha(filterColor(f), 0.85), 1.6);

    Iso.label(iso, Wo * 0.5, zFront + 1,
      st.type === 'done' ? `output  (${cfg.F}×${Ho}×${Wo}) — one map per filter`
                         : `feature map — filter ${f + 1}  (${Ho}×${Wo})`, { dy: 8 });
    Iso.label(iso, Wo * 0.5, zFront + 1, `output (B,F,Ho,Wo) = (${cfg.B},${cfg.F},${Ho},${Wo})`,
              { dy: 24 });
  }

  // --- flying value chip ---
  function drawFly(ctx, L, st, t) {
    if (t < PH.sum) return;
    const { res } = convAt(st.f, st.or, st.oc);
    const fc = filterColor(st.f);

    const ybC = Hp - st.or * cfg.S - cfg.K;
    const [x0, y0] = L.inIso.p(st.oc * cfg.S + cfg.K / 2, ybC + cfg.K / 2, cfg.C);
    const zFront = st.f; // one completed slab per previous filter
    const [ox, oy] = Iso.cellCenter(L.oIso, 0, 0, zFront + 1, st.oc, st.or, Ho);

    if (t < PH.write) {
      // sum collapse pulse at window
      const tp = (t - PH.sum) / (PH.write - PH.sum);
      chip(ctx, x0, y0 - 14 * U.easeOut(tp), res, fc, 0.4 + 0.6 * tp);
    } else {
      const tp = U.easeInOut((t - PH.write) / (1 - PH.write));
      const mx = (x0 + ox) / 2, my = Math.min(y0, oy) - 90;
      const bx = (1 - tp) * (1 - tp) * x0 + 2 * (1 - tp) * tp * mx + tp * tp * ox;
      const by = (1 - tp) * (1 - tp) * (y0 - 14) + 2 * (1 - tp) * tp * my + tp * tp * oy;
      ctx.save();
      ctx.strokeStyle = U.withAlpha(fc, 0.35 * (1 - tp) + 0.1);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x0, y0 - 14);
      ctx.quadraticCurveTo(mx, my, ox, oy);
      ctx.stroke();
      ctx.restore();
      if (tp < 0.98) chip(ctx, bx, by, res, fc, 1);
    }
  }

  function chip(ctx, x, y, v, color, alpha) {
    const txt = U.fmt(v);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.font = '650 13px ui-monospace, Menlo, monospace';
    const w = ctx.measureText(txt).width + 16;
    ctx.fillStyle = '#111214';
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x - w / 2, y - 12, w, 24, 8); else ctx.rect(x - w / 2, y - 12, w, 24);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(txt, x, y + 0.5);
    ctx.restore();
  }

  // ---------------- captions ----------------
  function caption(si, t) {
    const st = steps[si];
    switch (st.type) {
      case 'intro':
        return `A batch of <b>${cfg.B}</b> inputs, each <b>${cfg.C}×${cfg.H}×${cfg.W}</b> (channels × height × width). The convolution below runs identically on every batch element.`;
      case 'pad':
        return `<b>Padding P=${cfg.P}</b> — a border of zeros around every channel: ${cfg.H}×${cfg.W} → ${Hp}×${Wp}.`;
      case 'filter': {
        const g = Math.floor(st.f / Fg), a = g * Cg;
        return cfg.G > 1
          ? `<b>Filter ${st.f + 1}/${cfg.F}</b> — in <b>group ${g}</b>: it only sees channels <b>${a}–${a + Cg - 1}</b>, so its weights are just ${Cg}×${cfg.K}×${cfg.K} (${cfg.G}× fewer than a full filter).`
          : `<b>Filter ${st.f + 1}/${cfg.F}</b> — its own ${cfg.C}×${cfg.K}×${cfg.K} weights + bias. It slides across the input; each stop yields <b>one number</b>.`;
      }
      case 'slide': {
        if (t < PH.move) return `Slide the ${cfg.K}×${cfg.K}×${Cg} window to output position <b>(${st.or}, ${st.oc})</b> (stride ${cfg.S})`;
        if (t < PH.mult) return `Multiply window and filter <b>element-by-element</b> — all ${Cg}·${cfg.K}·${cfg.K} = ${Cg * cfg.K * cfg.K} pairs${cfg.G > 1 ? ` (group ${Math.floor(st.f / Fg)}'s channels only)` : ''}`;
        if (t < PH.sum) { const { sum } = convAt(st.f, st.or, st.oc); return `Sum all products → <b>${sum}</b>, then add bias ${bias[st.f] >= 0 ? '+' : ''}${bias[st.f]}`; }
        const { res } = convAt(st.f, st.or, st.oc);
        return `One number out: <b>output[${st.f}][${st.or}][${st.oc}] = ${res}</b>`;
      }
      case 'done':
        return `Done — <b>(${cfg.F}, ${Ho}, ${Wo})</b> output: each filter produced its own feature map. Next stop: max-pooling.`;
    }
    return '';
  }

  // ---------------- detail panel ----------------
  let dp = null; // cached refs

  function buildDetail(el, si) {
    const st = steps[si];
    dp = null;
    if (st.type !== 'slide') {
      const msgs = {
        intro: 'The math panel will show every multiply-and-add as the window slides.',
        pad: 'Zero-padding keeps the output the same size (for K=3, P=1, S=1) and lets the filter see the borders.',
        filter: 'Each filter has one K×K weight grid per input channel — it mixes space and channels at once.',
        done: 'Every output value was one window ⊙ filter dot product. Try more filters, another stride, or switch to Max Pooling.'
      };
      el.innerHTML = `<div class="dp-idle">${msgs[st.type] || ''}</div>`;
      return;
    }
    const K = cfg.K;
    const gBase = Math.floor(st.f / Fg) * Cg;
    const d = convAt(st.f, st.or, st.oc);
    let html = `<div class="dp-title">output[<b>f=${st.f}</b>][<b>${st.or}</b>][<b>${st.oc}</b>]  =  Σ window ⊙ filter  +  bias` +
               (cfg.G > 1 ? `  ·  group ${Math.floor(st.f / Fg)} (ch ${gBase}–${gBase + Cg - 1})` : '') + `</div>`;
    html += '<div class="dp-flow">';
    for (let c = 0; c < Cg; c++) {
      if (c > 0) html += '<span class="dp-plus">+</span>';
      html += `<div class="dp-ch"><span class="dp-ch-label">ch ${gBase + c}</span>`;
      html += gridHTML(K, (r, cc) => {
        const v = xpad(gBase + c, st.or * cfg.S + r, st.oc * cfg.S + cc);
        return cellHTML(v, U.seqColor(v / 9), 'win');
      });
      html += '<span class="dp-op">⊙</span>';
      html += gridHTML(K, (r, cc) => {
        const v = Wt[st.f][c][r][cc];
        return cellHTML((v > 0 ? '+' : '') + v, U.divColor(v, 2), 'ker');
      });
      html += '<span class="dp-op">=</span>';
      html += gridHTML(K, (r, cc) => {
        const v = d.prods[c * K * K + r * K + cc].p;
        return cellHTML(v, U.divColor(v, 18), 'prod pending');
      });
      html += '</div>';
    }
    html += '</div>';
    html += `<div class="dp-sum">Σ products <span class="chip" data-id="sum">…</span>` +
            `<span>+ bias</span> <span class="chip">${bias[st.f] >= 0 ? '+' : ''}${bias[st.f]}</span>` +
            `<span>→</span> <span class="chip res" data-id="res">…</span></div>`;
    el.innerHTML = html;
    dp = {
      win: el.querySelectorAll('.dp-cell.win'),
      ker: el.querySelectorAll('.dp-cell.ker'),
      prod: el.querySelectorAll('.dp-cell.prod'),
      sum: el.querySelector('[data-id=sum]'),
      res: el.querySelector('[data-id=res]'),
      d
    };
  }

  function gridHTML(K, fn) {
    let h = `<div class="dp-grid" style="grid-template-columns:repeat(${K},27px)">`;
    for (let r = 0; r < K; r++) for (let c = 0; c < K; c++) h += fn(r, c);
    return h + '</div>';
  }
  function cellHTML(txt, bg, cls) {
    return `<div class="dp-cell ${cls}" style="background:${bg};color:${U.inkFor(bg)};border-color:rgba(0,0,0,.3)">${txt}</div>`;
  }

  function updateDetail(si, t) {
    const st = steps[si];
    if (!dp || st.type !== 'slide') return;
    const n = Cg * cfg.K * cfg.K;
    const mi = multIndex(t);
    for (let i = 0; i < n; i++) {
      const revealed = i <= mi || t >= PH.mult;
      dp.prod[i].classList.toggle('pending', !revealed);
      const hot = i === mi && t >= PH.move && t < PH.mult;
      dp.prod[i].classList.toggle('hot', hot);
      dp.win[i].classList.toggle('hot', hot);
      dp.ker[i].classList.toggle('hot', hot);
    }
    const showSum = t >= PH.mult;
    dp.sum.textContent = showSum ? String(dp.d.sum) : '…';
    const showRes = t >= PH.sum;
    dp.res.textContent = showRes ? String(dp.d.res) : '…';
    dp.res.classList.toggle('hot', showRes);
  }

  // ---------------- controls / legend ----------------
  function init(controlsEl, legendEl) {
    controlsEl.innerHTML = '';
    const g1 = ctlGroup(controlsEl, 'Input');
    ctlNum(g1, 'batch', 'B', () => cfg.B, v => { cfg.B = v; }, 1, 6);
    ctlNum(g1, 'channels', 'C', () => cfg.C, v => { cfg.C = v; }, 1, 4);
    ctlNum(g1, 'height', 'H', () => cfg.H, v => { cfg.H = v; }, 4, 8);
    ctlNum(g1, 'width', 'W', () => cfg.W, v => { cfg.W = v; }, 4, 8);
    const g2 = ctlGroup(controlsEl, 'Convolution');
    ctlNum(g2, 'filters', 'F', () => cfg.F, v => { cfg.F = v; }, 1, 4);
    ctlNum(g2, 'kernel', 'K', () => cfg.K, v => { cfg.K = v; }, 2, 4);
    ctlNum(g2, 'stride', 'S', () => cfg.S, v => { cfg.S = v; }, 1, 2);
    ctlNum(g2, 'padding', 'P', () => cfg.P, v => { cfg.P = v; }, 0, 2);
    ctlGroups(g2);

    const btn = document.createElement('button');
    btn.className = 'btn';
    btn.textContent = '⚄ Regenerate values';
    btn.onclick = () => { seed = (seed * 16807 + 11) % 2147483647; regen(); App.resetTimeline(); };
    controlsEl.appendChild(btn);

    const note = document.createElement('div');
    note.className = 'shape-note';
    note.dataset.id = 'shapes';
    controlsEl.appendChild(note);
    updateShapeNote();

    legendEl.innerHTML =
      legendRamp('input value', [0, 9], U.seqColor) +
      legendRamp('weight / product', [-2, 2], v => U.divColor(v, 2)) +
      `<div class="legend-row" style="margin-top:8px"><span style="font-size:10.5px;color:#898781">filters:</span>` +
      Array.from({ length: cfg.F }, (_, i) =>
        `<span class="legend-swatch" style="background:${filterColor(i)}"></span>`).join('') + '</div>';
  }

  function updateShapeNote() {
    const el = document.querySelector('[data-id=shapes]');
    if (!el) return;
    el.innerHTML =
      `X <b>(${cfg.B},${cfg.C},${cfg.H},${cfg.W})</b><br>` +
      `&nbsp;→ pad <b>(${cfg.B},${cfg.C},${Hp},${Wp})</b><br>` +
      `&nbsp;→ conv <b>(${cfg.B},${cfg.F},${Ho},${Wo})</b><br>` +
      `Ho = ⌊(H+2P−K)/S⌋+1 = ${Ho}<br>` +
      (cfg.G > 1
        ? `groups <b>${cfg.G}</b>: filter is (${Cg}×${cfg.K}×${cfg.K})`
        : `filter is (${cfg.C}×${cfg.K}×${cfg.K})`);
    const gv = document.querySelector('[data-id=gval]');
    if (gv) gv.textContent = cfg.G;
  }

  // groups stepper — only divisors of both C and F are valid
  function ctlGroups(group) {
    const valid = () => {
      const out = [];
      for (let g = 1; g <= Math.min(cfg.C, cfg.F); g++)
        if (cfg.C % g === 0 && cfg.F % g === 0) out.push(g);
      return out;
    };
    const row = document.createElement('div');
    row.className = 'ctl-row';
    row.innerHTML = `<label>groups <span class="dim">G</span></label>` +
      `<span class="ctl-num"><button data-d="-1">−</button><span class="val" data-id="gval">${cfg.G}</span><button data-d="1">+</button></span>`;
    row.title = 'grouped convolution — G must divide both C and F';
    row.querySelectorAll('button').forEach(b => b.onclick = () => {
      const vs = valid();
      const i = vs.indexOf(cfg.G);
      cfg.G = vs[U.clamp(i + Number(b.dataset.d), 0, vs.length - 1)];
      regen(); updateShapeNote();
      App.resetTimeline();
    });
    group.appendChild(row);
  }

  function ctlGroup(parent, title) {
    const g = document.createElement('div');
    g.className = 'ctl-group';
    g.innerHTML = `<h3>${title}</h3>`;
    parent.appendChild(g);
    return g;
  }

  function ctlNum(group, label, dim, get, set, lo, hi) {
    const row = document.createElement('div');
    row.className = 'ctl-row';
    row.innerHTML = `<label>${label} <span class="dim">${dim}</span></label>` +
      `<span class="ctl-num"><button data-d="-1">−</button><span class="val">${get()}</span><button data-d="1">+</button></span>`;
    const val = row.querySelector('.val');
    row.querySelectorAll('button').forEach(b => b.onclick = () => {
      let v = U.clamp(get() + Number(b.dataset.d), lo, hi);
      set(v);
      // keep output valid: K must fit in padded input
      cfg.K = Math.min(cfg.K, Math.min(cfg.H, cfg.W) + 2 * cfg.P);
      regen(); updateShapeNote();
      val.textContent = get();
      App.resetTimeline();
    });
    group.appendChild(row);
  }

  function legendRamp(name, [lo, hi], fn) {
    const stops = [];
    for (let i = 0; i <= 10; i++) stops.push(fn(U.lerp(lo, hi, i / 10)));
    return `<div class="legend-row"><span>${name}</span></div>` +
      `<div class="legend-ramp" style="background:linear-gradient(90deg, ${stops.join(',')})"></div>` +
      `<div class="legend-cap"><span>${lo}</span><span>${hi}</span></div>`;
  }

  regen();

  return {
    id: 'conv',
    title: 'Convolution 2D',
    desc: 'A filter of weights slides over the padded input. At every stop: multiply the overlapping values element-wise, sum all of them, add the bias — that single number goes to the output feature map.',
    VW, VH,
    get steps() { return steps; },
    init, regen, render, caption, buildDetail, updateDetail
  };
})();
