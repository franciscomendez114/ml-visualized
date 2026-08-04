'use strict';
/* ============ CNN pipeline module: [conv → max-pool] × L ============ */
const CNNModule = (() => {

  const VW = 1280, VH = 640;
  const cfg = {
    C: 3, H: 16, L: 3,
    layers: [ { K: 3, P: 1, F: 8 }, { K: 3, P: 1, F: 16 }, { K: 3, P: 1, F: 32 } ]
  };
  let seed = 77;

  const KS = [1, 3, 5];                      // allowed kernel sizes
  const FS = [2, 4, 8, 16, 32, 64];          // allowed filter counts

  let chain = [];        // stages: input, conv/pool per layer, computed in regen
  let flatN = 0, totalParams = 0, invalidAt = -1;
  let steps = [];
  let caches = [];       // offscreen cube renders, one per stage
  let slots = [];        // screen placement per stage

  // ---------------- chain / data ----------------
  function regen() {
    const rnd = U.mulberry32(seed);
    while (cfg.layers.length < cfg.L) {
      const last = cfg.layers[cfg.layers.length - 1];
      cfg.layers.push({ K: 3, P: 1, F: Math.min(64, last.F * 2) });
    }
    cfg.layers.length = cfg.L;

    chain = [];
    invalidAt = -1; totalParams = 0;
    let C = cfg.C, Hs = cfg.H, rf = 1, jump = 1;
    chain.push(stage('input', -1, C, Hs, rf, 0, rnd));

    for (let li = 0; li < cfg.L; li++) {
      const { K, P, F } = cfg.layers[li];
      const Hc = Hs + 2 * P - K + 1;               // conv stride 1
      if (Hc < 1) { invalidAt = li; break; }
      const params = F * (C * K * K + 1);
      totalParams += params;
      rf += (K - 1) * jump;
      chain.push(stage('conv', li, F, Hc, rf, params, rnd));
      C = F; Hs = Hc;
      if (Hs >= 2) {                                // max-pool 2×2 stride 2
        Hs = Math.floor(Hs / 2);
        rf += 1 * jump;
        jump *= 2;
        chain.push(stage('pool', li, C, Hs, rf, 0, rnd));
      } else {
        cfg.layers[li].poolSkipped = true;
      }
    }
    flatN = C * Hs * Hs;
    buildSteps();
    buildLayout();
    buildCaches();
  }

  function stage(kind, li, C, Hs, rf, params, rnd) {
    return {
      kind, li, C, Hs, rf, params,
      dz: Math.min(C, 8),
      // activation-map-looking texture: low-frequency sinusoid, random phases
      pa: 0.3 + rnd() * 0.6, pb: 0.3 + rnd() * 0.6,
      p1: rnd() * 6.28, p2: rnd() * 6.28
    };
  }
  const texVal = (st, r, c, d) =>
    4.5 * (1 + Math.sin(r * st.pa + d * 0.7 + st.p1) * Math.cos(c * st.pb + d * 0.4 + st.p2));
  // conv stages carry signed pre-activations in [-9, 9]; ReLU clamps them
  const texSigned = (st, r, c, d) =>
    9 * Math.sin(r * st.pa + d * 0.7 + st.p1) * Math.cos(c * st.pb + d * 0.4 + st.p2);

  // padding of the conv that CONSUMES this stage (ring drawn around it)
  function padNext(ci) {
    const next = chain[ci + 1];
    return next && next.kind === 'conv' ? cfg.layers[next.li].P : 0;
  }

  function buildSteps() {
    steps = [{ type: 'intro', dur: 2.4, rowKey: 'input' }];
    for (let ci = 1; ci < chain.length; ci++) {
      const st = chain[ci];
      steps.push({
        type: st.kind, ci, li: st.li,
        dur: st.kind === 'conv' ? 2.8 : 2.2,
        rowKey: st.kind + st.li
      });
      if (st.kind === 'conv')       // non-linearity between conv and pool
        steps.push({ type: 'relu', ci, li: st.li, dur: 1.9, rowKey: 'relu' + st.li });
    }
    if (invalidAt >= 0) steps.push({ type: 'invalid', li: invalidAt, dur: 2.5, rowKey: 'invalid' });
    else steps.push({ type: 'flat', dur: 2.2, rowKey: 'flatten' });
    steps.push({ type: 'done', dur: 2.0, rowKey: 'total' });
  }

  // ---------------- layout & caches ----------------
  const BASE_Y = 285;      // vertical center of the cube chain

  function buildLayout() {
    const n = chain.length;
    const flatW = 120, gap = 26, x0 = 46;
    const avail = VW - x0 - 40 - flatW - gap * n;
    const slotW = avail / n;
    slots = [];
    let x = x0;
    for (let ci = 0; ci < n; ci++) {
      const st = chain[ci];
      const P = padNext(ci);
      const Wt = st.Hs + 2 * P;                       // footprint incl. pad ring
      const wU = (Wt + st.dz) * 0.9;
      const hU = Wt * 0.92 + (Wt + st.dz) * 0.45;
      const s = Math.min(slotW / wU, 330 / hU);
      const cw = wU * s, chh = hU * s;
      slots.push({
        x: x + (slotW - cw) / 2, y: BASE_Y - chh / 2,
        s, cw, ch: chh, P, Wt,
        oxL: st.dz * s * 0.9 + 2,                     // iso origin inside cache
        oyL: Wt * s * 0.92 + 2
      });
      x += slotW + gap;
    }
    slots.flatX = x;                                   // flatten chip position
  }

  function buildCaches() {
    caches = [];
    for (let ci = 0; ci < chain.length; ci++) {
      const st = chain[ci], sl = slots[ci];
      const mk = mode => {
        const cv = document.createElement('canvas');
        cv.width = Math.ceil((sl.cw + 4) * 2);
        cv.height = Math.ceil((sl.ch + 4) * 2);
        const c2 = cv.getContext('2d');
        c2.scale(2, 2);
        drawCube(c2, Iso.make(c2, sl.s, sl.oxL, sl.oyL), st, sl.P, mode);
        return cv;
      };
      // conv stages get two renders: signed pre-activations and post-ReLU
      caches.push(st.kind === 'conv' ? { pre: mk('pre'), post: mk('post') }
                                     : { pre: mk('plain'), post: null });
    }
  }

  // draw one activation cube (core only; pad ring is a live overlay)
  // mode: 'plain' = 0..9 sequential · 'pre' = signed diverging · 'post' = ReLU'd
  function drawCube(ctx, iso, st, P, mode) {
    const N = st.Hs, dz = st.dz;
    const doStroke = iso.s >= 6;
    const cell = q => Iso.poly(ctx, q, ctx.fillStyle, doStroke ? 'rgba(0,0,0,0.28)' : null);
    const color = (r, c, d) => {
      if (mode === 'plain') return U.seqColor(texVal(st, r, c, d) / 9);
      const v = texSigned(st, r, c, d);
      if (mode === 'pre') return U.divColor(v, 9);
      return v <= 0 ? '#191a1c' : U.seqColor(v / 9);   // post: negatives → 0 (dark)
    };
    Iso.box(iso, P, P, 0, N, N, dz, '#1f232b');
    // top face
    for (let d = 0; d < dz; d++)
      for (let c = 0; c < N; c++) {
        ctx.fillStyle = U.shade(color(0, c, d), 1.25);
        cell([iso.p(P + c, P + N, d), iso.p(P + c + 1, P + N, d),
              iso.p(P + c + 1, P + N, d + 1), iso.p(P + c, P + N, d + 1)]);
      }
    // right face
    for (let d = 0; d < dz; d++)
      for (let r = 0; r < N; r++) {
        const yb = P + N - 1 - r;
        ctx.fillStyle = U.shade(color(r, N - 1, d), 0.62);
        cell([iso.p(P + N, yb, d), iso.p(P + N, yb, d + 1),
              iso.p(P + N, yb + 1, d + 1), iso.p(P + N, yb + 1, d)]);
      }
    // front face
    for (let r = 0; r < N; r++)
      for (let c = 0; c < N; c++) {
        ctx.fillStyle = color(r, c, 0);
        cell(Iso.cellQuad(iso, P, P, dz, c, r, N));
      }
  }

  // where is stage ci relative to its ReLU step? -> 'pre' | t (sweeping) | 'post'
  function rectPhase(si, t, ci) {
    const idx = steps.findIndex(s => s.type === 'relu' && s.ci === ci);
    if (idx < 0 || si < idx) return 'pre';
    if (si === idx) return U.easeInOut(t);
    return 'post';
  }

  // iso helper positioned over a stage's blit location (for live overlays)
  function screenIso(ctx, ci) {
    const sl = slots[ci];
    return Iso.make(ctx, sl.s, sl.x + sl.oxL, sl.y + sl.oyL);
  }

  // ---------------- render ----------------
  function layerColor(li) { return U.CAT[(li + 1) % U.CAT.length]; }

  function stageState(si, ci) { // 'done' | 'appearing' | 'hidden'
    if (ci === 0) return 'done';
    const prodIdx = steps.findIndex(s => s.ci === ci);
    if (prodIdx < si) return 'done';
    if (prodIdx === si) return 'appearing';
    return 'hidden';
  }

  function render(ctx, si, t) {
    const st = steps[si];

    for (let ci = 0; ci < chain.length; ci++) {
      const state = stageState(si, ci);
      const sl = slots[ci], sg = chain[ci];
      if (state === 'hidden') { drawGhost(ctx, ci); continue; }

      if (ci > 0) drawArrow(ctx, ci, state === 'appearing' ? U.easeOut(Math.min(t * 2.5, 1)) : 1, si);

      ctx.save();
      if (state === 'appearing') {
        const reveal = U.easeInOut(U.clamp((t - 0.15) / 0.75, 0, 1));
        ctx.globalAlpha = 0.25 + 0.75 * reveal;
        ctx.beginPath();
        ctx.rect(sl.x - 4, sl.y - 4, sl.cw + 8, (sl.ch + 8) * (0.12 + 0.88 * reveal));
        ctx.clip();
      }
      const phase = sg.kind === 'conv' ? rectPhase(si, t, ci) : 'pre';
      if (phase === 'pre') {
        ctx.drawImage(caches[ci].pre, sl.x, sl.y, sl.cw + 4, sl.ch + 4);
      } else if (phase === 'post') {
        ctx.drawImage(caches[ci].post, sl.x, sl.y, sl.cw + 4, sl.ch + 4);
      } else {  // ReLU sweeping top -> bottom: rectified above the line, raw below
        ctx.drawImage(caches[ci].pre, sl.x, sl.y, sl.cw + 4, sl.ch + 4);
        ctx.save();
        ctx.beginPath();
        ctx.rect(sl.x - 4, sl.y - 4, sl.cw + 8, (sl.ch + 8) * phase);
        ctx.clip();
        ctx.drawImage(caches[ci].post, sl.x, sl.y, sl.cw + 4, sl.ch + 4);
        ctx.restore();
        ctx.strokeStyle = U.withAlpha(layerColor(sg.li), 0.8);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(sl.x - 4, sl.y - 4 + (sl.ch + 8) * phase);
        ctx.lineTo(sl.x + sl.cw + 4, sl.y - 4 + (sl.ch + 8) * phase);
        ctx.stroke();
      }
      ctx.restore();

      // ReLU badge while (and after) the non-linearity fires
      if (sg.kind === 'conv' && phase !== 'pre') {
        const active = typeof phase === 'number';
        drawReluBadge(ctx, sl.x + sl.cw / 2, sl.y + sl.ch + 34, layerColor(sg.li), active);
      }

      // labels
      const name = sg.kind === 'input' ? 'input' : `${sg.kind}${sg.li + 1}`;
      txt(ctx, sl.x + sl.cw / 2, sl.y - 12, name,
          { align: 'center', weight: 600, size: 11.5,
            color: sg.kind === 'input' ? '#c3c2b7' : layerColor(sg.li) });
      txt(ctx, sl.x + sl.cw / 2, sl.y + sl.ch + 16, `(${sg.C},${sg.Hs},${sg.Hs})`,
          { align: 'center', mono: true, size: 10.5 });
    }

    // live overlays on the source cube of the running op
    if (st.type === 'conv' || st.type === 'pool') drawSweep(ctx, si, t);

    drawFlat(ctx, si, t);

    if (st.type === 'invalid') {
      const sl = slots[slots.length - 1];
      txt(ctx, sl.x + sl.cw + 60, BASE_Y, '⚠', { size: 26, color: '#e34948', align: 'center' });
      txt(ctx, sl.x + sl.cw + 60, BASE_Y + 24, `layer ${st.li + 1} impossible`, { size: 11, color: '#e34948', align: 'center' });
    }

    txt(ctx, 46, 60, 'CNN feature extractor', { size: 15, weight: 650, color: '#ffffff' });
    txt(ctx, 46, 80, `input (${cfg.C},${cfg.H},${cfg.H})  ·  ${cfg.L} × [conv → relu → pool]  ·  ${fmtN(totalParams)} params`, { size: 12 });
  }

  // small pill with the ReLU curve glyph, under a conv cube
  function drawReluBadge(ctx, cx, cy, col, active) {
    ctx.save();
    ctx.globalAlpha = active ? 1 : 0.75;
    ctx.fillStyle = '#20242b';
    ctx.strokeStyle = active ? col : U.withAlpha(col, 0.5);
    ctx.lineWidth = active ? 1.6 : 1;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(cx - 34, cy - 10, 68, 20, 10); else ctx.rect(cx - 34, cy - 10, 68, 20);
    ctx.fill(); ctx.stroke();
    // _/ curve
    ctx.strokeStyle = active ? '#ffffff' : '#c3c2b7';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(cx - 27, cy + 4); ctx.lineTo(cx - 18, cy + 4); ctx.lineTo(cx - 9, cy - 5);
    ctx.stroke();
    ctx.fillStyle = active ? '#ffffff' : '#c3c2b7';
    ctx.font = '600 9.5px ui-monospace, Menlo, monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText('ReLU', cx - 2, cy + 0.5);
    ctx.restore();
  }

  function drawGhost(ctx, ci) {
    const sl = slots[ci];
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(sl.x, sl.y, sl.cw, sl.ch);
    ctx.setLineDash([]);
    ctx.restore();
  }

  function drawArrow(ctx, ci, a, si) {
    const from = slots[ci - 1], to = slots[ci], sg = chain[ci];
    const x0 = from.x + from.cw + 4, x1 = to.x - 4;
    const cur = steps[si].ci === ci;
    const col = cur ? layerColor(sg.li) : 'rgba(137,135,129,0.55)';
    ctx.save();
    ctx.globalAlpha = a;
    ctx.strokeStyle = col; ctx.fillStyle = col;
    ctx.lineWidth = cur ? 2 : 1.3;
    ctx.beginPath();
    ctx.moveTo(x0, BASE_Y); ctx.lineTo(x1 - 6, BASE_Y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x1, BASE_Y); ctx.lineTo(x1 - 7, BASE_Y - 4); ctx.lineTo(x1 - 7, BASE_Y + 4);
    ctx.closePath(); ctx.fill();
    const L = cfg.layers[sg.li];
    const lines = sg.kind === 'conv'
      ? [`conv ${L.K}×${L.K} P${L.P}`, `F=${L.F}`]
      : ['max-pool', '2×2 s2'];
    txt(ctx, (x0 + x1) / 2, BASE_Y - 18, lines[0], { align: 'center', mono: true, size: 9.5, color: cur ? '#ffffff' : '#898781' });
    txt(ctx, (x0 + x1) / 2, BASE_Y - 7, lines[1], { align: 'center', mono: true, size: 9.5, color: cur ? '#ffffff' : '#898781' });
    ctx.restore();
  }

  function drawSweep(ctx, si, t) {
    const st = steps[si];
    const srcCi = st.ci - 1;
    const src = chain[srcCi], sl = slots[srcCi];
    const iso = screenIso(ctx, srcCi);
    const P = st.type === 'conv' ? cfg.layers[st.li].P : 0;
    const K = st.type === 'conv' ? cfg.layers[st.li].K : 2;
    const col = layerColor(st.li);
    const Wp2 = src.Hs + 2 * P;

    // pad ring (conv only), dashed empty cells around the core front face
    if (P > 0) {
      const a = Math.min(t / 0.15, 1) * 0.9;
      ctx.save();
      ctx.globalAlpha = a;
      for (let r = 0; r < Wp2; r++)
        for (let c = 0; c < Wp2; c++) {
          if (r >= P && r < P + src.Hs && c >= P && c < P + src.Hs) continue;
          const q = Iso.cellQuad(iso, 0, 0, src.dz, c, r, Wp2);
          ctx.setLineDash([2, 2]);
          Iso.poly(ctx, q, 'rgba(40,40,38,0.5)', 'rgba(160,158,150,0.5)');
          ctx.setLineDash([]);
        }
      Iso.wire(iso, 0, 0, 0, Wp2, Wp2, src.dz, 'rgba(195,194,183,0.4)', 1, a);
      ctx.restore();
    }

    // window glides corner-to-corner over the (padded) face
    const p = U.clamp(t / 0.8, 0, 1);
    const colF = p * (Wp2 - K), rowF = p * (Wp2 - K);
    const yb = Wp2 - rowF - K;
    Iso.wire(iso, colF, yb, 0, K, K, src.dz, col, 1.6, 0.8);
    const q = [iso.p(colF, yb, src.dz), iso.p(colF + K, yb, src.dz),
               iso.p(colF + K, yb + K, src.dz), iso.p(colF, yb + K, src.dz)];
    Iso.poly(ctx, q, U.withAlpha(col, 0.3), col, 2);
  }

  function drawFlat(ctx, si, t) {
    const st = steps[si];
    const on = st.type === 'flat' || st.type === 'done';
    if (invalidAt >= 0) return;
    const visible = on || steps[si].type !== 'intro' && si >= steps.length - 2;
    const a = st.type === 'flat' ? U.easeOut(t) : (st.type === 'done' ? 1 : 0.28);
    const x = slots.flatX + 8, y = BASE_Y;
    const last = slots[slots.length - 1];
    ctx.save();
    ctx.globalAlpha = a;
    ctx.strokeStyle = on ? '#3987e5' : 'rgba(137,135,129,0.5)';
    ctx.fillStyle = '#20242b';
    ctx.lineWidth = on ? 1.8 : 1.2;
    // arrow from last cube
    ctx.beginPath();
    ctx.moveTo(last.x + last.cw + 4, y); ctx.lineTo(x - 4, y);
    ctx.stroke();
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y - 40, 104, 80, 10); else ctx.rect(x, y - 40, 104, 80);
    ctx.fill(); ctx.stroke();
    txt(ctx, x + 52, y - 16, 'flatten', { align: 'center', weight: 600, size: 12, color: on ? '#ffffff' : '#898781' });
    txt(ctx, x + 52, y + 4, `${fmtN(flatN)}-d`, { align: 'center', mono: true, size: 11.5, color: on ? '#c3c2b7' : '#898781' });
    txt(ctx, x + 52, y + 24, 'vector', { align: 'center', size: 10.5 });
    txt(ctx, x + 52, y + 58, '→ MLP head', { align: 'center', size: 10.5, color: on ? '#c3c2b7' : '#63615c' });
    ctx.restore();
  }

  function txt(ctx, x, y, s, opts = {}) {
    ctx.save();
    ctx.fillStyle = opts.color || '#898781';
    ctx.font = `${opts.weight || 500} ${opts.size || 11.5}px ${opts.mono ? 'ui-monospace, Menlo, monospace' : 'system-ui, sans-serif'}`;
    ctx.textAlign = opts.align || 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(s, x, y);
    ctx.restore();
  }

  const fmtN = n => n.toLocaleString('en-US');

  // ---------------- captions ----------------
  function caption(si, t) {
    const st = steps[si];
    switch (st.type) {
      case 'intro':
        return `A CNN backbone: <b>${cfg.L} × [conv → ReLU → max-pool]</b>. Space shrinks, channels grow — pixels become features.`;
      case 'conv': {
        const sg = chain[st.ci], L = cfg.layers[st.li], Cin = chain[st.ci - 1].C;
        return `<b>conv${st.li + 1}</b>: ${L.F} filters ${L.K}×${L.K}×${Cin}, pad ${L.P} → <b>(${sg.C}, ${sg.Hs}, ${sg.Hs})</b> pre-activations (blue = negative, red = positive) · +${fmtN(sg.params)} params`;
      }
      case 'relu':
        return `<b>ReLU</b>: max(0, x) on every value — negatives go dark. Without this non-linearity, stacked convs would collapse into <b>one</b> big linear filter.`;
      case 'pool': {
        const sg = chain[st.ci], prev = chain[st.ci - 1];
        return `<b>pool${st.li + 1}</b>: max 2×2 stride 2 → <b>(${sg.C}, ${sg.Hs}, ${sg.Hs})</b> — no params, resolution halved`;
      }
      case 'flat':
        return `Flatten <b>(${chain[chain.length - 1].C}, ${chain[chain.length - 1].Hs}, ${chain[chain.length - 1].Hs})</b> → a <b>${fmtN(flatN)}-d</b> vector for the classifier (see the MLP tab).`;
      case 'invalid': {
        const L = cfg.layers[st.li];
        return `⚠ <b>layer ${st.li + 1}</b> is impossible: kernel ${L.K} &gt; input ${'' + (chainEndH() + 2 * L.P)} — add padding, shrink K, or remove a layer.`;
      }
      case 'done':
        return `<b>${fmtN(totalParams)}</b> learnable parameters. One output "pixel" now sees <b>${chain[chain.length - 1].rf}×${chain[chain.length - 1].rf}</b> pixels of the original image (its receptive field).`;
    }
    return '';
  }
  const chainEndH = () => chain[chain.length - 1].Hs;

  // ---------------- detail: architecture table ----------------
  function buildDetail(el, si) {
    const cur = steps[si].rowKey;
    let rows = '';
    const row = (key, name, shape, params, rf, extra = '') =>
      `<tr class="${key === cur ? 'cur' : ''}${extra}"><td>${name}</td><td>${shape}</td><td>${params}</td><td>${rf}</td></tr>`;
    for (const sg of chain) {
      const key = sg.kind === 'input' ? 'input' : sg.kind + sg.li;
      const name = sg.kind === 'input' ? 'input'
        : sg.kind === 'conv' ? `conv${sg.li + 1}  (${cfg.layers[sg.li].K}×${cfg.layers[sg.li].K}, P${cfg.layers[sg.li].P})`
        : `pool${sg.li + 1}  (2×2, s2)`;
      rows += row(key, name, `(${sg.C}, ${sg.Hs}, ${sg.Hs})`,
                  sg.params ? fmtN(sg.params) : '—', `${sg.rf}×${sg.rf}`);
      if (sg.kind === 'conv')
        rows += row('relu' + sg.li, `relu${sg.li + 1}  max(0,x)`,
                    `(${sg.C}, ${sg.Hs}, ${sg.Hs})`, '—', `${sg.rf}×${sg.rf}`);
    }
    if (invalidAt >= 0)
      rows += row('invalid', `⚠ conv${invalidAt + 1}`, 'kernel does not fit', '—', '—');
    else
      rows += row('flatten', 'flatten', `${fmtN(flatN)}`, '—', '');
    rows += row('total', '<b>total</b>', '', `<b>${fmtN(totalParams)}</b>`, '');
    el.innerHTML =
      `<div class="dp-title">architecture — shape · parameters · receptive field</div>` +
      `<table class="dp-table"><thead><tr><th>stage</th><th>output</th><th>params</th><th>rf</th></tr></thead>` +
      `<tbody>${rows}</tbody></table>`;
  }
  function updateDetail() {}

  // ---------------- controls ----------------
  function init(controlsEl, legendEl) {
    controlsEl.innerHTML = '';
    const g1 = group(controlsEl, 'Input image');
    stepList(g1, 'channels', () => cfg.C, v => cfg.C = v, [1, 2, 3, 4]);
    stepList(g1, 'size H=W', () => cfg.H, v => cfg.H = v, [8, 12, 16, 20, 24, 28, 32]);

    const g2 = group(controlsEl, 'Network');
    stepList(g2, 'layers', () => cfg.L, v => { cfg.L = v; regen(); init(controlsEl, legendEl); App.resetTimeline(); }, [1, 2, 3, 4], true);

    for (let li = 0; li < cfg.L; li++) {
      const gl = group(controlsEl, `Layer ${li + 1}`);
      stepList(gl, 'kernel K', () => cfg.layers[li].K, v => cfg.layers[li].K = v, KS);
      stepList(gl, 'padding P', () => cfg.layers[li].P, v => cfg.layers[li].P = v, [0, 1, 2]);
      stepList(gl, 'filters F', () => cfg.layers[li].F, v => cfg.layers[li].F = v, FS);
    }

    const btn = document.createElement('button');
    btn.className = 'btn';
    btn.textContent = '⚄ Regenerate activations';
    btn.onclick = () => { seed = (seed * 16807 + 13) % 2147483647; regen(); App.resetTimeline(); };
    controlsEl.appendChild(btn);

    legendEl.innerHTML = '';
  }

  function group(parent, title) {
    const g = document.createElement('div');
    g.className = 'ctl-group';
    g.innerHTML = `<h3>${title}</h3>`;
    parent.appendChild(g);
    return g;
  }

  // stepper over an explicit list of allowed values
  function stepList(g, label, get, set, list, selfRegen = false) {
    const row = document.createElement('div');
    row.className = 'ctl-row';
    row.innerHTML = `<label>${label}</label>` +
      `<span class="ctl-num"><button data-d="-1">−</button><span class="val">${get()}</span><button data-d="1">+</button></span>`;
    const val = row.querySelector('.val');
    row.querySelectorAll('button').forEach(b => b.onclick = () => {
      const i = list.indexOf(get());
      const v = list[U.clamp(i + Number(b.dataset.d), 0, list.length - 1)];
      if (v === get()) return;
      set(v);
      if (!selfRegen) { regen(); val.textContent = get(); App.resetTimeline(); }
    });
    g.appendChild(row);
  }

  regen();

  return {
    id: 'cnn',
    title: 'CNN',
    desc: 'A full convolutional backbone: each layer runs conv2d, ReLU, then 2×2 max-pooling. Watch the tensor chain — spatial size shrinking, channels growing — with parameter counts and receptive fields.',
    VW, VH,
    get steps() { return steps; },
    init, regen, render, caption, buildDetail, updateDetail
  };
})();
