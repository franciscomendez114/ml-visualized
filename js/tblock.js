'use strict';
/* ============ Transformer block module: LN → MHA → ⊕ → LN → FFN → ⊕ ============ */
const TBlockModule = (() => {

  const VW = 1280, VH = 640;
  const cfg = { T: 5, dm: 8, H: 2 };
  let seed = 113;

  const SENTENCE = ['the', 'robot', 'ate', 'an', 'apple', 'today'];

  let words = [], dk = 4, dff = 32, fi = 2;   // fi = focus token for the math panel
  let X = [], LN1 = [], MhaOut = [], Add1 = [], LN2 = [], FfnOut = [], Add2 = [];
  let Pheads = [], mu1 = [], sd1 = [], mu2 = [], sd2 = [];
  let nodes = [], steps = [];

  // ---------------- math ----------------
  function regen() {
    const rnd = U.mulberry32(seed);
    const r1 = () => Math.round((rnd() * 2 - 1) * 10) / 10;
    dk = cfg.dm / cfg.H;
    dff = 4 * cfg.dm;
    words = SENTENCE.slice(0, cfg.T);
    fi = Math.min(2, cfg.T - 1);

    const Ev = {};
    for (const w of new Set(words)) Ev[w] = Array.from({ length: cfg.dm }, r1);
    X = words.map(w => Ev[w]);

    [LN1, mu1, sd1] = layerNorm(X);

    // multi-head attention on LN1 (same construction as the Multi-Head tab)
    const proj = (x, W) => W[0].map((_, j) => W.reduce((s, row, i) => s + row[j] * x[i], 0));
    const mk = (r, c) => Array.from({ length: r }, () => Array.from({ length: c }, r1));
    Pheads = [];
    const heads = [];
    for (let h = 0; h < cfg.H; h++) {
      const Wq = mk(cfg.dm, dk), Wk = mk(cfg.dm, dk), Wv = mk(cfg.dm, dk);
      const q = LN1.map(x => proj(x, Wq)), k = LN1.map(x => proj(x, Wk)), v = LN1.map(x => proj(x, Wv));
      const Ph = LN1.map((_, i) => softmax(k.map((kv, j) => j > i ? -Infinity
        : q[i].reduce((s, x, d) => s + x * kv[d], 0) / Math.sqrt(dk))));
      Pheads.push(Ph);
      heads.push(LN1.map((_, i) => {
        const o = Array.from({ length: dk }, () => 0);
        for (let j = 0; j < cfg.T; j++) for (let d = 0; d < dk; d++) o[d] += Ph[i][j] * v[j][d];
        return o;
      }));
    }
    const WO = mk(cfg.dm, cfg.dm);
    MhaOut = X.map((_, i) => proj([].concat(...heads.map(hh => hh[i])), WO));

    Add1 = X.map((x, i) => x.map((v, d) => v + MhaOut[i][d]));
    [LN2, mu2, sd2] = layerNorm(Add1);

    // position-wise FFN: d -> 4d -> d, ReLU
    const W1 = mk(cfg.dm, dff), b1 = Array.from({ length: dff }, r1);
    const W2 = mk(dff, cfg.dm), b2 = Array.from({ length: cfg.dm }, r1);
    FfnOut = LN2.map(x => {
      const hid = proj(x, W1).map((z, j) => Math.max(0, z + b1[j]));
      return proj(hid, W2).map((z, j) => z + b2[j]);
    });
    Add2 = Add1.map((x, i) => x.map((v, d) => v + FfnOut[i][d]));

    buildNodes();
    buildSteps();
  }

  function layerNorm(M) {
    const out = [], mus = [], sds = [];
    for (const row of M) {
      const mu = row.reduce((a, b) => a + b, 0) / row.length;
      const sd = Math.sqrt(row.reduce((a, b) => a + (b - mu) * (b - mu), 0) / row.length) || 1;
      mus.push(mu); sds.push(sd);
      out.push(row.map(v => (v - mu) / sd));      // γ=1, β=0
    }
    return [out, mus, sds];
  }
  function softmax(xs) {
    const m = Math.max(...xs.filter(x => x > -Infinity));
    const es = xs.map(x => x === -Infinity ? 0 : Math.exp(x - m));
    const Z = es.reduce((a, b) => a + b, 0);
    return es.map(e => e / Z);
  }

  function buildSteps() {
    steps = [
      { type: 'intro', dur: 2.4 }, { type: 'ln1', dur: 2.4 }, { type: 'attn', dur: 2.8 },
      { type: 'add1', dur: 2.4 }, { type: 'ln2', dur: 2.0 }, { type: 'ffn', dur: 2.6 },
      { type: 'add2', dur: 2.4 }, { type: 'done', dur: 2.2 }
    ];
  }

  // ---------------- layout: the station chain ----------------
  const CY = 300;
  function buildNodes() {
    const mw = Math.max(56, cfg.dm * 10), mh = cfg.T * 13;
    const defs = [
      { id: 'x',    kind: 'mat', data: () => X,      label: 'input x', step: 0 },
      { id: 'ln1',  kind: 'op',  w: 54,  label: 'Layer\nNorm', step: 1 },
      { id: 'm1',   kind: 'mat', data: () => LN1,    label: 'x̂', step: 1 },
      { id: 'mha',  kind: 'op',  w: 118, label: 'multi-head\nattention', step: 2 },
      { id: 'm2',   kind: 'mat', data: () => MhaOut, label: 'attn(x̂)', step: 2 },
      { id: 'add1', kind: 'plus', step: 3 },
      { id: 'm3',   kind: 'mat', data: () => Add1,   label: 'x + attn', step: 3 },
      { id: 'ln2',  kind: 'op',  w: 54,  label: 'Layer\nNorm', step: 4 },
      { id: 'm4',   kind: 'mat', data: () => LN2,    label: 'x̂', step: 4 },
      { id: 'ffn',  kind: 'op',  w: 96,  label: 'MLP\nd→4d→d', step: 5 },
      { id: 'm5',   kind: 'mat', data: () => FfnOut, label: 'mlp(x̂)', step: 5 },
      { id: 'add2', kind: 'plus', step: 6 },
      { id: 'm6',   kind: 'mat', data: () => Add2,   label: 'output', step: 6 },
    ];
    const gapA = 13;      // arrow gap between elements
    let total = 0;
    for (const d of defs) total += (d.kind === 'mat' ? mw : d.kind === 'plus' ? 28 : d.w) + gapA;
    total -= gapA;
    let x = Math.max(40, (VW - total) / 2) + 26;   // leave room for row labels
    nodes = defs.map(d => {
      const w = d.kind === 'mat' ? mw : d.kind === 'plus' ? 28 : d.w;
      const n = { ...d, x, w, mw, mh };
      x += w + gapA;
      return n;
    });
  }
  const nodeById = id => nodes.find(n => n.id === id);

  // ---------------- render ----------------
  function nodeState(si, n) {
    const prod = n.step + (steps[0].type === 'intro' ? 0 : 0);
    if (prod < si) return 'done';
    if (prod === si) return 'appearing';
    return 'hidden';
  }

  function render(ctx, si, t) {
    const st = steps[si];

    // residual skip curves (drawn first, under everything)
    drawSkip(ctx, si, t, 'x', 'add1', si === 3);
    drawSkip(ctx, si, t, 'm3', 'add2', si === 6);

    for (const n of nodes) {
      const state = nodeState(si, n);
      if (state === 'hidden') { drawGhost(ctx, n); continue; }
      const a = state === 'appearing' ? 0.25 + 0.75 * U.easeInOut(U.clamp(t / 0.6, 0, 1)) : 1;
      drawArrowInto(ctx, n, a);
      if (n.kind === 'mat') drawMat(ctx, n, a, si, state);
      else if (n.kind === 'op') drawOp(ctx, n, a, state === 'appearing');
      else drawPlus(ctx, n, a, state === 'appearing');
    }

    txt(ctx, 46, 48, 'one transformer block (pre-LN)', { size: 15, weight: 650, color: '#ffffff' });
    txt(ctx, 46, 68, `"${words.join(' ')}" · (T,d) = (${cfg.T},${cfg.dm}) · ${cfg.H} heads · FFN ${cfg.dm}→${dff}→${cfg.dm}`, { size: 12 });
    if (st.type === 'done')
      txt(ctx, 46, 92, `same shape in and out → stack N of these (GPT-3: 96)`, { size: 12, color: '#c3c2b7' });

    // focus-token marker
    txt(ctx, nodeById('x').x - 6, CY - nodes[0].mh / 2 + fi * 13 + 6.5, words[fi],
        { align: 'right', mono: true, size: 9.5, color: '#eda100' });
  }

  function drawGhost(ctx, n) {
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.setLineDash([4, 4]);
    const h = n.kind === 'mat' ? n.mh : n.kind === 'plus' ? 28 : 64;
    ctx.strokeRect(n.x, CY - h / 2, n.w, h);
    ctx.setLineDash([]);
    ctx.restore();
  }

  function drawArrowInto(ctx, n, a) {
    const i = nodes.indexOf(n);
    if (i === 0) return;
    const prev = nodes[i - 1];
    const x0 = prev.x + prev.w + 2, x1 = n.x - 3;
    ctx.save();
    ctx.globalAlpha = a;
    ctx.strokeStyle = ctx.fillStyle = 'rgba(137,135,129,0.7)';
    ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(x0, CY); ctx.lineTo(x1 - 4, CY); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x1, CY); ctx.lineTo(x1 - 5.5, CY - 3.5); ctx.lineTo(x1 - 5.5, CY + 3.5);
    ctx.closePath(); ctx.fill();
    ctx.restore();
  }

  function drawMat(ctx, n, a, si, state) {
    const M = n.data();
    const absM = Math.max(0.5, ...M.flat().map(Math.abs));
    const cw = n.w / cfg.dm, chh = 13;
    const y0 = CY - n.mh / 2;
    ctx.save();
    ctx.globalAlpha = a;
    for (let i = 0; i < cfg.T; i++)
      for (let d = 0; d < cfg.dm; d++) {
        ctx.fillStyle = U.divColor(M[i][d], absM);
        ctx.fillRect(n.x + d * cw, y0 + i * chh, cw - 1, chh - 1);
      }
    ctx.strokeStyle = 'rgba(255,255,255,0.14)';
    ctx.strokeRect(n.x - 0.5, y0 - 0.5, n.w + 0.5, cfg.T * chh);
    // focus-token row ring
    ctx.strokeStyle = U.withAlpha('#eda100', state === 'appearing' ? 0.95 : 0.4);
    ctx.lineWidth = 1.4;
    ctx.strokeRect(n.x - 1.5, y0 + fi * chh - 1.5, n.w + 2.5, chh + 2);
    ctx.restore();
    txt(ctx, n.x + n.w / 2, y0 + cfg.T * chh + 13, n.label,
        { align: 'center', mono: true, size: 10.5, color: n.id === 'm6' ? '#ffffff' : '#898781' });
  }

  function drawOp(ctx, n, a, active) {
    const h = 64, y0 = CY - h / 2;
    ctx.save();
    ctx.globalAlpha = a;
    ctx.fillStyle = '#22262e';
    ctx.strokeStyle = active ? '#3987e5' : 'rgba(255,255,255,0.18)';
    ctx.lineWidth = active ? 1.8 : 1.1;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(n.x, y0, n.w, h, 9); else ctx.rect(n.x, y0, n.w, h);
    ctx.fill(); ctx.stroke();
    const lines = n.label.split('\n');
    lines.forEach((L, i) =>
      txt(ctx, n.x + n.w / 2, CY - (lines.length - 1) * 6 + i * 12 - (n.id === 'mha' ? 16 : 0),
          L, { align: 'center', size: 10, weight: 600, color: active ? '#ffffff' : '#c3c2b7' }));

    if (n.id === 'mha') {           // real per-head attention patterns inside
      const gs = Math.min(5.5, (n.w - 16 - (cfg.H - 1) * 8) / (cfg.H * cfg.T));
      const gw = cfg.T * gs;
      let gx = n.x + (n.w - cfg.H * gw - (cfg.H - 1) * 8) / 2;
      for (let h2 = 0; h2 < cfg.H; h2++) {
        for (let i = 0; i < cfg.T; i++)
          for (let j = 0; j < cfg.T; j++) {
            ctx.fillStyle = j > i ? '#191918' : U.seqColor(Pheads[h2][i][j]);
            ctx.fillRect(gx + j * gs, CY - 2 + i * gs, gs - 0.7, gs - 0.7);
          }
        ctx.strokeStyle = U.withAlpha(U.CAT[h2 % U.CAT.length], 0.8);
        ctx.lineWidth = 1;
        ctx.strokeRect(gx - 1, CY - 3, gw + 1, gw + 1);
        gx += gw + 8;
      }
    }
    if (n.id === 'ffn') {           // tiny MLP glyph
      const cx = n.x + n.w / 2;
      const cols = [[-26, 3], [0, 5], [26, 3]];
      ctx.strokeStyle = 'rgba(137,135,129,0.4)';
      ctx.lineWidth = 0.7;
      for (let c = 0; c < 2; c++)
        for (let i = 0; i < cols[c][1]; i++)
          for (let j = 0; j < cols[c + 1][1]; j++) {
            ctx.beginPath();
            ctx.moveTo(cx + cols[c][0], CY + 2 + (i - (cols[c][1] - 1) / 2) * 9);
            ctx.lineTo(cx + cols[c + 1][0], CY + 2 + (j - (cols[c + 1][1] - 1) / 2) * 9);
            ctx.stroke();
          }
      ctx.fillStyle = '#6da7ec';
      for (const [dx, k] of cols)
        for (let i = 0; i < k; i++) {
          ctx.beginPath();
          ctx.arc(cx + dx, CY + 2 + (i - (k - 1) / 2) * 9, 2.6, 0, 7);
          ctx.fill();
        }
    }
    ctx.restore();
  }

  function drawPlus(ctx, n, a, active) {
    ctx.save();
    ctx.globalAlpha = a;
    ctx.fillStyle = '#22262e';
    ctx.strokeStyle = active ? '#eda100' : 'rgba(255,255,255,0.25)';
    ctx.lineWidth = active ? 2 : 1.2;
    ctx.beginPath(); ctx.arc(n.x + 14, CY, 14, 0, 7); ctx.fill(); ctx.stroke();
    txt(ctx, n.x + 14, CY + 0.5, '+', { align: 'center', size: 16, weight: 600, color: active ? '#eda100' : '#c3c2b7' });
    ctx.restore();
  }

  function drawSkip(ctx, si, t, fromId, toId, active) {
    const from = nodeById(fromId), to = nodeById(toId);
    if (nodeState(si, to) === 'hidden' && !active) {
      if (nodeState(si, from) === 'hidden') return;
    }
    const vis = nodeState(si, to) !== 'hidden';
    const a = active ? 0.9 : vis ? 0.35 : 0.15;
    const x0 = from.x + from.w / 2, x1 = to.x + 14;
    const yTop = CY - nodes[0].mh / 2 - 52;
    ctx.save();
    ctx.strokeStyle = U.withAlpha('#eda100', a);
    ctx.lineWidth = active ? 2.2 : 1.4;
    ctx.beginPath();
    ctx.moveTo(x0, CY - from.mh / 2 - 4);
    ctx.bezierCurveTo(x0, yTop, x1, yTop, x1, CY - 16);
    ctx.stroke();
    if (active) {          // pulse traveling along the residual path
      const tp = (t * 1.4) % 1;
      const bez = (p0, p1, p2, p3, u) => {
        const v = 1 - u;
        return v * v * v * p0 + 3 * v * v * u * p1 + 3 * v * u * u * p2 + u * u * u * p3;
      };
      const px = bez(x0, x0, x1, x1, tp);
      const py = bez(CY - from.mh / 2 - 4, yTop, yTop, CY - 16, tp);
      ctx.fillStyle = '#eda100';
      ctx.beginPath(); ctx.arc(px, py, 3.5, 0, 7); ctx.fill();
    }
    txt(ctx, (x0 + x1) / 2, yTop - 9, 'residual', { align: 'center', size: 9.5, color: U.withAlpha('#eda100', Math.max(a, 0.4)) });
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
  function totalParams() {
    return 4 * cfg.dm * cfg.dm                 // Wq,Wk,Wv (all heads) + WO
      + cfg.dm * dff + dff + dff * cfg.dm + cfg.dm   // FFN
      + 2 * 2 * cfg.dm;                        // two LayerNorms (γ, β)
  }
  function caption(si, t) {
    switch (steps[si].type) {
      case 'intro':
        return `One <b>transformer block</b>: attention + MLP, each wrapped in LayerNorm and a <b>residual connection</b>. The (${cfg.T}×${cfg.dm}) token matrix flows left to right; row "<b>${words[fi]}</b>" is tracked below.`;
      case 'ln1':
        return `<b>LayerNorm</b> (pre-LN): each token's ${cfg.dm} values are shifted to mean 0 and scaled to variance 1 — every row on the same footing before attention.`;
      case 'attn':
        return `<b>Multi-head attention</b> — the only place tokens exchange information. ${cfg.H} heads with their own patterns (see the Multi-Head tab for the internals).`;
      case 'add1':
        return `<b>Residual add</b>: the block's input is added straight back. The stream is never overwritten — attention only <b>nudges</b> it, and gradients flow through the +.`;
      case 'ln2':
        return `Normalize again before the MLP — same recipe, second LayerNorm with its own γ, β.`;
      case 'ffn':
        return `<b>Position-wise MLP</b>: the same ${cfg.dm}→${dff}→${cfg.dm} ReLU net applied to every token <b>independently</b> — it mixes features, never tokens.`;
      case 'add2':
        return `Second residual closes the block: output = x + attn + mlp — the stream accumulates edits.`;
      case 'done':
        return `Output shape = input shape → blocks stack. This one holds <b>${totalParams().toLocaleString('en-US')}</b> params: attention 4d² = ${4 * cfg.dm * cfg.dm}, MLP ≈ 8d² = ${2 * cfg.dm * dff}.`;
    }
    return '';
  }

  // ---------------- detail: follow one token ----------------
  const f2 = v => (Math.round(v * 100) / 100).toFixed(2);
  function vecChips(v, absM) {
    return `<span style="display:inline-flex;gap:2px;vertical-align:middle">` +
      v.map(x => `<span style="min-width:34px;height:19px;padding:0 2px;border-radius:4px;background:${U.divColor(x, absM)};color:${U.inkFor(U.divColor(x, absM))};font:600 9.5px var(--mono);display:flex;align-items:center;justify-content:center">${f2(x)}</span>`).join('') +
      `</span>`;
  }
  function buildDetail(el, si) {
    const st = steps[si];
    const w = `"${words[fi]}"`;
    const A = M => Math.max(0.5, ...M.flat().map(Math.abs));
    let html = `<div class="dp-title">following token <b>${w}</b> (row ${fi})</div><div class="dp-eq" style="line-height:2.6">`;
    switch (st.type) {
      case 'intro':
        html += `x = ${vecChips(X[fi], A(X))} — its ${cfg.dm}-d slice of the residual stream`;
        break;
      case 'ln1':
        html += `x = ${vecChips(X[fi], A(X))}  μ = <b>${f2(mu1[fi])}</b>, σ = <b>${f2(sd1[fi])}</b><br>` +
                `x̂ = (x−μ)/σ = ${vecChips(LN1[fi], A(LN1))}`;
        break;
      case 'attn':
        html += Pheads.map((Ph, h) =>
          `head ${h} weights for ${w}: ` + vecChips(Ph[fi].slice(0, fi + 1), 1)).join('<br>') +
          `<br>attn out = ${vecChips(MhaOut[fi], A(MhaOut))}`;
        break;
      case 'add1':
        html += `x ${vecChips(X[fi], A(X))}<br>+ attn ${vecChips(MhaOut[fi], A(MhaOut))}<br>= ${vecChips(Add1[fi], A(Add1))}`;
        break;
      case 'ln2':
        html += `μ = <b>${f2(mu2[fi])}</b>, σ = <b>${f2(sd2[fi])}</b> → x̂ = ${vecChips(LN2[fi], A(LN2))}`;
        break;
      case 'ffn':
        html += `x̂ (${cfg.dm}) → W₁ (${cfg.dm}×${dff}) → ReLU → W₂ (${dff}×${cfg.dm})<br>` +
                `mlp out = ${vecChips(FfnOut[fi], A(FfnOut))}`;
        break;
      case 'add2':
        html += `x+attn ${vecChips(Add1[fi], A(Add1))}<br>+ mlp ${vecChips(FfnOut[fi], A(FfnOut))}<br>= out ${vecChips(Add2[fi], A(Add2))}`;
        break;
      case 'done':
        html += `${w}: x ${vecChips(X[fi], A(X))} → out ${vecChips(Add2[fi], A(Add2))}<br>` +
                `<span style="color:var(--ink-muted);font:500 12px var(--sans)">still ${cfg.dm} numbers — enriched by context, ready for the next block (or the LM head).</span>`;
        break;
    }
    el.innerHTML = html + '</div>';
  }
  function updateDetail() {}

  // ---------------- controls ----------------
  function init(controlsEl, legendEl) {
    controlsEl.innerHTML = '';
    const g = document.createElement('div');
    g.className = 'ctl-group';
    g.innerHTML = '<h3>Transformer block</h3>';
    controlsEl.appendChild(g);
    stepList(g, 'tokens T', () => cfg.T, v => cfg.T = v, [3, 4, 5, 6]);
    stepList(g, 'model dim d', () => cfg.dm, v => cfg.dm = v, [4, 8]);
    stepList(g, 'heads H', () => cfg.H, v => cfg.H = v, [1, 2, 4]);

    const btn = document.createElement('button');
    btn.className = 'btn';
    btn.textContent = '⚄ Regenerate values';
    btn.onclick = () => { seed = (seed * 16807 + 19) % 2147483647; regen(); App.resetTimeline(); };
    controlsEl.appendChild(btn);

    const note = document.createElement('div');
    note.className = 'shape-note';
    note.innerHTML = `pre-LN (GPT-style)<br>FFN hidden = 4d = <b>${dff}</b><br>params = <b>${totalParams().toLocaleString('en-US')}</b>`;
    controlsEl.appendChild(note);

    legendEl.innerHTML =
      `<div class="legend-row"><span class="legend-swatch" style="background:#eda100"></span><span>residual path / focus row</span></div>` +
      `<div class="legend-row" style="margin-top:6px"><span>matrix value (per-matrix scale)</span></div>` +
      (() => { const s2 = []; for (let i = 0; i <= 10; i++) s2.push(U.divColor(U.lerp(-1, 1, i / 10), 1));
        return `<div class="legend-ramp" style="background:linear-gradient(90deg,${s2.join(',')})"></div>` +
               `<div class="legend-cap"><span>−</span><span>+</span></div>`; })();
  }

  function stepList(g, label, get, set, list) {
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
      if (cfg.H > cfg.dm) cfg.H = cfg.dm;
      regen(); val.textContent = get();
      init(document.getElementById('controls'), document.getElementById('legend'));
      App.resetTimeline();
    });
    g.appendChild(row);
  }

  regen();

  return {
    id: 'tblock',
    title: 'Transformer Block',
    desc: 'The full block, pre-LN style: LayerNorm → multi-head attention → residual add → LayerNorm → position-wise MLP → residual add. Attention mixes tokens; the MLP mixes features; residuals keep the stream intact.',
    VW, VH,
    get steps() { return steps; },
    init, regen, render, caption, buildDetail, updateDetail
  };
})();
