'use strict';
/* ============ Attention / language-model module ============ */
const AttnModule = (() => {

  const VW = 1280, VH = 620;
  const cfg = { T: 5, d: 4, causal: true };
  let seed = 55;

  const SENTENCE = ['the', 'robot', 'ate', 'an', 'apple', 'today'];
  const EXTRA_VOCAB = ['ran', 'slept'];

  let words = [], vocab = [], Evocab = {}, E = [], Q = [], K = [], V = [];
  let S = [], Sc = [], Pr = [], O = [], logits = [], probs = [];
  let steps = [];

  const PH = { dots: 0.50, scale: 0.62, smax: 0.80, wsum: 1.0 };

  // ---------------- data ----------------
  function regen() {
    const rnd = U.mulberry32(seed);
    const r1 = () => Math.round((rnd() * 2 - 1) * 10) / 10;   // 1-decimal in [-1,1]

    words = SENTENCE.slice(0, cfg.T);
    vocab = [...new Set([...SENTENCE, ...EXTRA_VOCAB])];

    Evocab = {};
    for (const w of vocab) Evocab[w] = Array.from({ length: cfg.d }, r1);
    E = words.map(w => Evocab[w]);

    const Wq = mat(cfg.d, r1), Wk = mat(cfg.d, r1), Wv = mat(cfg.d, r1);
    const proj = (x, W) => W.map(row => // rounded to 1dp so on-screen math is exact
      Math.round(row.reduce((s, w, i) => s + w * x[i], 0) * 10) / 10);
    Q = E.map(x => proj(x, Wq));
    K = E.map(x => proj(x, Wk));
    V = E.map(x => proj(x, Wv));

    S = []; Sc = []; Pr = []; O = [];
    for (let i = 0; i < cfg.T; i++) {
      const srow = [], scrow = [];
      for (let j = 0; j < cfg.T; j++) {
        const s = dot(Q[i], K[j]);
        srow.push(s);
        scrow.push(s / Math.sqrt(cfg.d));
      }
      S.push(srow); Sc.push(scrow);
      Pr.push(softmax(scrow.map((v, j) => masked(i, j) ? -Infinity : v)));
      const o = Array.from({ length: cfg.d }, () => 0);
      for (let j = 0; j < cfg.T; j++)
        for (let k = 0; k < cfg.d; k++) o[k] += Pr[i][j] * V[j][k];
      O.push(o);
    }
    logits = vocab.map(w => dot2(O[cfg.T - 1], Evocab[w]));
    probs = softmax(logits);
    buildSteps();
  }

  const mat = (d, r1) => Array.from({ length: d }, () => Array.from({ length: d }, r1));
  const dot = (a, b) => Math.round(a.reduce((s, v, i) => s + v * b[i], 0) * 100) / 100;
  const dot2 = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const masked = (i, j) => cfg.causal && j > i;
  function softmax(xs) {
    const m = Math.max(...xs.filter(x => x > -Infinity));
    const es = xs.map(x => x === -Infinity ? 0 : Math.exp(x - m));
    const Z = es.reduce((a, b) => a + b, 0);
    return es.map(e => e / Z);
  }
  const nAvail = i => cfg.causal ? i + 1 : cfg.T;

  function buildSteps() {
    steps = [{ type: 'intro', dur: 2.6 }, { type: 'proj', dur: 2.4 }];
    for (let i = 0; i < cfg.T; i++) steps.push({ type: 'row', i, dur: 3.6 });
    steps.push({ type: 'lm', dur: 3.2 }, { type: 'done', dur: 2.0 });
  }

  // ---------------- layout ----------------
  const CELL = 50;
  function G() {
    const gx = 560, gy = 190;
    return { gx, gy,
      keyY: 108, kStripY: 152,
      qChipX: 320, qStripX: 428,
      vY: gy + cfg.T * CELL + 34,
      outX: gx + cfg.T * CELL + 80 };
  }

  // ---------------- drawing helpers ----------------
  function stripW(cw) { return cfg.d * cw; }
  function strip(ctx, x, y, vals, cw = 12, ch = 15, alpha = 1, ring = null) {
    ctx.save();
    ctx.globalAlpha = alpha;
    for (let k = 0; k < vals.length; k++) {
      ctx.fillStyle = U.divColor(vals[k], 1.6);
      ctx.fillRect(x + k * cw, y, cw - 1, ch);
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x - 0.5, y - 0.5, vals.length * cw, ch + 1);
    if (ring) {
      ctx.strokeStyle = ring; ctx.lineWidth = 2;
      ctx.strokeRect(x - 2, y - 2, vals.length * cw + 4, ch + 4);
    }
    ctx.restore();
  }

  function tokenChip(ctx, x, y, word, active, color) {
    ctx.save();
    ctx.font = '600 12.5px ui-monospace, Menlo, monospace';
    const w = ctx.measureText(word).width + 18;
    ctx.fillStyle = active ? U.withAlpha(color || '#3987e5', 0.28) : '#242423';
    ctx.strokeStyle = active ? (color || '#3987e5') : 'rgba(255,255,255,0.14)';
    ctx.lineWidth = active ? 1.8 : 1;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x - w / 2, y - 12, w, 24, 7); else ctx.rect(x - w / 2, y - 12, w, 24);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = active ? '#ffffff' : '#c3c2b7';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(word, x, y + 0.5);
    ctx.restore();
    return w;
  }

  function txt(ctx, x, y, s, opts = {}) {
    ctx.save();
    ctx.fillStyle = opts.color || '#898781';
    ctx.font = `${opts.weight || 500} ${opts.size || 11.5}px ${opts.mono ? 'ui-monospace, Menlo, monospace' : 'system-ui, sans-serif'}`;
    ctx.textAlign = opts.align || 'left';
    ctx.textBaseline = opts.base || 'alphabetic';
    ctx.fillText(s, x, y);
    ctx.restore();
  }

  // ---------------- state helpers ----------------
  function rowState(si, i) { // 'done' | 'active' | 'pending'
    const st = steps[si];
    if (st.type === 'row') return i < st.i ? 'done' : i === st.i ? 'active' : 'pending';
    if (st.type === 'intro' || st.type === 'proj') return 'pending';
    return 'done';
  }
  function dotJ(st, t) { // current key index during dots phase
    const n = nAvail(st.i);
    if (t >= PH.dots) return n;
    return Math.min(Math.floor(t / PH.dots * n), n - 1);
  }

  // ---------------- render ----------------
  function render(ctx, si, t) {
    const st = steps[si];
    const g = G();
    const projA = st.type === 'intro' ? 0 : st.type === 'proj' ? U.easeInOut(t) : 1;

    // ---- keys along the top ----
    txt(ctx, g.gx - 14, g.keyY + 4, 'keys →', { align: 'right', weight: 600 });
    txt(ctx, g.gx - 14, g.kStripY + 12, projA > 0.5 ? 'k =' : 'emb', { align: 'right', mono: true });
    for (let j = 0; j < cfg.T; j++) {
      const cx = g.gx + j * CELL + CELL / 2;
      let active = false, color = '#199e70';
      if (st.type === 'row') { const cj = dotJ(st, t); active = t < PH.dots && j === cj && !masked(st.i, j); }
      tokenChip(ctx, cx, g.keyY, words[j], active, color);
      const sx = cx - stripW(10) / 2;
      if (projA < 1) strip(ctx, sx, g.kStripY, E[j], 10, 13, 1 - projA);
      if (projA > 0) strip(ctx, sx, g.kStripY, K[j], 10, 13, projA, active ? '#199e70' : null);
    }

    // ---- queries down the left ----
    txt(ctx, g.qChipX, g.gy - 26, '↓ queries', { weight: 600 });
    txt(ctx, g.qStripX + stripW(10) / 2, g.gy - 26, projA > 0.5 ? 'q' : 'emb', { align: 'center', mono: true });
    for (let i = 0; i < cfg.T; i++) {
      const cy = g.gy + i * CELL + CELL / 2;
      const active = st.type === 'row' && st.i === i;
      tokenChip(ctx, g.qChipX, cy, words[i], active, '#3987e5');
      const sx = g.qStripX;
      if (projA < 1) strip(ctx, sx, cy - 7, E[i], 10, 13, 1 - projA);
      if (projA > 0) strip(ctx, sx, cy - 7, Q[i], 10, 13, projA, active && t < PH.dots ? '#3987e5' : null);
    }

    // ---- attention grid ----
    for (let i = 0; i < cfg.T; i++) {
      for (let j = 0; j < cfg.T; j++) {
        const x = g.gx + j * CELL, y = g.gy + i * CELL;
        const state = rowState(si, i);
        const mk = masked(i, j);
        let fill = '#20201f', text = null, tcol = '#c3c2b7';

        if (mk) {
          fill = '#191918';
        } else if (state === 'done') {
          fill = U.seqColor(Pr[i][j]); text = Pr[i][j].toFixed(2); tcol = U.inkFor(fill);
        } else if (state === 'active' && st.type === 'row') {
          const cj = dotJ(st, t);
          if (t < PH.dots) {
            if (j < cj) { fill = '#2c2c2a'; text = S[i][j].toFixed(2); }
            else if (j === cj) { fill = '#33332f'; text = S[i][j].toFixed(2); }
          } else if (t < PH.scale) {
            fill = '#2c2c2a'; text = S[i][j].toFixed(2);
          } else if (t < PH.smax) {
            fill = '#2c2c2a'; text = Sc[i][j].toFixed(2); tcol = '#eda100';
          } else {
            fill = U.seqColor(Pr[i][j]); text = Pr[i][j].toFixed(2); tcol = U.inkFor(fill);
          }
        }

        ctx.fillStyle = fill;
        ctx.fillRect(x, y, CELL - 2, CELL - 2);
        if (mk) { // hatch masked cells
          ctx.save();
          ctx.strokeStyle = 'rgba(255,255,255,0.06)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          for (let h = -CELL; h < CELL; h += 7) {
            ctx.moveTo(x + Math.max(0, h), y + Math.max(0, -h));
            ctx.lineTo(x + Math.min(CELL - 2, h + CELL), y + Math.min(CELL - 2, CELL - h));
          }
          ctx.stroke();
          ctx.restore();
        }
        if (st.type === 'row' && st.i === i && j === dotJ(st, t) && t < PH.dots && !mk) {
          ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2;
          ctx.strokeRect(x + 1, y + 1, CELL - 4, CELL - 4);
        }
        if (text) {
          ctx.fillStyle = tcol;
          ctx.font = '600 10.5px ui-monospace, Menlo, monospace';
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(text, x + (CELL - 2) / 2, y + (CELL - 2) / 2);
        }
      }
    }
    // grid frame + row highlight
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;
    ctx.strokeRect(g.gx - 1, g.gy - 1, cfg.T * CELL, cfg.T * CELL);
    if (st.type === 'row') {
      ctx.strokeStyle = '#3987e5'; ctx.lineWidth = 2;
      ctx.strokeRect(g.gx - 2, g.gy + st.i * CELL - 2, cfg.T * CELL + 2, CELL + 2);
    }
    let gridLabel = 'attention scores  s = q·k';
    if (st.type === 'row' && t >= PH.scale && t < PH.smax) gridLabel = `scaled  s/√d = s/${Math.sqrt(cfg.d).toFixed(2)}`;
    else if ((st.type === 'row' && t >= PH.smax) || st.type === 'lm' || st.type === 'done') gridLabel = 'attention weights  softmax(s/√d)';
    txt(ctx, g.gx + cfg.T * CELL / 2, g.gy - 10, gridLabel, { align: 'center', mono: true, size: 11 });
    if (cfg.causal)
      txt(ctx, g.gx + cfg.T * CELL + 8, g.gy + cfg.T * CELL - 6, 'future masked', { size: 10.5 });

    // ---- values below ----
    txt(ctx, g.gx - 14, g.vY + 12, 'v =', { align: 'right', mono: true });
    for (let j = 0; j < cfg.T; j++) {
      const cx = g.gx + j * CELL + CELL / 2;
      let ring = null, alpha = projA;
      if (st.type === 'row' && t >= PH.smax && !masked(st.i, j)) ring = U.withAlpha('#d55181', 0.4 + 0.6 * Pr[st.i][j]);
      if (projA > 0) strip(ctx, cx - stripW(10) / 2, g.vY, V[j], 10, 13, alpha, ring);
    }

    // ---- outputs on the right ----
    txt(ctx, g.outX + stripW(12) / 2, g.gy - 26, 'output = Σ pⱼ·vⱼ', { align: 'center', mono: true, size: 11 });
    for (let i = 0; i < cfg.T; i++) {
      const cy = g.gy + i * CELL + CELL / 2;
      const state = rowState(si, i);
      const showFull = state === 'done' || (state === 'active' && st.type === 'row' && t >= 0.97);
      if (showFull) {
        strip(ctx, g.outX, cy - 8, O[i], 12, 16, 1);
      } else if (state === 'active' && st.type === 'row' && t >= PH.wsum - 0.001 && t >= PH.smax) {
        const tp = U.clamp((t - PH.smax) / (1 - PH.smax), 0, 1);
        strip(ctx, g.outX, cy - 8, O[i], 12, 16, tp);
        // flow curves from each value vector, weighted by attention
        for (let j = 0; j < nAvail(st.i); j++) {
          const p = Pr[st.i][j];
          if (p < 0.01) continue;
          const [vx, vy] = [g.gx + j * CELL + CELL / 2, g.vY + 7];
          ctx.save();
          ctx.strokeStyle = U.withAlpha('#d55181', (0.15 + 0.85 * p) * tp);
          ctx.lineWidth = 1 + 5 * p;
          ctx.beginPath();
          ctx.moveTo(vx, vy);
          ctx.quadraticCurveTo((vx + g.outX) / 2 + 60, vy + 60, g.outX - 4, cy);
          ctx.stroke();
          ctx.restore();
        }
      } else {
        ctx.save();
        ctx.strokeStyle = 'rgba(255,255,255,0.10)';
        ctx.strokeRect(g.outX - 0.5, cy - 8.5, stripW(12), 17);
        ctx.restore();
      }
    }

    // ---- language-model head ----
    if (st.type === 'lm' || st.type === 'done') drawLM(ctx, si, st.type === 'lm' ? U.easeInOut(t) : 1, g);

    // header
    txt(ctx, 60, 64, 'self-attention', { size: 15, weight: 650, color: '#ffffff' });
    txt(ctx, 60, 86, `"${words.join(' ')}"  ·  d = ${cfg.d}  ·  ${cfg.causal ? 'causal (decoder)' : 'bidirectional (encoder)'}`, { size: 12 });
    txt(ctx, 60, 108, 'every token asks (q) every earlier token (k)', { size: 11.5, color: '#63615c' });
    txt(ctx, 60, 124, 'how relevant it is, then blends their values (v)', { size: 11.5, color: '#63615c' });
  }

  function drawLM(ctx, si, a, g) {
    const x0 = 60, y0 = 310, bw = 120;
    txt(ctx, x0, y0 - 40, 'next-token prediction', { weight: 650, size: 13, color: '#ffffff' });
    txt(ctx, x0, y0 - 22, `logits = out("${words[cfg.T - 1]}") · Eᵀ`, { mono: true, size: 11 });
    const best = probs.indexOf(Math.max(...probs));
    vocab.forEach((w, k) => {
      const y = y0 + k * 26;
      const p = probs[k] * a;
      ctx.fillStyle = k === best && a > 0.9 ? '#3987e5' : '#2e4763';
      ctx.fillRect(x0 + 58, y, Math.max(2, bw * p), 15);
      txt(ctx, x0 + 50, y + 12, w, { align: 'right', mono: true, size: 11.5, color: k === best ? '#ffffff' : '#898781' });
      txt(ctx, x0 + 62 + bw * p, y + 12, (probs[k] * 100).toFixed(0) + '%', { mono: true, size: 10.5 });
    });
    if (a > 0.9)
      txt(ctx, x0, y0 + vocab.length * 26 + 20,
          `"${words.join(' ')} ${vocab[best]}"`, { mono: true, size: 11.5, color: '#c3c2b7' });
  }

  // ---------------- caption ----------------
  function caption(si, t) {
    const st = steps[si];
    switch (st.type) {
      case 'intro':
        return `Each token becomes a <b>${cfg.d}-dim embedding</b> vector (same word → same vector; positional encoding omitted here for clarity).`;
      case 'proj':
        return `Three learned projections turn every embedding into a <b>query</b>, a <b>key</b>, and a <b>value</b> vector.`;
      case 'row': {
        const i = st.i;
        if (t < PH.dots) return `"<b>${words[i]}</b>" compares its query with each key: one <b>dot product per token</b>${cfg.causal ? ' it may attend to (no peeking at the future)' : ''}`;
        if (t < PH.scale) return `Raw scores for row "<b>${words[i]}</b>" done — divide by <b>√d = ${Math.sqrt(cfg.d).toFixed(2)}</b> to keep them tame`;
        if (t < PH.smax) return `<b>softmax</b> turns the scaled scores into attention weights that sum to 1`;
        return `Blend the <b>value</b> vectors with those weights → the new representation of "<b>${words[i]}</b>"`;
      }
      case 'lm':
        return `Language modeling: the <b>last token's output</b> is scored against every vocab embedding — softmax gives the next-token distribution.`;
      case 'done':
        return `That's one attention head. Real transformers run many heads in parallel, add MLP blocks, and stack the whole thing dozens of times.`;
    }
    return '';
  }

  // ---------------- detail panel ----------------
  let dp = null;
  function buildDetail(el, si) {
    const st = steps[si];
    dp = null;
    if (st.type === 'row') {
      const i = st.i;
      let html = `<div class="dp-title">row "<b>${words[i]}</b>" — score, scale, softmax, blend</div>`;
      html += `<div class="dp-flow" data-id="dots">`;
      html += `<span class="dp-op" style="font-size:12px">q("${words[i]}")</span>`;
      html += vecHTML(Q[i], 'q');
      html += `<span class="dp-op">·</span><span class="dp-op" style="font-size:12px" data-id="kname">k("${words[0]}")</span>`;
      html += vecHTML(K[0], 'k');
      html += `<span class="dp-op">=</span><span class="chip" data-id="dot">…</span>`;
      html += `</div>`;
      html += `<div class="dp-sum" data-id="soft" style="display:none"></div>`;
      el.innerHTML = html;
      dp = {
        kname: el.querySelector('[data-id=kname]'),
        kcells: el.querySelectorAll('.dp-cell.k'),
        dot: el.querySelector('[data-id=dot]'),
        dots: el.querySelector('[data-id=dots]'),
        soft: el.querySelector('[data-id=soft]'),
        lastJ: -1, lastPhase: ''
      };
      return;
    }
    const msgs = {
      intro: 'Attention lets every token build its meaning from the tokens around it — "ate" needs to know who ate and what got eaten.',
      proj: 'q = "what am I looking for?" · k = "what do I contain?" · v = "what do I hand over if someone attends to me?"',
      lm: 'Weight tying: the same embedding matrix maps words in and scores words out.',
      done: 'Try toggling the causal mask, changing d, or replaying row by row with ⏮ ⏭.'
    };
    el.innerHTML = `<div class="dp-idle">${msgs[st.type] || ''}</div>`;
  }

  function vecHTML(v, cls) {
    let h = `<div class="dp-grid" style="grid-template-columns:repeat(${cfg.d},34px)">`;
    for (const x of v) h += `<div class="dp-cell ${cls}" style="width:34px;background:${U.divColor(x, 1.6)};color:${U.inkFor(U.divColor(x, 1.6))}">${x.toFixed(1)}</div>`;
    return h + '</div>';
  }

  function updateDetail(si, t) {
    const st = steps[si];
    if (!dp || st.type !== 'row') return;
    const i = st.i;
    if (t < PH.scale) {
      const j = Math.min(dotJ(st, t), nAvail(i) - 1);
      dp.dots.style.display = '';
      dp.soft.style.display = 'none';
      if (j !== dp.lastJ) {
        dp.lastJ = j;
        dp.kname.textContent = `k("${words[j]}")`;
        dp.kcells.forEach((c, k) => {
          const v = K[j][k];
          c.textContent = v.toFixed(1);
          c.style.background = U.divColor(v, 1.6);
          c.style.color = U.inkFor(U.divColor(v, 1.6));
        });
      }
      dp.dot.textContent = S[i][j].toFixed(2);
    } else {
      dp.dots.style.display = 'none';
      dp.soft.style.display = '';
      const phase = t < PH.smax ? 'scale' : t < PH.wsum ? 'smax' : 'smax';
      if (phase !== dp.lastPhase || true) {
        dp.lastPhase = phase;
        const n = nAvail(i);
        if (t < PH.smax) {
          dp.soft.innerHTML = `<span>scaled scores:</span>` +
            Array.from({ length: n }, (_, j) => `<span class="chip">${Sc[i][j].toFixed(2)}</span>`).join('') +
            `<span style="color:#63615c">(÷ √${cfg.d})</span>`;
        } else {
          dp.soft.innerHTML = `<span>softmax →</span>` +
            Array.from({ length: n }, (_, j) =>
              `<span class="chip" style="background:${U.seqColor(Pr[i][j])};color:${U.inkFor(U.seqColor(Pr[i][j]))}">${words[j]} ${Pr[i][j].toFixed(2)}</span>`).join('') +
            `<span>→ out = Σ pⱼ·vⱼ = [${O[i].map(v => v.toFixed(2)).join(', ')}]</span>`;
        }
      }
    }
  }

  // ---------------- controls ----------------
  function init(controlsEl, legendEl) {
    controlsEl.innerHTML = '';
    const g = document.createElement('div');
    g.className = 'ctl-group';
    g.innerHTML = '<h3>Attention</h3>';
    controlsEl.appendChild(g);
    num(g, 'tokens', 'T', () => cfg.T, v => cfg.T = v, 3, 6);
    num(g, 'model dim', 'd', () => cfg.d, v => cfg.d = v, 2, 6);

    const row = document.createElement('div');
    row.className = 'ctl-row';
    row.innerHTML = `<label>mask</label><select class="ctl-select">
      <option value="1">causal (GPT)</option><option value="0">none (BERT)</option></select>`;
    row.querySelector('select').value = cfg.causal ? '1' : '0';
    row.querySelector('select').onchange = e => { cfg.causal = e.target.value === '1'; regen(); App.resetTimeline(); };
    g.appendChild(row);

    const btn = document.createElement('button');
    btn.className = 'btn';
    btn.textContent = '⚄ Regenerate values';
    btn.onclick = () => { seed = (seed * 16807 + 9) % 2147483647; regen(); App.resetTimeline(); };
    controlsEl.appendChild(btn);

    legendEl.innerHTML =
      `<div class="legend-row"><span class="legend-swatch" style="background:#3987e5"></span><span>queries q</span></div>` +
      `<div class="legend-row"><span class="legend-swatch" style="background:#199e70"></span><span>keys k</span></div>` +
      `<div class="legend-row"><span class="legend-swatch" style="background:#d55181"></span><span>values v</span></div>` +
      `<div class="legend-row" style="margin-top:6px"><span>attention weight</span></div>` +
      (() => { const stops = []; for (let i = 0; i <= 10; i++) stops.push(U.seqColor(i / 10));
        return `<div class="legend-ramp" style="background:linear-gradient(90deg,${stops.join(',')})"></div>` +
               `<div class="legend-cap"><span>0</span><span>1</span></div>`; })();
  }

  function num(group, label, dim, get, set, lo, hi) {
    const row = document.createElement('div');
    row.className = 'ctl-row';
    row.innerHTML = `<label>${label} <span class="dim">${dim}</span></label>` +
      `<span class="ctl-num"><button data-d="-1">−</button><span class="val">${get()}</span><button data-d="1">+</button></span>`;
    const val = row.querySelector('.val');
    row.querySelectorAll('button').forEach(b => b.onclick = () => {
      set(U.clamp(get() + Number(b.dataset.d), lo, hi));
      regen(); val.textContent = get(); App.resetTimeline();
    });
    group.appendChild(row);
  }

  regen();

  return {
    id: 'attn',
    title: 'Attention',
    desc: 'Scaled dot-product self-attention: every token scores every (visible) token with q·k, softmax turns scores into weights, and the weights blend the value vectors. Plus a next-token prediction head.',
    VW, VH,
    get steps() { return steps; },
    init, regen, render, caption, buildDetail, updateDetail
  };
})();
