'use strict';
/* ============ Multi-head attention module ============ */
const MHAModule = (() => {

  const VW = 1280, VH = 640;
  const cfg = { T: 5, H: 2, dm: 8, causal: true };
  let seed = 91;

  const SENTENCE = ['the', 'robot', 'ate', 'an', 'apple', 'today'];

  let words = [], E = [], dk = 4;
  let Q = [], K = [], V = [], P = [], Ohead = [], Concat = [], Final = [];
  let absHead = 1, absOut = 1;
  let steps = [];

  // ---------------- data ----------------
  function regen() {
    const rnd = U.mulberry32(seed);
    const r1 = () => Math.round((rnd() * 2 - 1) * 10) / 10;
    dk = cfg.dm / cfg.H;
    words = SENTENCE.slice(0, cfg.T);

    const Ev = {};
    for (const w of new Set(words)) Ev[w] = Array.from({ length: cfg.dm }, r1);
    E = words.map(w => Ev[w]);

    const proj = (x, W) => W[0].map((_, j) =>
      Math.round(W.reduce((s, row, i) => s + row[j] * x[i], 0) * 10) / 10);

    Q = []; K = []; V = []; P = []; Ohead = [];
    for (let h = 0; h < cfg.H; h++) {
      const mk = () => Array.from({ length: cfg.dm }, () => Array.from({ length: dk }, r1));
      const Wq = mk(), Wk = mk(), Wv = mk();
      const q = E.map(x => proj(x, Wq)), k = E.map(x => proj(x, Wk)), v = E.map(x => proj(x, Wv));
      Q.push(q); K.push(k); V.push(v);
      const Ph = [];
      for (let i = 0; i < cfg.T; i++) {
        const sc = k.map((kv, j) => masked(i, j) ? -Infinity
          : q[i].reduce((s, x, d) => s + x * kv[d], 0) / Math.sqrt(dk));
        Ph.push(softmax(sc));
      }
      P.push(Ph);
      Ohead.push(E.map((_, i) => {
        const o = Array.from({ length: dk }, () => 0);
        for (let j = 0; j < cfg.T; j++)
          for (let d = 0; d < dk; d++) o[d] += Ph[i][j] * v[j][d];
        return o;
      }));
    }

    Concat = E.map((_, i) => [].concat(...Ohead.map(oh => oh[i])));
    const WO = Array.from({ length: cfg.dm }, () => Array.from({ length: cfg.dm }, r1));
    Final = Concat.map(c => WO[0].map((_, j) =>
      WO.reduce((s, row, i) => s + row[j] * c[i], 0)));

    absHead = Math.max(1, ...Ohead.flat(2).map(Math.abs));
    absOut = Math.max(1, ...Final.flat().map(Math.abs));
    buildSteps();
  }

  const masked = (i, j) => cfg.causal && j > i;
  function softmax(xs) {
    const m = Math.max(...xs.filter(x => x > -Infinity));
    const es = xs.map(x => x === -Infinity ? 0 : Math.exp(x - m));
    const Z = es.reduce((a, b) => a + b, 0);
    return es.map(e => e / Z);
  }

  function buildSteps() {
    steps = [{ type: 'intro', dur: 2.2 }, { type: 'proj', dur: 2.4 }];
    for (let h = 0; h < cfg.H; h++) steps.push({ type: 'head', h, dur: 2.8 });
    steps.push({ type: 'concat', dur: 2.4 }, { type: 'wo', dur: 2.2 }, { type: 'done', dur: 1.8 });
  }

  // ---------------- layout ----------------
  function G() {
    const pw = Math.min(310, (1180 - (cfg.H - 1) * 24) / cfg.H);
    const total = cfg.H * pw + (cfg.H - 1) * 24;
    const cell = Math.min(26, (pw - 44) / cfg.T);
    return {
      tokX: j => 640 - (cfg.T - 1) * 55 + j * 110,
      tokY: 66, embY: 88,
      px0: (1280 - total) / 2, pw, py: 158, ph: 236, cell,
      cRows: 436, cLabX: 380, cStripX: 402, rowH: 23,
      outX: 760, arrX: 690
    };
  }
  const headColor = h => U.CAT[h % U.CAT.length];

  // ---------------- draw helpers ----------------
  function strip(ctx, x, y, vals, absM, cw, chh, alpha = 1) {
    ctx.save();
    ctx.globalAlpha = alpha;
    for (let i = 0; i < vals.length; i++) {
      ctx.fillStyle = U.divColor(vals[i], absM);
      ctx.fillRect(x + i * cw, y, cw - 1, chh);
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.strokeRect(x - 0.5, y - 0.5, vals.length * cw, chh + 1);
    ctx.restore();
  }
  function chipTok(ctx, x, y, word, active, color) {
    ctx.save();
    ctx.font = '600 12px ui-monospace, Menlo, monospace';
    const w = ctx.measureText(word).width + 16;
    ctx.fillStyle = active ? U.withAlpha(color, 0.28) : '#242423';
    ctx.strokeStyle = active ? color : 'rgba(255,255,255,0.14)';
    ctx.lineWidth = active ? 1.7 : 1;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x - w / 2, y - 11, w, 22, 7); else ctx.rect(x - w / 2, y - 11, w, 22);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = active ? '#fff' : '#c3c2b7';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(word, x, y + 0.5);
    ctx.restore();
  }
  function txt(ctx, x, y, s, o = {}) {
    ctx.save();
    ctx.fillStyle = o.color || '#898781';
    ctx.font = `${o.weight || 500} ${o.size || 11.5}px ${o.mono ? 'ui-monospace, Menlo, monospace' : 'system-ui, sans-serif'}`;
    ctx.textAlign = o.align || 'left';
    ctx.textBaseline = o.base || 'middle';
    ctx.fillText(s, x, y);
    ctx.restore();
  }

  // ---------------- render ----------------
  function headState(si, h) {
    const st = steps[si];
    if (st.type === 'head') return h < st.h ? 'done' : h === st.h ? 'active' : 'pending';
    if (st.type === 'intro' || st.type === 'proj') return 'pending';
    return 'done';
  }

  function render(ctx, si, t) {
    const st = steps[si];
    const g = G();

    // tokens + embeddings
    const ew = Math.min(11, 96 / cfg.dm);
    for (let j = 0; j < cfg.T; j++) {
      chipTok(ctx, g.tokX(j), g.tokY, words[j], false, '#3987e5');
      strip(ctx, g.tokX(j) - cfg.dm * ew / 2, g.embY, E[j], 1.2, ew, 13);
    }
    txt(ctx, g.tokX(0) - 60, g.embY + 7, `emb (d=${cfg.dm})`, { align: 'right', mono: true, size: 10.5 });

    // projection curves during proj step
    if (st.type === 'proj') {
      const a = U.easeInOut(t);
      ctx.save();
      for (let h = 0; h < cfg.H; h++) {
        const hx = g.px0 + h * (g.pw + 24) + g.pw / 2;
        const ha = U.clamp(a * cfg.H - h * 0.5, 0, 1);
        ctx.strokeStyle = U.withAlpha(headColor(h), 0.45 * ha);
        ctx.lineWidth = 1.4;
        for (let j = 0; j < cfg.T; j++) {
          ctx.beginPath();
          ctx.moveTo(g.tokX(j), g.embY + 15);
          ctx.quadraticCurveTo(g.tokX(j), g.py - 20, hx, g.py + 4);
          ctx.stroke();
        }
      }
      ctx.restore();
    }

    // head panels
    for (let h = 0; h < cfg.H; h++) {
      drawHeadPanel(ctx, g, h, si, t);
    }

    // concat rows
    const showConcat = ['concat', 'wo', 'done'].includes(st.type);
    if (showConcat || st.type === 'head') drawConcat(ctx, g, si, t);

    // W_O -> final
    const showOut = st.type === 'wo' || st.type === 'done';
    if (showOut) {
      const a = st.type === 'wo' ? U.easeInOut(t) : 1;
      ctx.save();
      ctx.globalAlpha = a;
      txt(ctx, g.arrX + 18, g.cRows + (cfg.T * g.rowH) / 2 - 10, '× W_O', { mono: true, size: 12, color: '#ffffff', align: 'center', weight: 600 });
      txt(ctx, g.arrX + 18, g.cRows + (cfg.T * g.rowH) / 2 + 8, '→', { size: 16, align: 'center' });
      const ow = Math.min(13, 110 / cfg.dm);
      for (let i = 0; i < cfg.T; i++)
        strip(ctx, g.outX, g.cRows + 14 + i * g.rowH, Final[i], absOut, ow, 15, a);
      txt(ctx, g.outX + cfg.dm * ow / 2, g.cRows, 'output (T×d) — heads mixed', { align: 'center', mono: true, size: 10.5 });
      ctx.restore();
    }

    txt(ctx, 46, 46, 'multi-head attention', { size: 15, weight: 650, color: '#ffffff' });
    txt(ctx, 46, 66, `H = ${cfg.H} heads · d_k = d/H = ${dk} · ${cfg.causal ? 'causal' : 'no mask'}`, { size: 12 });
  }

  function drawHeadPanel(ctx, g, h, si, t) {
    const st = steps[si];
    const x = g.px0 + h * (g.pw + 24), y = g.py;
    const state = headState(si, h);
    const col = headColor(h);
    const visible = state !== 'pending' || st.type === 'proj';
    const a = st.type === 'proj' ? U.clamp(U.easeInOut(t) * cfg.H - h * 0.5, 0, 1)
            : visible ? 1 : 0.25;

    ctx.save();
    ctx.globalAlpha = Math.max(a, 0.2);
    ctx.strokeStyle = state === 'active' ? col : U.withAlpha(col, 0.4);
    ctx.lineWidth = state === 'active' ? 2 : 1.2;
    ctx.fillStyle = '#1e1e1d';
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, g.pw, g.ph, 10); else ctx.rect(x, y, g.pw, g.ph);
    ctx.fill(); ctx.stroke();

    txt(ctx, x + g.pw / 2, y + 16, `head ${h}`, { align: 'center', weight: 650, size: 12.5, color: col });
    txt(ctx, x + g.pw / 2, y + 32, `own W_q W_k W_v : d→${dk}`, { align: 'center', mono: true, size: 9.5 });

    // attention grid
    const cell = g.cell;
    const gx = x + (g.pw - cfg.T * cell) / 2, gy = y + 46;
    let rowsShown = 0;
    if (state === 'done') rowsShown = cfg.T;
    else if (state === 'active') rowsShown = U.clamp(t / 0.7, 0, 1) * cfg.T;

    for (let i = 0; i < cfg.T; i++) {
      for (let j = 0; j < cfg.T; j++) {
        const cx2 = gx + j * cell, cy2 = gy + i * cell;
        let fill = '#20201f';
        const shown = i < rowsShown;
        if (masked(i, j)) fill = '#191918';
        else if (shown) fill = U.seqColor(P[h][i][j]);
        ctx.fillStyle = fill;
        ctx.fillRect(cx2, cy2, cell - 1.5, cell - 1.5);
        if (shown && !masked(i, j) && cell >= 22) {
          ctx.fillStyle = U.inkFor(fill);
          ctx.font = '600 8.5px ui-monospace, Menlo, monospace';
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(P[h][i][j].toFixed(2), cx2 + cell / 2, cy2 + cell / 2);
        }
      }
    }
    // current row marker
    if (state === 'active' && rowsShown < cfg.T) {
      const ri = Math.floor(rowsShown);
      ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5;
      ctx.strokeRect(gx - 1, gy + ri * cell - 1, cfg.T * cell + 1, cell + 1);
    }
    // token initials on axes
    ctx.font = '600 8.5px ui-monospace, Menlo, monospace';
    ctx.fillStyle = '#898781';
    ctx.textAlign = 'center';
    for (let j = 0; j < cfg.T; j++)
      ctx.fillText(words[j][0], gx + j * cell + cell / 2, gy - 7);
    ctx.textAlign = 'right';
    for (let i = 0; i < cfg.T; i++)
      ctx.fillText(words[i][0], gx - 5, gy + i * cell + cell / 2);

    txt(ctx, x + g.pw / 2, gy + cfg.T * cell + 14, 'softmax(q·kᵀ/√' + dk + ')', { align: 'center', mono: true, size: 9.5 });
    ctx.restore();
  }

  function drawConcat(ctx, g, si, t) {
    const st = steps[si];
    const cw = Math.min(13, 110 / cfg.dm);
    txt(ctx, g.cStripX + cfg.dm * cw / 2, g.cRows, `concat heads → (T, ${cfg.H}·${dk})`, { align: 'center', mono: true, size: 10.5 });

    for (let i = 0; i < cfg.T; i++) {
      const y = g.cRows + 14 + i * g.rowH;
      txt(ctx, g.cLabX, y + 7, words[i], { align: 'right', mono: true, size: 10.5, color: '#c3c2b7' });
      for (let h = 0; h < cfg.H; h++) {
        let a = 0;
        if (st.type === 'head') a = h < st.h ? 1 : h === st.h && t > 0.75 ? (t - 0.75) / 0.25 : 0;
        else if (st.type === 'concat') a = U.clamp(U.easeInOut(t) * cfg.H - h * 0.6, 0, 1);
        else a = 1;
        if (a <= 0) {
          ctx.save();
          ctx.strokeStyle = 'rgba(255,255,255,0.07)';
          ctx.strokeRect(g.cStripX + h * dk * cw, y, dk * cw, 15);
          ctx.restore();
          continue;
        }
        strip(ctx, g.cStripX + h * dk * cw, y, Ohead[h][i], absHead, cw, 15, a);
        // head-colored provenance underline
        ctx.save();
        ctx.globalAlpha = a;
        ctx.fillStyle = headColor(h);
        ctx.fillRect(g.cStripX + h * dk * cw, y + 17, dk * cw - 1, 2.5);
        ctx.restore();
      }
    }
    // flow curves from panels during concat
    if (st.type === 'concat') {
      const a = U.easeInOut(t);
      ctx.save();
      for (let h = 0; h < cfg.H; h++) {
        const ha = U.clamp(a * cfg.H - h * 0.6, 0, 1);
        const hx = g.px0 + h * (g.pw + 24) + g.pw / 2;
        ctx.strokeStyle = U.withAlpha(headColor(h), 0.5 * ha);
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(hx, g.py + g.ph + 2);
        ctx.quadraticCurveTo(hx, g.cRows - 16, g.cStripX + h * dk * cw + dk * cw / 2, g.cRows + 10);
        ctx.stroke();
      }
      ctx.restore();
    }
  }

  // ---------------- captions ----------------
  function caption(si, t) {
    const st = steps[si];
    switch (st.type) {
      case 'intro':
        return `One head must average all its evidence into a single pattern. <b>Multi-head</b> attention runs <b>${cfg.H} heads in parallel</b>, each in a smaller d_k = ${dk} subspace.`;
      case 'proj':
        return `Every head gets its <b>own</b> learned W_q, W_k, W_v — the same ${cfg.dm}-d embeddings, projected ${cfg.H} different ways into ${dk}-d.`;
      case 'head':
        return `<b>Head ${st.h}</b> runs plain scaled dot-product attention — note its pattern differs from the other head${cfg.H > 2 ? 's' : ''}: different projections → different “questions asked”.`;
      case 'concat':
        return `Each head hands back a ${dk}-d output per token — <b>concatenate</b> them side by side: (T, ${cfg.H}×${dk}) = (T, ${cfg.dm}). Colored underlines show which head produced which slice.`;
      case 'wo':
        return `A final linear layer <b>W_O</b> (${cfg.dm}×${cfg.dm}) mixes the heads' slices together — after this, head boundaries disappear.`;
      case 'done':
        return `Same cost as one big head, but <b>${cfg.H} independent attention patterns</b>. This whole thing is the “attention” inside a transformer block →`;
    }
    return '';
  }

  // ---------------- detail ----------------
  function buildDetail(el, si) {
    const st = steps[si];
    if (st.type === 'head') {
      const h = st.h;
      let html = `<div class="dp-title">head <b>${h}</b> — attention weights (rows = queries, cols = keys)</div>`;
      html += `<div style="display:flex;gap:3px;flex-direction:column">`;
      html += `<div style="display:flex;gap:3px"><span style="width:52px"></span>` +
        words.map(w => `<span style="width:44px;text-align:center;font:600 10px var(--mono);color:var(--ink-muted)">${w}</span>`).join('') + `</div>`;
      for (let i = 0; i < cfg.T; i++) {
        html += `<div style="display:flex;gap:3px;align-items:center">` +
          `<span style="width:52px;text-align:right;font:600 10.5px var(--mono);color:var(--ink-2);padding-right:4px">${words[i]}</span>`;
        for (let j = 0; j < cfg.T; j++) {
          const mk = masked(i, j);
          const bg = mk ? '#191918' : U.seqColor(P[h][i][j]);
          html += `<span style="width:44px;height:20px;border-radius:4px;background:${bg};color:${mk ? '#565550' : U.inkFor(bg)};font:600 10px var(--mono);display:flex;align-items:center;justify-content:center">${mk ? '·' : P[h][i][j].toFixed(2)}</span>`;
        }
        html += '</div>';
      }
      html += '</div>';
      el.innerHTML = html;
      return;
    }
    const msgs = {
      intro: `Why multiple heads? One pattern can track syntax while another tracks meaning — “ate” can simultaneously look at who ate (head 0) and what was eaten (head 1).`,
      proj: `Head h: qₕ = E·W_qₕ, kₕ = E·W_kₕ, vₕ = E·W_vₕ — each (${cfg.dm}→${dk}). Total projection cost equals one full-size head.`,
      concat: `concat(head₀ … head${cfg.H - 1}) per token: ${cfg.H} slices of ${dk} numbers → one ${cfg.dm}-d vector. Nothing is mixed yet — slices just sit side by side.`,
      wo: `out = concat · W_O. Every output number is a weighted blend of all heads — W_O learns how much to trust each head.`,
      done: `Full picture: MultiHead(x) = concat(head₀…head${cfg.H - 1}) · W_O, with headₕ = softmax(qₕkₕᵀ/√${dk}) · vₕ. Now see it inside a full block: Transformer tab.`
    };
    el.innerHTML = `<div class="dp-idle">${msgs[st.type] || ''}</div>`;
  }
  function updateDetail() {}

  // ---------------- controls ----------------
  function init(controlsEl, legendEl) {
    controlsEl.innerHTML = '';
    const g = document.createElement('div');
    g.className = 'ctl-group';
    g.innerHTML = '<h3>Multi-head attention</h3>';
    controlsEl.appendChild(g);
    stepList(g, 'tokens T', () => cfg.T, v => cfg.T = v, [3, 4, 5, 6]);
    stepList(g, 'model dim d', () => cfg.dm, v => cfg.dm = v, [4, 8]);
    stepList(g, 'heads H', () => cfg.H, v => cfg.H = v, [1, 2, 4]);

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
    btn.onclick = () => { seed = (seed * 16807 + 17) % 2147483647; regen(); App.resetTimeline(); };
    controlsEl.appendChild(btn);

    legendEl.innerHTML =
      Array.from({ length: cfg.H }, (_, h) =>
        `<div class="legend-row"><span class="legend-swatch" style="background:${headColor(h)}"></span><span>head ${h}</span></div>`).join('') +
      `<div class="legend-row" style="margin-top:6px"><span>attention weight</span></div>` +
      (() => { const s2 = []; for (let i = 0; i <= 10; i++) s2.push(U.seqColor(i / 10));
        return `<div class="legend-ramp" style="background:linear-gradient(90deg,${s2.join(',')})"></div>` +
               `<div class="legend-cap"><span>0</span><span>1</span></div>`; })();
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
      if (cfg.H > cfg.dm) cfg.H = cfg.dm;      // keep d_k ≥ 1
      regen(); val.textContent = get();
      App.resetTimeline();
      // legend head count may change
      init(document.getElementById('controls'), document.getElementById('legend'));
    });
    g.appendChild(row);
  }

  regen();

  return {
    id: 'mha',
    title: 'Multi-Head Attention',
    desc: 'H attention heads run in parallel, each with its own q/k/v projections into a smaller d/H subspace and its own attention pattern. Their outputs are concatenated and mixed by a final linear layer W_O.',
    VW, VH,
    get steps() { return steps; },
    init, regen, render, caption, buildDetail, updateDetail
  };
})();
