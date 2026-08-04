'use strict';
/* ============ MLP (fully-connected) module ============ */
const MLPModule = (() => {

  const VW = 1280, VH = 620;
  const cfg = { nIn: 6, nHid: 5, nOut: 3, act: 'relu' };
  let seed = 33;

  let sizes = [], A = [], Z = [], Wm = [], Bm = [], absA = [];
  let steps = [];

  const PH = { gather: 0.55, sum: 0.75, act: 1.0 };

  const ACTS = {
    relu:    { fn: z => Math.max(0, z),                name: 'ReLU' },
    tanh:    { fn: z => Math.round(Math.tanh(z) * 100) / 100, name: 'tanh' },
    sigmoid: { fn: z => Math.round(1 / (1 + Math.exp(-z)) * 100) / 100, name: 'σ' },
  };

  function regen() {
    const rnd = U.mulberry32(seed);
    const ri = (lo, hi) => lo + Math.floor(rnd() * (hi - lo + 1));
    sizes = [cfg.nIn, cfg.nHid, cfg.nOut];

    A = [Array.from({ length: cfg.nIn }, () => ri(0, 5))];
    Z = [null]; Wm = []; Bm = [];
    for (let l = 1; l < sizes.length; l++) {
      const W = [], b = [];
      for (let j = 0; j < sizes[l]; j++) {
        const row = [];
        for (let i = 0; i < sizes[l - 1]; i++) row.push(ri(-2, 2));
        W.push(row); b.push(ri(-2, 2));
      }
      Wm.push(W); Bm.push(b);
      const z = [], a = [];
      for (let j = 0; j < sizes[l]; j++) {
        let s = b[j];
        for (let i = 0; i < sizes[l - 1]; i++) s += W[j][i] * A[l - 1][i];
        z.push(s);
        a.push(l === sizes.length - 1 ? s : ACTS[cfg.act].fn(s)); // linear output layer
      }
      Z.push(z); A.push(a);
    }
    absA = A.map(layer => Math.max(1, ...layer.map(v => Math.abs(v))));
    buildSteps();
  }

  function buildSteps() {
    steps = [{ type: 'intro', dur: 2.2 }];
    for (let l = 1; l < sizes.length; l++)
      for (let j = 0; j < sizes[l]; j++)
        steps.push({ type: 'neuron', l, j, dur: 2.3 });
    steps.push({ type: 'done', dur: 1.5 });
  }

  // ---------- geometry ----------
  function nodePos(l, j) {
    const xs = [290, 650, 1010];
    const x = xs[l];
    const n = sizes[l];
    const gap = Math.min(84, 480 / Math.max(1, n - 1) || 480);
    const y0 = 290 - gap * (n - 1) / 2;
    return [x, y0 + j * gap];
  }
  const R = 25;

  function neuronState(si, l, j) {
    // 'done' | 'active' | 'pending'  (input layer always done)
    if (l === 0) return 'done';
    const st = steps[si];
    if (st.type === 'done') return 'done';
    if (st.type === 'intro') return 'pending';
    const order = (ll, jj) => { let n = 0; for (let x = 1; x < ll; x++) n += sizes[x]; return n + jj; };
    const cur = order(st.l, st.j), me = order(l, j);
    return me < cur ? 'done' : me === cur ? 'active' : 'pending';
  }

  // ---------- render ----------
  function render(ctx, si, t) {
    const st = steps[si];

    // edges
    for (let l = 1; l < sizes.length; l++) {
      for (let j = 0; j < sizes[l]; j++) {
        const state = neuronState(si, l, j);
        for (let i = 0; i < sizes[l - 1]; i++) {
          const w = Wm[l - 1][j][i];
          const [x0, y0] = nodePos(l - 1, i);
          const [x1, y1] = nodePos(l, j);
          let alpha = 0.10, lw = 0.8 + Math.abs(w) * 0.5;
          if (state === 'done') alpha = 0.28;
          let hot = false;
          if (state === 'active' && st.type === 'neuron') {
            const k = gatherIndex(si, t);
            if (i <= k) { alpha = 0.9; hot = i === k && t < PH.gather; lw = 1.4 + Math.abs(w) * 1.1; }
            else alpha = 0.15;
          }
          ctx.save();
          ctx.globalAlpha = alpha;
          ctx.strokeStyle = w === 0 ? '#565550' : U.divColor(w, 2);
          ctx.lineWidth = hot ? lw + 1 : lw;
          ctx.beginPath();
          ctx.moveTo(x0 + R, y0);
          ctx.lineTo(x1 - R, y1);
          ctx.stroke();
          ctx.restore();

          // traveling dot on hot edge
          if (hot) {
            const tt = (t / PH.gather * sizes[l - 1]) % 1;
            const dx = U.lerp(x0 + R, x1 - R, tt), dy = U.lerp(y0, y1, tt);
            ctx.save();
            ctx.fillStyle = '#ffffff';
            ctx.beginPath(); ctx.arc(dx, dy, 3.4, 0, 7); ctx.fill();
            ctx.restore();
          }
        }
      }
    }

    // nodes
    for (let l = 0; l < sizes.length; l++) {
      for (let j = 0; j < sizes[l]; j++) {
        const [x, y] = nodePos(l, j);
        const state = neuronState(si, l, j);
        const v = A[l][j];
        let fill = '#20242b', txt = '', ring = 'rgba(255,255,255,0.14)';
        if (state === 'done') {
          fill = l === 0 ? U.seqColor(v / 5) : U.divColor(v, absA[l]);
          txt = U.fmt(v); ring = 'rgba(0,0,0,0.4)';
        } else if (state === 'active') {
          ring = '#eda100';
          if (st.type === 'neuron' && t >= PH.sum) { fill = '#2c2c28'; txt = U.fmt(Z[l][j]); }
          if (st.type === 'neuron' && t >= 0.92) { fill = U.divColor(v, absA[l]); txt = U.fmt(v); }
        }
        ctx.save();
        ctx.beginPath(); ctx.arc(x, y, R, 0, 7);
        ctx.fillStyle = fill; ctx.fill();
        ctx.lineWidth = state === 'active' ? 2.6 : 1.2;
        ctx.strokeStyle = ring; ctx.stroke();
        if (txt !== '') {
          ctx.fillStyle = U.inkFor(typeof fill === 'string' && fill[0] === '#' ? fill : '#20242b');
          ctx.font = '650 12.5px ui-monospace, Menlo, monospace';
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(txt, x, y + 0.5);
        }
        ctx.restore();
      }
    }

    // layer labels
    const labels = [
      `input  x  (${cfg.nIn})`,
      `hidden  (${cfg.nHid}) · ${ACTS[cfg.act].name}`,
      `output  (${cfg.nOut}) · linear`];
    for (let l = 0; l < 3; l++) {
      const [x] = nodePos(l, 0);
      ctx.save();
      ctx.fillStyle = '#898781';
      ctx.font = '600 12px ui-monospace, Menlo, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(labels[l], x, 552);
      ctx.restore();
    }
    ctx.save();
    ctx.fillStyle = '#63615c';
    ctx.font = '500 11.5px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('in a CNN this input is the flattened stack of pooled feature maps', 290, 576);
    ctx.restore();
  }

  function gatherIndex(si, t) {
    const st = steps[si];
    const n = sizes[st.l - 1];
    if (t >= PH.gather) return n - 1 + 1; // all
    return Math.min(Math.floor(t / PH.gather * n), n - 1);
  }

  // ---------- caption ----------
  function caption(si, t) {
    const st = steps[si];
    switch (st.type) {
      case 'intro':
        return `An <b>MLP</b>: every neuron takes <b>all</b> activations from the previous layer — multiply each by its own weight, sum, add bias, apply the activation.`;
      case 'neuron': {
        const layer = st.l === sizes.length - 1 ? 'output' : 'hidden';
        if (t < PH.gather) return `${layer} neuron <b>${st.j}</b> gathers all ${sizes[st.l - 1]} inputs, each scaled by its weight`;
        if (t < PH.sum) return `Weighted sum + bias → <b>z = ${U.fmt(Z[st.l][st.j])}</b>`;
        return st.l === sizes.length - 1
          ? `Output layer stays linear → <b>${U.fmt(A[st.l][st.j])}</b>`
          : `${ACTS[cfg.act].name}(${U.fmt(Z[st.l][st.j])}) = <b>${U.fmt(A[st.l][st.j])}</b> — non-linearity is what lets layers stack usefully`;
      }
      case 'done':
        return `Forward pass complete: (${sizes.join(' → ')}). Every arrow was one multiply — ${cfg.nIn * cfg.nHid + cfg.nHid * cfg.nOut} weights total.`;
    }
    return '';
  }

  // ---------- detail ----------
  let dp = null;
  function buildDetail(el, si) {
    const st = steps[si];
    dp = null;
    if (st.type !== 'neuron') {
      el.innerHTML = `<div class="dp-idle">${st.type === 'intro'
        ? 'Watch each neuron compute  z = Σ wᵢ·aᵢ + b,  then squash it with the activation.'
        : 'The equation panel replays one neuron at a time — step back through with ⏮.'}</div>`;
      return;
    }
    const l = st.l, j = st.j;
    const n = sizes[l - 1];
    let html = `<div class="dp-title">${l === sizes.length - 1 ? 'output' : 'hidden'} neuron <b>${j}</b></div>`;
    html += `<div class="dp-eq">z = `;
    for (let i = 0; i < n; i++) {
      const w = Wm[l - 1][j][i], a = A[l - 1][i];
      html += `<span class="term" data-i="${i}">(${w >= 0 ? '+' : ''}${w})·${U.fmt(a)}</span>`;
      if (i < n - 1) html += ' + ';
    }
    const b = Bm[l - 1][j];
    html += `  ${b >= 0 ? '+' : '−'} <span class="term bias">${Math.abs(b)}</span>`;
    html += `  =  <b data-id="z">…</b>`;
    if (l !== sizes.length - 1)
      html += `<br>a = ${ACTS[cfg.act].name}(z) = <b data-id="a">…</b>`;
    html += '</div>';
    el.innerHTML = html;
    dp = { terms: el.querySelectorAll('.term[data-i]'), z: el.querySelector('[data-id=z]'),
           a: el.querySelector('[data-id=a]'), st };
  }

  function updateDetail(si, t) {
    const st = steps[si];
    if (!dp || st.type !== 'neuron') return;
    const k = gatherIndex(si, t);
    dp.terms.forEach((term, i) => term.classList.toggle('hot', i === k && t < PH.gather));
    dp.z.textContent = t >= PH.gather ? U.fmt(Z[st.l][st.j]) : '…';
    if (dp.a) dp.a.textContent = t >= PH.sum ? U.fmt(A[st.l][st.j]) : '…';
  }

  // ---------- controls ----------
  function init(controlsEl, legendEl) {
    controlsEl.innerHTML = '';
    const g = document.createElement('div');
    g.className = 'ctl-group';
    g.innerHTML = '<h3>Layers</h3>';
    controlsEl.appendChild(g);
    num(g, 'input size', '', () => cfg.nIn, v => cfg.nIn = v, 2, 8);
    num(g, 'hidden size', '', () => cfg.nHid, v => cfg.nHid = v, 2, 8);
    num(g, 'output size', '', () => cfg.nOut, v => cfg.nOut = v, 1, 6);

    const row = document.createElement('div');
    row.className = 'ctl-row';
    row.innerHTML = `<label>activation</label><select class="ctl-select">
      <option value="relu">ReLU</option><option value="tanh">tanh</option><option value="sigmoid">sigmoid</option></select>`;
    row.querySelector('select').value = cfg.act;
    row.querySelector('select').onchange = e => { cfg.act = e.target.value; regen(); App.resetTimeline(); };
    g.appendChild(row);

    const btn = document.createElement('button');
    btn.className = 'btn';
    btn.textContent = '⚄ Regenerate values';
    btn.onclick = () => { seed = (seed * 16807 + 5) % 2147483647; regen(); App.resetTimeline(); };
    controlsEl.appendChild(btn);

    legendEl.innerHTML = '';
  }

  function num(group, label, dim, get, set, lo, hi) {
    const row = document.createElement('div');
    row.className = 'ctl-row';
    row.innerHTML = `<label>${label}</label>` +
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
    id: 'mlp',
    title: 'MLP',
    desc: 'A multi-layer perceptron: dense layers where every neuron computes a weighted sum of all previous activations, adds a bias, and applies a non-linearity.',
    VW, VH,
    get steps() { return steps; },
    init, regen, render, caption, buildDetail, updateDetail
  };
})();
