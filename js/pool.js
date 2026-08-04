'use strict';
/* ============ Max Pooling module ============ */
const PoolModule = (() => {

  const VW = 1280, VH = 620;
  const cfg = { C: 3, H: 6, W: 6, K: 2, S: 2, mode: 'max' };
  let seed = 21;

  let X = [], Out = [], Ho = 0, Wo = 0;
  let steps = [];

  const PH = { move: 0.22, pick: 0.70, write: 1.0 };

  function regen() {
    const rnd = U.mulberry32(seed);
    const ri = (lo, hi) => lo + Math.floor(rnd() * (hi - lo + 1));
    Ho = Math.floor((cfg.H - cfg.K) / cfg.S) + 1;
    Wo = Math.floor((cfg.W - cfg.K) / cfg.S) + 1;
    X = []; Out = [];
    for (let c = 0; c < cfg.C; c++) {
      const g = [];
      for (let r = 0; r < cfg.H; r++) { const row = []; for (let cc = 0; cc < cfg.W; cc++) row.push(ri(0, 9)); g.push(row); }
      X.push(g);
    }
    for (let c = 0; c < cfg.C; c++) {
      const m = [];
      for (let or = 0; or < Ho; or++) {
        const row = [];
        for (let oc = 0; oc < Wo; oc++) row.push(poolAt(c, or, oc).v);
        m.push(row);
      }
      Out.push(m);
    }
    buildSteps();
  }

  function poolAt(c, or, oc) {
    let v = cfg.mode === 'max' ? -Infinity : 0;
    let arg = [0, 0];
    const win = [];
    for (let kr = 0; kr < cfg.K; kr++)
      for (let kc = 0; kc < cfg.K; kc++) {
        const xv = X[c][or * cfg.S + kr][oc * cfg.S + kc];
        win.push(xv);
        if (cfg.mode === 'max') { if (xv > v) { v = xv; arg = [kr, kc]; } }
        else v += xv;
      }
    if (cfg.mode === 'avg') v = Math.round(v / (cfg.K * cfg.K) * 100) / 100;
    return { v, arg, win };
  }

  function buildSteps() {
    steps = [{ type: 'intro', dur: 2.0 }];
    let prev = null;
    for (let or = 0; or < Ho; or++)
      for (let oc = 0; oc < Wo; oc++) {
        steps.push({ type: 'slide', or, oc, prev, dur: 2.0 });
        prev = { or, oc };
      }
    steps.push({ type: 'done', dur: 1.5 });
  }

  function layout(ctx) {
    const s = U.clamp(480 / (0.9 * (cfg.W + cfg.C) + 2), 18, 40);
    const inIso = Iso.make(ctx, s, 215 + cfg.C * s * 0.9, 100 + cfg.H * s * 0.92);
    const os = U.clamp(380 / (0.9 * (Wo + cfg.C) + 2), 18, 40);
    const oIso = Iso.make(ctx, os, 870 + cfg.C * os * 0.9, 185 + Ho * os * 0.92);
    return { inIso, oIso };
  }

  function doneCount(si) {
    const st = steps[si];
    if (st.type === 'done') return Ho * Wo;
    let n = 0;
    for (let i = 0; i < si; i++) if (steps[i].type === 'slide') n++;
    return n;
  }

  function currentWindow(si, t) {
    const st = steps[si];
    if (st.type !== 'slide') return null;
    const e = U.easeInOut(Math.min(t / PH.move, 1));
    const pc = st.prev ? st.prev.oc * cfg.S : st.oc * cfg.S;
    const pr = st.prev ? st.prev.or * cfg.S : st.or * cfg.S;
    return { col: U.lerp(pc, st.oc * cfg.S, e), row: U.lerp(pr, st.or * cfg.S, e), settled: t >= PH.move };
  }

  function render(ctx, si, t) {
    const st = steps[si];
    const L = layout(ctx);
    drawCube(ctx, L.inIso, X, cfg.H, cfg.W, si, t, true);
    drawOut(ctx, L.oIso, si, t);
    if (st.type === 'slide') drawFly(ctx, L, st, t);

    ctx.save();
    ctx.fillStyle = '#898781';
    ctx.font = '300 30px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('→', 700, 300);
    ctx.font = '500 12px system-ui, sans-serif';
    ctx.fillText(cfg.mode === 'max' ? 'keep the max' : 'average', 700, 328);
    ctx.restore();
  }

  function drawCube(ctx, iso, T, H, W, si, t, isInput) {
    const C = cfg.C;
    Iso.box(iso, 0, 0, 0, W, H, C, '#1f232b');
    for (let d = 0; d < C; d++) {
      const ch = C - 1 - d;
      for (let c = 0; c < W; c++)
        Iso.topCell(iso, 0, 0, H, c, d, U.shade(U.seqColor(T[ch][0][c] / 9), 1.25));
      for (let r = 0; r < H; r++)
        Iso.rightCell(iso, 0, 0, W, d, r, H, U.shade(U.seqColor(T[C - 1 - d][r][W - 1] / 9), 0.62));
    }
    for (let r = 0; r < H; r++)
      for (let c = 0; c < W; c++)
        Iso.cell(iso, 0, 0, C, c, r, H, U.seqColor(T[0][r][c] / 9), String(T[0][r][c]));

    const win = currentWindow(si, t);
    if (win && isInput) {
      const yb = H - win.row - cfg.K;
      Iso.wire(iso, win.col, yb, 0, cfg.K, cfg.K, C, '#eda100', 2, 0.9);
      const q = [iso.p(win.col, yb, C), iso.p(win.col + cfg.K, yb, C),
                 iso.p(win.col + cfg.K, yb + cfg.K, C), iso.p(win.col, yb + cfg.K, C)];
      Iso.poly(ctx, q, U.withAlpha('#eda100', win.settled ? 0.18 : 0.28), '#eda100', 2.4);

      // highlight the max cell (channel 0 face) once picking
      const st = steps[si];
      if (t >= PH.move && st.type === 'slide') {
        const { arg } = poolAt(0, st.or, st.oc);
        const tp = U.clamp((t - PH.move) / (PH.pick - PH.move), 0, 1);
        if (cfg.mode === 'max' && tp > 0.3) {
          const q2 = Iso.cellQuad(iso, 0, 0, C, st.oc * cfg.S + arg[1], st.or * cfg.S + arg[0], H);
          Iso.poly(ctx, q2, null, '#ffffff', 2.6);
        }
      }
    }
    Iso.label(iso, W * 0.5, C, `input  (${C}×${H}×${W}) — front face = channel 0`, { dy: 8 });
    Iso.label(iso, W * 0.5, C, `${cfg.mode === 'max' ? 'max' : 'avg'}-pool ${cfg.K}×${cfg.K}, stride ${cfg.S} — every channel pooled independently`, { dy: 24 });
  }

  function drawOut(ctx, iso, si, t) {
    const st = steps[si];
    const C = cfg.C;
    const nd = doneCount(si);
    Iso.box(iso, 0, 0, 0, Wo, Ho, C, '#1f232b');
    for (let d = 0; d < C; d++) {
      const ch = C - 1 - d;
      for (let c = 0; c < Wo; c++) {
        const idx = 0 * Wo + c;
        if (idx < nd || st.type === 'done')
          Iso.topCell(iso, 0, 0, Ho, c, d, U.shade(U.seqColor(Out[ch][0][c] / 9), 1.25));
      }
    }
    for (let r = 0; r < Ho; r++)
      for (let c = 0; c < Wo; c++) {
        const idx = r * Wo + c;
        let show = idx < nd || st.type === 'done';
        let isCurrent = st.type === 'slide' && st.or === r && st.oc === c;
        if (isCurrent && t >= PH.write - 0.02) show = true;
        if (show)
          Iso.cell(iso, 0, 0, C, c, r, Ho, U.seqColor(Out[0][r][c] / 9), U.fmt(Out[0][r][c]));
        else
          Iso.cell(iso, 0, 0, C, c, r, Ho, '#20201f', null, { stroke: 'rgba(255,255,255,0.07)' });
        if (isCurrent) {
          const q = Iso.cellQuad(iso, 0, 0, C, c, r, Ho);
          Iso.poly(ctx, q, null, '#eda100', 2.4);
        }
      }
    Iso.wire(iso, 0, 0, 0, Wo, Ho, C, 'rgba(237,161,0,0.55)', 1.4);
    Iso.label(iso, Wo * 0.5, C, `output  (${C}×${Ho}×${Wo})`, { dy: 8 });
  }

  function drawFly(ctx, L, st, t) {
    if (t < PH.pick) return;
    const v = Out[0][st.or][st.oc];
    const yb = cfg.H - st.or * cfg.S - cfg.K;
    const [x0, y0] = L.inIso.p(st.oc * cfg.S + cfg.K / 2, yb + cfg.K / 2, cfg.C);
    const [x1, y1] = Iso.cellCenter(L.oIso, 0, 0, cfg.C, st.oc, st.or, Ho);
    const tp = U.easeInOut((t - PH.pick) / (PH.write - PH.pick));
    const mx = (x0 + x1) / 2, my = Math.min(y0, y1) - 80;
    const bx = (1 - tp) * (1 - tp) * x0 + 2 * (1 - tp) * tp * mx + tp * tp * x1;
    const by = (1 - tp) * (1 - tp) * y0 + 2 * (1 - tp) * tp * my + tp * tp * y1;
    ctx.save();
    ctx.strokeStyle = U.withAlpha('#eda100', 0.3);
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.quadraticCurveTo(mx, my, x1, y1); ctx.stroke();
    ctx.restore();
    if (tp < 0.97) {
      ctx.save();
      ctx.font = '650 13px ui-monospace, Menlo, monospace';
      const txt = U.fmt(v);
      const w = ctx.measureText(txt).width + 16;
      ctx.fillStyle = '#111214'; ctx.strokeStyle = '#eda100'; ctx.lineWidth = 1.6;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(bx - w / 2, by - 12, w, 24, 8); else ctx.rect(bx - w / 2, by - 12, w, 24);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(txt, bx, by + 0.5);
      ctx.restore();
    }
  }

  function caption(si, t) {
    const st = steps[si];
    switch (st.type) {
      case 'intro':
        return `<b>${cfg.mode === 'max' ? 'Max' : 'Average'} pooling</b> shrinks each feature map: a ${cfg.K}×${cfg.K} window keeps ${cfg.mode === 'max' ? 'only the strongest activation' : 'the average'} — no weights to learn.`;
      case 'slide':
        if (t < PH.move) return `Window to output position <b>(${st.or}, ${st.oc})</b> — stride ${cfg.S} means no overlap${cfg.S >= cfg.K ? '' : ' (here windows overlap)'}`;
        if (t < PH.pick) return cfg.mode === 'max'
          ? `Compare the ${cfg.K * cfg.K} values in the window — <b>keep the max</b>, per channel`
          : `Average the ${cfg.K * cfg.K} values in the window, per channel`;
        return `<b>out[c][${st.or}][${st.oc}]</b> written for all ${cfg.C} channels — channels never mix in pooling`;
      case 'done':
        return `Done — (${cfg.C}×${cfg.H}×${cfg.W}) → <b>(${cfg.C}×${Ho}×${Wo})</b>: ${cfg.S === cfg.K ? `spatial size ÷${cfg.K}` : 'downsampled'}, channels unchanged.`;
    }
    return '';
  }

  // ---------- detail ----------
  let dp = null;
  function buildDetail(el, si) {
    const st = steps[si];
    dp = null;
    if (st.type !== 'slide') {
      el.innerHTML = `<div class="dp-idle">${st.type === 'intro'
        ? 'Pooling adds translation tolerance and cuts computation — the exact position of a feature stops mattering.'
        : 'Height and width shrank; the channel count stayed the same. Pooling has zero parameters.'}</div>`;
      return;
    }
    let html = `<div class="dp-title">window at (<b>${st.or}</b>, <b>${st.oc}</b>) — ${cfg.mode} per channel</div>`;
    html += '<div class="dp-flow">';
    const picks = [];
    for (let c = 0; c < cfg.C; c++) {
      const { v, arg, win } = poolAt(c, st.or, st.oc);
      picks.push(v);
      html += `<div class="dp-ch"><span class="dp-ch-label">ch ${c}</span>`;
      html += `<div class="dp-grid" style="grid-template-columns:repeat(${cfg.K},27px)">`;
      for (let kr = 0; kr < cfg.K; kr++)
        for (let kc = 0; kc < cfg.K; kc++) {
          const xv = win[kr * cfg.K + kc];
          const isMax = cfg.mode === 'max' && kr === arg[0] && kc === arg[1];
          html += `<div class="dp-cell pw ${isMax ? 'is-max' : ''}" style="background:${U.seqColor(xv / 9)};color:${U.inkFor(U.seqColor(xv / 9))}">${xv}</div>`;
        }
      html += '</div>';
      html += `<span class="dp-op">→</span><span class="chip pick" style="font:650 13px var(--mono);border:1px solid var(--hairline);border-radius:7px;padding:3px 10px;background:var(--surface-3);color:#fff">…</span>`;
      html += '</div>';
    }
    html += '</div>';
    el.innerHTML = html;
    dp = { picks, chips: el.querySelectorAll('.chip.pick'), maxCells: el.querySelectorAll('.dp-cell.is-max'), all: el.querySelectorAll('.dp-cell.pw') };
  }

  function updateDetail(si, t) {
    const st = steps[si];
    if (!dp || st.type !== 'slide') return;
    const picked = t >= PH.pick - 0.15;
    dp.chips.forEach((ch, i) => { ch.textContent = picked ? U.fmt(dp.picks[i]) : '…'; });
    dp.maxCells.forEach(c => c.classList.toggle('hot', t >= PH.move + 0.1 && t < PH.write));
  }

  // ---------- controls ----------
  function init(controlsEl, legendEl) {
    controlsEl.innerHTML = '';
    const g1 = document.createElement('div');
    g1.className = 'ctl-group';
    g1.innerHTML = '<h3>Input</h3>';
    controlsEl.appendChild(g1);
    num(g1, 'channels', 'C', () => cfg.C, v => cfg.C = v, 1, 4);
    num(g1, 'size', 'H=W', () => cfg.H, v => { cfg.H = v; cfg.W = v; }, 4, 8);
    const g2 = document.createElement('div');
    g2.className = 'ctl-group';
    g2.innerHTML = '<h3>Pooling</h3>';
    controlsEl.appendChild(g2);
    num(g2, 'window', 'K', () => cfg.K, v => cfg.K = v, 2, 3);
    num(g2, 'stride', 'S', () => cfg.S, v => cfg.S = v, 1, 3);

    const row = document.createElement('div');
    row.className = 'ctl-row';
    row.innerHTML = `<label>mode</label><select class="ctl-select"><option value="max">max</option><option value="avg">avg</option></select>`;
    row.querySelector('select').onchange = e => { cfg.mode = e.target.value; regen(); App.resetTimeline(); };
    g2.appendChild(row);

    const btn = document.createElement('button');
    btn.className = 'btn';
    btn.textContent = '⚄ Regenerate values';
    btn.onclick = () => { seed = (seed * 16807 + 3) % 2147483647; regen(); App.resetTimeline(); };
    controlsEl.appendChild(btn);

    legendEl.innerHTML = '';
  }

  function num(group, label, dim, get, set, lo, hi) {
    const row = document.createElement('div');
    row.className = 'ctl-row';
    row.innerHTML = `<label>${label} <span class="dim">${dim}</span></label>` +
      `<span class="ctl-num"><button data-d="-1">−</button><span class="val">${get()}</span><button data-d="1">+</button></span>`;
    const val = row.querySelector('.val');
    row.querySelectorAll('button').forEach(b => b.onclick = () => {
      set(U.clamp(get() + Number(b.dataset.d), lo, hi));
      cfg.K = Math.min(cfg.K, cfg.H);
      regen(); val.textContent = get(); App.resetTimeline();
    });
    group.appendChild(row);
  }

  regen();

  return {
    id: 'pool',
    title: 'Max Pooling',
    desc: 'A window slides over each channel independently and keeps only the largest value (or the average) — shrinking height and width while keeping the channel count.',
    VW, VH,
    get steps() { return steps; },
    init, regen, render, caption, buildDetail, updateDetail
  };
})();
