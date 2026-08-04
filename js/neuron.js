'use strict';
/* ============ Lesson 1: a single neuron ============
 * The atom of every network: multiply each input by a weight, add them up,
 * add a bias, squash through an activation.
 */
const NeuronModule = (() => {

  const VW = 1280, VH = 600;
  const cfg = { n: 2, x: [1.5, -0.8, 0.6], w: [0.8, -1.2, 0.5], b: 0.4, act: 'relu' };

  const ACT = {
    relu:    { f: z => Math.max(0, z),           name: 'ReLU',    tex: 'max(0, z)' },
    sigmoid: { f: z => 1 / (1 + Math.exp(-z)),   name: 'sigmoid', tex: '1 / (1 + e⁻ᶻ)' },
    tanh:    { f: z => Math.tanh(z),             name: 'tanh',    tex: 'tanh(z)' },
    step:    { f: z => z >= 0 ? 1 : 0,           name: 'step',    tex: '1 if z ≥ 0, else 0' },
    linear:  { f: z => z,                        name: 'linear',  tex: 'z  (no squashing)' },
  };

  const steps = [
    { type: 'intro', dur: 2.6 },
    { type: 'mult',  dur: 3.4 },
    { type: 'sum',   dur: 2.2 },
    { type: 'bias',  dur: 2.0 },
    { type: 'act',   dur: 3.0 },
    { type: 'done',  dur: 2.6 },
  ];

  // ---------------- math ----------------
  const prod = i => cfg.x[i] * cfg.w[i];
  const sumProd = () => { let s = 0; for (let i = 0; i < cfg.n; i++) s += prod(i); return s; };
  const zVal = () => sumProd() + cfg.b;
  const aVal = () => ACT[cfg.act].f(zVal());

  // ---------------- layout ----------------
  const IN_X = 152, NEU_X = 500, OUT_X = 858, CY = 236;
  const NR = 48, OR = 28;
  const CHIP_Y = 404;
  const PANEL = { x: 946, w: 292 };
  const CURVE = { x: 946, y: 54, w: 292, h: 196 };
  const PLANE = { x: 946, y: 300, w: 292, h: 200 };

  function inPos(i) {
    const gap = 96;
    return [IN_X, CY - gap * (cfg.n - 1) / 2 + i * gap];
  }

  // ---------------- render ----------------
  function render(ctx, si, t) {
    const st = steps[si];
    const stage = si;            // 0 intro .. 5 done

    drawWires(ctx, si, t);
    drawInputs(ctx, si, t);
    drawNeuron(ctx, si, t);
    drawOutput(ctx, si, t);
    drawChips(ctx, si, t);
    drawCurve(ctx, si, t);
    if (cfg.n === 2) drawPlane(ctx, si, t);

    txt(ctx, 48, 34, 'a single neuron', { size: 15, weight: 650, color: '#fff' });
    txt(ctx, 48, 54, 'every network in this app is made of these', { size: 11.5 });
  }

  // how far the "mult" animation has progressed: index + fraction
  function multPhase(si, t) {
    if (si < 1) return -1;
    if (si > 1) return cfg.n;
    return U.clamp(t / 0.88, 0, 1) * cfg.n;
  }

  function drawWires(ctx, si, t) {
    const ph = multPhase(si, t);
    for (let i = 0; i < cfg.n; i++) {
      const [x0, y0] = inPos(i);
      const done = ph >= i + 1, active = ph >= i && ph < i + 1;
      const w = cfg.w[i];
      const col = U.divColor(w, 2);
      ctx.save();
      ctx.globalAlpha = si === 0 ? 0.5 : done || active ? 0.95 : 0.35;
      ctx.strokeStyle = col;
      ctx.lineWidth = 1 + 2.6 * Math.min(1, Math.abs(w) / 2);
      ctx.beginPath();
      ctx.moveTo(x0 + 26, y0);
      ctx.lineTo(NEU_X - NR, CY);
      ctx.stroke();
      ctx.restore();

      // weight label on the wire
      const mx = (x0 + 26 + NEU_X - NR) / 2, my = (y0 + CY) / 2;
      chip(ctx, mx, my - 12, `w${sub(i + 1)} = ${fmt(w)}`, col, si >= 1 ? 1 : 0.6, 10);

      // traveling dot while this input is being multiplied
      if (active) {
        const u = U.easeInOut(ph - i);
        ctx.save();
        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.arc(U.lerp(x0 + 26, NEU_X - NR, u), U.lerp(y0, CY, u), 4, 0, 7);
        ctx.fill();
        ctx.restore();
      }
    }
  }

  function drawInputs(ctx, si, t) {
    for (let i = 0; i < cfg.n; i++) {
      const [x, y] = inPos(i);
      const v = cfg.x[i];
      const fill = U.divColor(v, 3);
      ctx.save();
      ctx.beginPath(); ctx.arc(x, y, 26, 0, 7);
      ctx.fillStyle = fill; ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 1.2; ctx.stroke();
      ctx.fillStyle = U.inkFor(fill);
      ctx.font = '650 13px ui-monospace, Menlo, monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(fmt(v), x, y + 0.5);
      ctx.restore();
      txt(ctx, x - 40, y, `x${sub(i + 1)}`, { align: 'right', mono: true, size: 12, color: '#c3c2b7' });
    }
    txt(ctx, IN_X, inPos(cfg.n - 1)[1] + 52, 'inputs', { align: 'center', size: 11, weight: 600 });
  }

  function drawNeuron(ctx, si, t) {
    const active = si >= 2;
    const zshow = si >= 3 && !(si === 3 && t < 0.45);
    ctx.save();
    ctx.beginPath(); ctx.arc(NEU_X, CY, NR, 0, 7);
    ctx.fillStyle = '#20242b';
    ctx.strokeStyle = si >= 1 ? '#3987e5' : 'rgba(255,255,255,0.2)';
    ctx.lineWidth = si >= 1 ? 2 : 1.2;
    ctx.fill(); ctx.stroke();
    ctx.restore();
    txt(ctx, NEU_X, CY - 14, 'Σ', { align: 'center', size: 22, color: active ? '#fff' : '#898781' });
    // running value inside the neuron
    let inner = '';
    if (si === 1) inner = fmt(partialSum(si, t));
    else if (si === 2) inner = fmt(sumProd());
    else if (si >= 3) inner = fmt(zVal());
    if (inner) txt(ctx, NEU_X, CY + 16, inner, { align: 'center', mono: true, size: 13, color: '#fff', weight: 650 });
    txt(ctx, NEU_X, CY + NR + 16, 'weighted sum + bias', { align: 'center', size: 11, weight: 600 });

    // bias entering from below
    const bcol = U.divColor(cfg.b, 2);
    const bShow = si >= 3;
    ctx.save();
    ctx.globalAlpha = bShow ? 1 : 0.4;
    ctx.strokeStyle = bcol; ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(NEU_X, CY + NR + 44); ctx.lineTo(NEU_X, CY + NR + 4);
    ctx.stroke();
    ctx.restore();
    chip(ctx, NEU_X, CY + NR + 56, `b = ${fmt(cfg.b)}`, bcol, bShow ? 1 : 0.5, 11);

    // z → activation box
    const abox = { x: NEU_X + NR + 34, y: CY - 30, w: 128, h: 60 };
    const aActive = si >= 4;
    ctx.save();
    ctx.strokeStyle = aActive ? '#eda100' : 'rgba(255,255,255,0.18)';
    ctx.fillStyle = '#1e1e1d';
    ctx.lineWidth = aActive ? 1.8 : 1.1;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(abox.x, abox.y, abox.w, abox.h, 9); else ctx.rect(abox.x, abox.y, abox.w, abox.h);
    ctx.fill(); ctx.stroke();
    ctx.restore();
    txt(ctx, abox.x + abox.w / 2, abox.y + 20, ACT[cfg.act].name,
        { align: 'center', weight: 650, size: 13, color: aActive ? '#eda100' : '#c3c2b7' });
    txt(ctx, abox.x + abox.w / 2, abox.y + 40, ACT[cfg.act].tex,
        { align: 'center', mono: true, size: 10 });
    // arrows
    arrow(ctx, NEU_X + NR + 4, CY, abox.x - 4, CY, si >= 4 ? '#eda100' : 'rgba(137,135,129,0.6)');
    arrow(ctx, abox.x + abox.w + 4, CY, OUT_X - OR - 4, CY, si >= 5 ? '#eda100' : 'rgba(137,135,129,0.6)');
    if (si >= 3) chip(ctx, (NEU_X + NR + abox.x) / 2, CY - 26, `z = ${fmt(zVal())}`, '#3987e5', 1, 10);
  }

  function partialSum(si, t) {
    const ph = multPhase(si, t);
    let s = 0;
    for (let i = 0; i < cfg.n; i++) if (ph >= i + 1) s += prod(i);
    return s;
  }

  function drawOutput(ctx, si, t) {
    const show = si >= 5 || (si === 4 && t > 0.7);
    const a = aVal();
    const fill = show ? U.divColor(a, Math.max(1, Math.abs(a))) : '#20242b';
    ctx.save();
    ctx.beginPath(); ctx.arc(OUT_X, CY, OR, 0, 7);
    ctx.fillStyle = fill; ctx.fill();
    ctx.strokeStyle = show ? '#eda100' : 'rgba(255,255,255,0.2)';
    ctx.lineWidth = show ? 2.2 : 1.2;
    ctx.stroke();
    if (show) {
      ctx.fillStyle = U.inkFor(fill);
      ctx.font = '650 13px ui-monospace, Menlo, monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(fmt(a), OUT_X, CY + 0.5);
    }
    ctx.restore();
    txt(ctx, OUT_X, CY + OR + 18, 'output  a', { align: 'center', size: 11, weight: 600,
        color: show ? '#c3c2b7' : '#63615c' });
  }

  // the products lined up under the wires
  function drawChips(ctx, si, t) {
    if (si < 1) return;
    const ph = multPhase(si, t);
    let x = 150;
    txt(ctx, x - 8, CHIP_Y - 24, 'the arithmetic', { size: 11, weight: 600, color: '#c3c2b7' });
    for (let i = 0; i < cfg.n; i++) {
      const vis = ph >= i + 1;
      if (i > 0) { txt(ctx, x - 14, CHIP_Y, '+', { align: 'center', size: 15, color: vis ? '#c3c2b7' : '#3a3a38' }); }
      const label = `${fmt(cfg.w[i])} × ${fmt(cfg.x[i])} = ${fmt(prod(i))}`;
      chip(ctx, x + 62, CHIP_Y, label, U.divColor(prod(i), 3), vis ? 1 : 0.18, 11.5);
      x += 148;
    }
    if (si >= 3) {
      txt(ctx, x - 14, CHIP_Y, '+', { align: 'center', size: 15, color: '#c3c2b7' });
      chip(ctx, x + 34, CHIP_Y, `b = ${fmt(cfg.b)}`, U.divColor(cfg.b, 2), 1, 11.5);
      x += 96;
      txt(ctx, x - 6, CHIP_Y, '=', { align: 'center', size: 15, color: '#c3c2b7' });
      chip(ctx, x + 44, CHIP_Y, `z = ${fmt(zVal())}`, '#2a4a74', 1, 12);
    } else if (si === 2) {
      txt(ctx, x - 6, CHIP_Y, '=', { align: 'center', size: 15, color: '#c3c2b7' });
      chip(ctx, x + 40, CHIP_Y, fmt(sumProd()), '#2a4a74', U.easeOut(t), 12);
    }
  }

  // activation curve with the current z marked
  function drawCurve(ctx, si, t) {
    const C = CURVE;
    panel(ctx, C.x, C.y, C.w, C.h, si >= 4 ? '#eda100' : null);
    txt(ctx, C.x + 10, C.y + 14, `${ACT[cfg.act].name} — what the neuron does to z`,
        { size: 10.5, weight: 600, color: '#c3c2b7' });

    const zlo = -4, zhi = 4;
    const vals = [];
    for (let i = 0; i <= 120; i++) vals.push(ACT[cfg.act].f(zlo + (zhi - zlo) * i / 120));
    let alo = Math.min(...vals), ahi = Math.max(...vals);
    if (ahi - alo < 0.6) { ahi += 0.3; alo -= 0.3; }
    const pad = (ahi - alo) * 0.15;
    alo -= pad; ahi += pad;
    const px = z => C.x + 24 + (z - zlo) / (zhi - zlo) * (C.w - 40);
    const py = a => C.y + C.h - 22 - (a - alo) / (ahi - alo) * (C.h - 52);

    ctx.save();
    // axes
    ctx.strokeStyle = 'rgba(255,255,255,0.13)'; ctx.lineWidth = 1;
    ctx.beginPath();
    if (alo < 0 && ahi > 0) { ctx.moveTo(px(zlo), py(0)); ctx.lineTo(px(zhi), py(0)); }
    ctx.moveTo(px(0), C.y + 26); ctx.lineTo(px(0), C.y + C.h - 18);
    ctx.stroke();
    // curve
    ctx.strokeStyle = '#3987e5'; ctx.lineWidth = 2; ctx.lineJoin = 'round';
    ctx.beginPath();
    vals.forEach((a, i) => {
      const z = zlo + (zhi - zlo) * i / 120;
      i ? ctx.lineTo(px(z), py(a)) : ctx.moveTo(px(z), py(a));
    });
    ctx.stroke();
    ctx.restore();

    if (si >= 3) {
      const z = U.clamp(zVal(), zlo, zhi), a = ACT[cfg.act].f(zVal());
      const ay = py(U.clamp(a, alo, ahi));
      ctx.save();
      ctx.strokeStyle = U.withAlpha('#eda100', 0.6);
      ctx.setLineDash([3, 3]); ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(px(z), C.y + C.h - 18); ctx.lineTo(px(z), ay); ctx.lineTo(px(zlo), ay);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#eda100';
      ctx.beginPath(); ctx.arc(px(z), ay, 4.5, 0, 7); ctx.fill();
      ctx.restore();
      txt(ctx, px(z), C.y + C.h - 8, `z=${fmt(zVal())}`, { align: 'center', mono: true, size: 9.5, color: '#eda100' });
      txt(ctx, C.x + 22, ay - 9, `a=${fmt(a)}`, { mono: true, size: 9.5, color: '#eda100' });
    }
  }

  // what this neuron does to the whole input plane (2 inputs only)
  function drawPlane(ctx, si, t) {
    const P = PLANE;
    panel(ctx, P.x, P.y, P.w, P.h, si >= 5 ? '#3987e5' : null);
    txt(ctx, P.x + 10, P.y + 14, 'the line it draws through the inputs',
        { size: 10.5, weight: 600, color: '#c3c2b7' });
    const lo = -3, hi = 3;
    const sx = v => P.x + 20 + (v - lo) / (hi - lo) * (P.w - 36);
    const sy = v => P.y + P.h - 20 - (v - lo) / (hi - lo) * (P.h - 50);

    ctx.save();
    ctx.beginPath();
    ctx.rect(P.x + 20, P.y + 26, P.w - 36, P.h - 46);
    ctx.clip();
    // shade the two sides of w·x + b = 0
    const [w1, w2] = cfg.w;
    for (let gy = 0; gy < 24; gy++)
      for (let gx = 0; gx < 24; gx++) {
        const vx = lo + (hi - lo) * (gx + 0.5) / 24;
        const vy = lo + (hi - lo) * (gy + 0.5) / 24;
        const z = w1 * vx + w2 * vy + cfg.b;
        ctx.fillStyle = U.withAlpha(z >= 0 ? '#3987e5' : '#d95926', 0.10 + 0.20 * Math.min(1, Math.abs(z) / 3));
        ctx.fillRect(sx(vx) - (P.w - 36) / 48, sy(vy) - (P.h - 46) / 48, (P.w - 36) / 24 + 1, (P.h - 46) / 24 + 1);
      }
    // the boundary line itself
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.8;
    ctx.beginPath();
    if (Math.abs(w2) > 1e-6) {
      ctx.moveTo(sx(lo), sy(-(w1 * lo + cfg.b) / w2));
      ctx.lineTo(sx(hi), sy(-(w1 * hi + cfg.b) / w2));
    } else if (Math.abs(w1) > 1e-6) {
      ctx.moveTo(sx(-cfg.b / w1), sy(lo)); ctx.lineTo(sx(-cfg.b / w1), sy(hi));
    }
    ctx.stroke();
    // the current input point
    ctx.fillStyle = '#eda100';
    ctx.beginPath(); ctx.arc(sx(cfg.x[0]), sy(cfg.x[1]), 5, 0, 7); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.4; ctx.stroke();
    ctx.restore();
    txt(ctx, P.x + P.w - 10, P.y + P.h - 6, 'z > 0 blue · z < 0 orange',
        { align: 'right', size: 9.5 });
  }

  // ---------------- small draw helpers ----------------
  function panel(ctx, x, y, w, h, accent) {
    ctx.save();
    ctx.fillStyle = '#161616';
    ctx.strokeStyle = accent ? U.withAlpha(accent, 0.7) : 'rgba(255,255,255,0.10)';
    ctx.lineWidth = accent ? 1.6 : 1;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, 8); else ctx.rect(x, y, w, h);
    ctx.fill(); ctx.stroke();
    ctx.restore();
  }
  function chip(ctx, cx, cy, label, color, alpha, size) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.font = `600 ${size}px ui-monospace, Menlo, monospace`;
    const w = ctx.measureText(label).width + 16;
    ctx.fillStyle = '#15171a';
    ctx.strokeStyle = U.withAlpha(color, 0.9);
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(cx - w / 2, cy - 11, w, 22, 7); else ctx.rect(cx - w / 2, cy - 11, w, 22);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#e8e7e2';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(label, cx, cy + 0.5);
    ctx.restore();
  }
  function arrow(ctx, x0, y0, x1, y1, col) {
    ctx.save();
    ctx.strokeStyle = ctx.fillStyle = col;
    ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - 5, y1); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x1, y1); ctx.lineTo(x1 - 6, y1 - 4); ctx.lineTo(x1 - 6, y1 + 4);
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
  const fmt = v => (Math.round(v * 100) / 100).toFixed(2).replace(/\.00$/, '');
  const sub = n => '₀₁₂₃₄₅₆₇₈₉'[n] || String(n);

  // ---------------- caption ----------------
  function caption(si, t) {
    const st = steps[si];
    switch (st.type) {
      case 'intro':
        return `A neuron takes a few numbers in and puts <b>one number</b> out. Each input travels a wire that has a <b>weight</b> — how much that input matters.`;
      case 'mult': {
        const ph = multPhase(si, t);
        const i = U.clamp(Math.floor(ph), 0, cfg.n - 1);
        return `Multiply each input by its weight: <b>${fmt(cfg.w[i])} × ${fmt(cfg.x[i])} = ${fmt(prod(i))}</b>. A big weight makes that input count for more; a negative weight counts <i>against</i>.`;
      }
      case 'sum':
        return `Add the products together → <b>${fmt(sumProd())}</b>. That is the entire "learning" part: a weighted vote over the inputs.`;
      case 'bias':
        return `Add the <b>bias</b> (${fmt(cfg.b)}) — a number the neuron can shift by regardless of input. Now <b>z = ${fmt(zVal())}</b>.`;
      case 'act':
        return `Squash z through <b>${ACT[cfg.act].name}</b> → <b>a = ${fmt(aVal())}</b>. Without this, stacking neurons would be pointless — see the curve on the right.`;
      case 'done':
        return cfg.n === 2
          ? `Output <b>${fmt(aVal())}</b>. Notice the panel below-right: one neuron can only ever split its inputs with a <b>straight line</b>. That limit is exactly why we stack them — next lesson.`
          : `Output <b>${fmt(aVal())}</b>. Put many neurons side by side, all reading the same inputs, and you have a <b>layer</b> — the next lesson.`;
    }
    return '';
  }

  // ---------------- detail ----------------
  function buildDetail(el, si) {
    const st = steps[si];
    const terms = [];
    for (let i = 0; i < cfg.n; i++)
      terms.push(`<span class="term ${st.type === 'mult' ? '' : ''}">${fmt(cfg.w[i])}·${fmt(cfg.x[i])}</span>`);
    let h = `<div class="dp-title">one neuron, start to finish</div><div class="dp-eq">`;
    h += `z = ${terms.join(' + ')} + ${fmt(cfg.b)}`;
    if (si >= 2) h += ` = <b>${fmt(zVal())}</b>`;
    h += `<br>a = ${ACT[cfg.act].name}(z)`;
    if (si >= 4) h += ` = ${ACT[cfg.act].tex.replace('z', fmt(zVal()))} = <b>${fmt(aVal())}</b>`;
    h += `</div>`;
    if (si >= 5)
      h += `<div class="dp-note">Try it: drag the weight sliders and watch the output — and the boundary line — move. ` +
           `Set a weight to 0 and that input stops mattering entirely.</div>`;
    else
      h += `<div class="dp-note">Every slider on the left is live — change a weight, a bias, or an input and the whole picture updates.</div>`;
    el.innerHTML = h;
  }
  function updateDetail() {}

  // ---------------- controls ----------------
  function init(controlsEl, legendEl) {
    controlsEl.innerHTML = '';
    const g0 = grp(controlsEl, 'Inputs');
    stepper(g0, 'how many', () => cfg.n, v => cfg.n = v, [1, 2, 3], true);
    for (let i = 0; i < cfg.n; i++)
      slider(g0, `x${sub(i + 1)}`, -3, 3, 0.1, () => cfg.x[i], v => cfg.x[i] = v);

    const g1 = grp(controlsEl, 'Weights & bias');
    for (let i = 0; i < cfg.n; i++)
      slider(g1, `w${sub(i + 1)}`, -2, 2, 0.1, () => cfg.w[i], v => cfg.w[i] = v);
    slider(g1, 'bias b', -2, 2, 0.1, () => cfg.b, v => cfg.b = v);

    const g2 = grp(controlsEl, 'Activation');
    const row = document.createElement('div');
    row.className = 'ctl-row';
    row.innerHTML = `<label>function</label><select class="ctl-select">` +
      Object.keys(ACT).map(k => `<option value="${k}">${ACT[k].name}</option>`).join('') + `</select>`;
    row.querySelector('select').value = cfg.act;
    row.querySelector('select').onchange = e => { cfg.act = e.target.value; App.resetTimeline(); };
    g2.appendChild(row);

    legendEl.innerHTML =
      `<div class="legend-row"><span>weight / value</span></div>` +
      (() => { const s = []; for (let i = 0; i <= 10; i++) s.push(U.divColor(U.lerp(-1, 1, i / 10), 1));
        return `<div class="legend-ramp" style="background:linear-gradient(90deg,${s.join(',')})"></div>` +
               `<div class="legend-cap"><span>negative</span><span>positive</span></div>`; })();
  }

  function grp(parent, title) {
    const g = document.createElement('div');
    g.className = 'ctl-group';
    g.innerHTML = `<h3>${title}</h3>`;
    parent.appendChild(g);
    return g;
  }
  function slider(g, label, lo, hi, stepv, get, set) {
    const row = document.createElement('div');
    row.className = 'ctl-slider';
    row.innerHTML = `<div class="cs-top"><span class="cs-label">${label}</span><span class="cs-val">${fmt(get())}</span></div>` +
                    `<input type="range" min="${lo}" max="${hi}" step="${stepv}" value="${get()}">`;
    const val = row.querySelector('.cs-val');
    row.querySelector('input').oninput = e => { set(Number(e.target.value)); val.textContent = fmt(get()); };
    g.appendChild(row);
  }
  function stepper(g, label, get, set, list, rebuild) {
    const row = document.createElement('div');
    row.className = 'ctl-row';
    row.innerHTML = `<label>${label}</label>` +
      `<span class="ctl-num"><button data-d="-1">−</button><span class="val">${get()}</span><button data-d="1">+</button></span>`;
    row.querySelectorAll('button').forEach(b => b.onclick = () => {
      const i = list.indexOf(get());
      const v = list[U.clamp(i + Number(b.dataset.d), 0, list.length - 1)];
      if (v === get()) return;
      set(v);
      if (rebuild) init(document.getElementById('controls'), document.getElementById('legend'));
      App.resetTimeline();
    });
    g.appendChild(row);
  }

  return {
    id: 'neuron',
    title: 'A Single Neuron',
    desc: 'The smallest piece of a neural network: multiply each input by a weight, add them up, add a bias, and squash the result. Every slider is live.',
    VW, VH,
    get steps() { return steps; },
    init, regen: () => {}, render, caption, buildDetail, updateDetail
  };
})();
