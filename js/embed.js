'use strict';
/* ============ Lesson: Tokens & Embeddings ============
 * text → tokens → integer IDs → a lookup in the embedding table →
 * a flat horizontal row of tokens expanded into a (T × d) tensor.
 * This is the thing every attention lesson starts from.
 */
const EmbedModule = (() => {

  const VW = 1280, VH = 620;
  const D = 6;                                  // embedding dimension
  const DIMS = ['living', 'action', 'object', 'size', 'feeling', 'grammar'];

  // A tiny vocabulary, including subword pieces (## = "continues the word
  // before me"), which is how real tokenizers handle words they don't know.
  const VOCAB = [
    '[?]', 'the', 'a', 'an', 'and', 'is', 'are', 'was',
    'cat', 'dog', 'robot', 'apple', 'book', 'tree',
    'run', 'ran', 'eat', 'ate', 'love', 'see',
    'big', 'small', 'red', 'happy', 'sad',
    'i', 'it', 'they',
    '##s', '##ing', '##ed', '##d', '##ly', '##ning',
  ];

  // Hand-built so the dimensions mean something you can read. In a real model
  // these are LEARNED and individual dimensions are not interpretable — the
  // captions say so.
  const EMB = {
    '[?]':   [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    'the':   [ 0.0,  0.0,  0.1,  0.0,  0.0,  0.9],
    'a':     [ 0.0,  0.0,  0.1, -0.1,  0.0,  0.9],
    'an':    [ 0.0,  0.0,  0.1, -0.1,  0.0,  0.9],
    'and':   [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.8],
    'is':    [ 0.0,  0.4,  0.0,  0.0,  0.0,  0.7],
    'are':   [ 0.0,  0.4,  0.0,  0.0,  0.0,  0.7],
    'was':   [ 0.0,  0.4,  0.0,  0.0,  0.0,  0.5],
    'cat':   [ 0.9,  0.0,  0.5, -0.3,  0.4, -0.2],
    'dog':   [ 0.9,  0.0,  0.5,  0.0,  0.5, -0.2],
    'robot': [ 0.2,  0.1,  0.8,  0.2,  0.0, -0.2],
    'apple': [ 0.3,  0.0,  0.9, -0.4,  0.3, -0.2],
    'book':  [ 0.0,  0.0,  0.9, -0.2,  0.2, -0.2],
    'tree':  [ 0.6,  0.0,  0.7,  0.7,  0.2, -0.2],
    'run':   [ 0.1,  0.9,  0.0,  0.0,  0.2, -0.1],
    'ran':   [ 0.1,  0.9,  0.0,  0.0,  0.2, -0.4],
    'eat':   [ 0.1,  0.9,  0.1,  0.0,  0.2, -0.1],
    'ate':   [ 0.1,  0.9,  0.1,  0.0,  0.2, -0.4],
    'love':  [ 0.1,  0.7,  0.0,  0.0,  0.9, -0.1],
    'see':   [ 0.1,  0.8,  0.0,  0.0,  0.1, -0.1],
    'big':   [ 0.0,  0.0,  0.1,  0.9,  0.1,  0.3],
    'small': [ 0.0,  0.0,  0.1, -0.9,  0.1,  0.3],
    'red':   [ 0.0,  0.0,  0.2,  0.0,  0.1,  0.3],
    'happy': [ 0.2,  0.0,  0.0,  0.0,  0.9,  0.3],
    'sad':   [ 0.2,  0.0,  0.0,  0.0, -0.9,  0.3],
    'i':     [ 0.8,  0.0,  0.2,  0.0,  0.1,  0.5],
    'it':    [ 0.1,  0.0,  0.4,  0.0,  0.0,  0.5],
    'they':  [ 0.7,  0.0,  0.2,  0.0,  0.1,  0.6],
    '##s':   [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.6],
    '##ing': [ 0.0,  0.5,  0.0,  0.0,  0.0,  0.4],
    '##ed':  [ 0.0,  0.3,  0.0,  0.0,  0.0, -0.5],
    '##d':   [ 0.0,  0.3,  0.0,  0.0,  0.0, -0.5],
    '##ly':  [ 0.0,  0.2,  0.0,  0.0,  0.0,  0.4],
    '##ning':[ 0.0,  0.5,  0.0,  0.0,  0.0,  0.4],
  };

  const SENTENCES = [
    'the robot ate a red apple',
    'the cats are running',
    'a happy dog sees the tree',
    'the zebra is big',
  ];

  const cfg = { text: SENTENCES[0] };
  let toks = [];                                  // [{t, id, from, kind}]

  // ---------------- tokenizer ----------------
  // greedy: whole word if we know it, else longest known prefix + "##rest"
  function tokenize(text) {
    const words = (text.toLowerCase().match(/[a-z]+/g) || []).slice(0, 8);
    const out = [];
    for (const w of words) {
      if (VOCAB.includes(w)) { out.push({ t: w, from: w, kind: 'word' }); continue; }
      let split = false;
      for (let L = w.length - 1; L >= 2 && !split; L--) {
        const head = w.slice(0, L), tail = '##' + w.slice(L);
        if (VOCAB.includes(head) && VOCAB.includes(tail)) {
          out.push({ t: head, from: w, kind: 'piece1' },
                   { t: tail, from: w, kind: 'piece2' });
          split = true;
        }
      }
      if (!split) out.push({ t: '[?]', from: w, kind: 'unk' });
    }
    for (const o of out) o.id = VOCAB.indexOf(o.t);
    return out;
  }

  function rebuild() { toks = tokenize(cfg.text); }

  const vec = tk => EMB[tk] || EMB['[?]'];
  function cosine(a, b) {
    let d = 0, na = 0, nb = 0;
    for (let i = 0; i < D; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return d / Math.max(1e-9, Math.sqrt(na) * Math.sqrt(nb));
  }

  const steps = [
    { type: 'text',    dur: 2.6 },
    { type: 'split',   dur: 3.4 },
    { type: 'ids',     dur: 3.0 },
    { type: 'lookup',  dur: 4.4 },
    { type: 'expand',  dur: 3.2 },
    { type: 'tensor',  dur: 2.6 },
    { type: 'similar', dur: 3.4 },
    { type: 'done',    dur: 2.6 },
  ];
  const idxOf = ty => steps.findIndex(s => s.type === ty);

  // ---------------- layout ----------------
  const TEXT_Y = 92, CHIP_Y = 148, ID_Y = 208;
  const TENSOR = { x: 44, top: 268, w: 400, h: 292, lab: 56 };   // lab = room for dim names
  const TABLE = { x: 470, y: 268, w: 330, h: 292 };
  const SIM = { x: 836, y: 268, w: 404, h: 292 };

  const chipW = () => Math.min(104, (VW - 120) / Math.max(1, toks.length));
  const chipX = i => 44 + i * (chipW() + 8) + chipW() / 2;

  // ---------------- render ----------------
  function render(ctx, si, t) {
    const st = steps[si];
    drawText(ctx, si, t);
    drawChips(ctx, si, t);
    drawIds(ctx, si, t);
    drawTensor(ctx, si, t);
    drawTable(ctx, si, t);
    drawSim(ctx, si, t);

    txt(ctx, 44, 40, 'tokens & embeddings — how text becomes numbers',
        { size: 15, weight: 650, color: '#fff' });
    txt(ctx, 44, 60, `vocabulary of ${VOCAB.length} tokens · embedding dimension d = ${D}`,
        { size: 11.5 });
  }

  // step 1: the raw string
  function drawText(ctx, si, t) {
    const alive = si === idxOf('text');
    const n = alive ? Math.floor(U.clamp(t / 0.55, 0, 1) * cfg.text.length) : cfg.text.length;
    txt(ctx, 44, TEXT_Y - 20, 'what you type', { size: 9.5, weight: 600, color: '#63615c' });
    txt(ctx, 44, TEXT_Y, `"${cfg.text.slice(0, n)}${alive && n < cfg.text.length ? '▌' : '"'}`,
        { size: 21, weight: 600, color: si === 0 ? '#fff' : '#c3c2b7', mono: true });
  }

  const kindColor = k => k === 'unk' ? '#e34948' : k === 'piece2' ? '#d95926' : '#3987e5';

  // step 2: tokens as chips, laid out horizontally
  function drawChips(ctx, si, t) {
    const s0 = idxOf('split');
    if (si < s0) return;
    const rev = si === s0 ? U.clamp(t / 0.8, 0, 1) * toks.length : toks.length;
    txt(ctx, 44, CHIP_Y - 26, `tokens (${toks.length})`, { size: 9.5, weight: 600, color: '#c3c2b7' });
    const w = chipW();
    for (let i = 0; i < toks.length; i++) {
      if (i >= rev) continue;
      const tk = toks[i], cx = chipX(i);
      const col = kindColor(tk.kind);
      const hot = activeTok(si, t) === i;
      ctx.save();
      ctx.globalAlpha = i + 1 > rev ? rev - i : 1;
      ctx.fillStyle = hot ? U.withAlpha(col, 0.3) : '#1e2228';
      ctx.strokeStyle = hot ? col : U.withAlpha(col, 0.6);
      ctx.lineWidth = hot ? 2 : 1.2;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(cx - w / 2, CHIP_Y - 15, w, 30, 8);
      else ctx.rect(cx - w / 2, CHIP_Y - 15, w, 30);
      ctx.fill(); ctx.stroke();
      ctx.restore();
      txt(ctx, cx, CHIP_Y, tk.t, { align: 'center', mono: true, size: 12.5,
          weight: 600, color: hot ? '#fff' : '#c3c2b7' });
      // show where a split came from
      if (tk.kind === 'piece1' || tk.kind === 'piece2' || tk.kind === 'unk')
        txt(ctx, cx, CHIP_Y + 26, tk.kind === 'unk' ? `"${tk.from}" not in vocab` : `from "${tk.from}"`,
            { align: 'center', size: 8.5, color: kindColor(tk.kind) });
    }
  }

  // step 3: integer IDs
  function drawIds(ctx, si, t) {
    const s0 = idxOf('ids');
    if (si < s0) return;
    const rev = si === s0 ? U.clamp(t / 0.8, 0, 1) * toks.length : toks.length;
    txt(ctx, 44, ID_Y - 22, 'token IDs — the model only ever sees these integers',
        { size: 9.5, weight: 600, color: '#c3c2b7' });
    const w = chipW();
    for (let i = 0; i < toks.length; i++) {
      if (i >= rev) continue;
      const cx = chipX(i);
      const hot = activeTok(si, t) === i;
      ctx.save();
      ctx.fillStyle = hot ? '#2a4a74' : '#16181c';
      ctx.strokeStyle = hot ? '#3987e5' : 'rgba(255,255,255,0.14)';
      ctx.lineWidth = hot ? 1.8 : 1;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(cx - 22, ID_Y - 12, 44, 24, 6);
      else ctx.rect(cx - 22, ID_Y - 12, 44, 24);
      ctx.fill(); ctx.stroke();
      ctx.restore();
      txt(ctx, cx, ID_Y, String(toks[i].id), { align: 'center', mono: true, size: 12.5,
          weight: 650, color: '#fff' });
    }
  }

  // which token is being looked up right now
  function activeTok(si, t) {
    if (steps[si].type !== 'lookup') return -1;
    return U.clamp(Math.floor(U.clamp(t / 0.9, 0, 1) * toks.length), 0, toks.length - 1);
  }

  // how far the flat row has morphed into a 3-D slab (0 = flat, 1 = isometric)
  function morph(si, t) {
    const e = idxOf('expand');
    if (si < e) return 0;
    if (si === e) return U.easeInOut(U.clamp(t / 0.85, 0, 1));
    return 1;
  }

  // a 2-D→3-D interpolating isometric projection
  function lerpIso(ctx, s, ox, oy, k) {
    const ux = s * U.lerp(1.0, 0.90, k);
    const uy = s * U.lerp(0.0, 0.45, k);
    const h = s * U.lerp(1.0, 0.92, k);
    return { ctx, s, ux, uy, h, ox, oy,
             p: (x, y, z) => [ox + (x - z) * ux, oy + (x + z) * uy - y * h] };
  }

  // steps 4-6: the (T × d) tensor
  function drawTensor(ctx, si, t) {
    const s0 = idxOf('lookup');
    if (si < s0) return;
    const T = toks.length;
    const k = morph(si, t);
    const s = Math.min((TENSOR.w - TENSOR.lab) / ((T + 1) * 0.9),
                       TENSOR.h / (D * 0.92 + (T + 1) * 0.45));
    const iso = lerpIso(ctx, s, TENSOR.x + TENSOR.lab + s * 0.9,
                        TENSOR.top + D * 0.92 * s, k);

    txt(ctx, TENSOR.x, TENSOR.top - 20,
        k > 0.05 ? `embeddings — a (${T} × ${D}) tensor` : 'embeddings, one row per token',
        { size: 10.5, weight: 600, color: '#c3c2b7' });

    const act = activeTok(si, t);
    const filled = i => si > s0 || (act >= 0 && i <= act);

    // dimension labels, hung off the far-left edge of the slab (z = 1)
    for (let d = 0; d < D; d++) {
      const [lx, ly] = iso.p(0, D - 1 - d + 0.5, 1);
      txt(ctx, lx - 7, ly, DIMS[d], { align: 'right', size: 8.5, color: '#63615c' });
    }

    for (let i = 0; i < T; i++) {
      if (!filled(i)) continue;
      const v = vec(toks[i].t);
      const isAct = i === act;
      // give the slab thickness as it becomes 3-D
      if (k > 0.02) {
        Iso.box(iso, i, 0, 0, 1, D, 1, '#1b1f26', { alpha: 0.9 });
      }
      for (let d = 0; d < D; d++) {
        const val = v[d];
        const col = U.divColor(val, 1);
        Iso.cell(iso, i, 0, 1, 0, d, D, col, s >= 26 ? val.toFixed(1) : null,
                 { alpha: 1 });
      }
      if (isAct) Iso.wire(iso, i, 0, 0, 1, D, 1, '#eda100', 2, 1);
      // token label under its column
      const [cx, cy] = iso.p(i + 0.5, 0, 1);
      txt(ctx, cx, cy + 14, toks[i].t, { align: 'center', mono: true, size: 9,
          color: isAct ? '#eda100' : '#898781' });
    }

    // connector from the chip row down into the tensor while looking up
    if (act >= 0) {
      const [tx, ty] = iso.p(act + 0.5, D, 0.5);
      ctx.save();
      ctx.strokeStyle = U.withAlpha('#eda100', 0.5);
      ctx.setLineDash([4, 3]); ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(chipX(act), ID_Y + 14);
      ctx.lineTo(tx, ty - 6);
      ctx.stroke();
      ctx.restore();
    }
  }

  // the embedding table, windowed around the token being looked up
  function drawTable(ctx, si, t) {
    const s0 = idxOf('lookup');
    if (si < s0) return;
    const B = TABLE;
    panel(ctx, B.x, B.y, B.w, B.h);
    txt(ctx, B.x, B.y - 20, `embedding table — ${VOCAB.length} rows × ${D}`,
        { size: 10.5, weight: 600, color: '#c3c2b7' });

    const act = activeTok(si, t);
    const centre = act >= 0 ? toks[act].id : (toks[0] ? toks[0].id : 0);
    const rows = 11, rowH = 23;
    let start = U.clamp(centre - (rows >> 1), 0, Math.max(0, VOCAB.length - rows));

    // header
    const cw = 26, cx0 = B.x + 118;
    for (let d = 0; d < D; d++)
      txt(ctx, cx0 + d * cw + cw / 2, B.y + 14, DIMS[d].slice(0, 3),
          { align: 'center', size: 8, color: '#5d5c58' });

    for (let r = 0; r < rows; r++) {
      const id = start + r;
      if (id >= VOCAB.length) break;
      const y = B.y + 28 + r * rowH;
      const tk = VOCAB[id];
      const hot = act >= 0 && id === toks[act].id;
      if (hot) {
        ctx.save();
        ctx.fillStyle = 'rgba(237,161,0,0.14)';
        ctx.strokeStyle = '#eda100'; ctx.lineWidth = 1.3;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(B.x + 6, y - 10, B.w - 12, rowH - 3, 5);
        else ctx.rect(B.x + 6, y - 10, B.w - 12, rowH - 3);
        ctx.fill(); ctx.stroke();
        ctx.restore();
      }
      txt(ctx, B.x + 30, y, String(id), { align: 'right', mono: true, size: 10,
          color: hot ? '#eda100' : '#5d5c58' });
      txt(ctx, B.x + 40, y, tk, { mono: true, size: 10.5,
          color: hot ? '#fff' : '#898781' });
      const v = vec(tk);
      for (let d = 0; d < D; d++) {
        const col = U.divColor(v[d], 1);
        ctx.save();
        ctx.globalAlpha = hot ? 1 : 0.55;
        ctx.fillStyle = col;
        ctx.fillRect(cx0 + d * cw, y - 7, cw - 2, 14);
        ctx.restore();
      }
    }
    txt(ctx, B.x + 8, B.y + B.h - 10,
        'every row is learned during training', { size: 9, color: '#5d5c58' });
  }

  // why embeddings beat raw IDs
  function drawSim(ctx, si, t) {
    const s0 = idxOf('similar');
    if (si < s0) return;
    const B = SIM;
    panel(ctx, B.x, B.y, B.w, B.h);
    txt(ctx, B.x, B.y - 20, 'why not just use the ID number?',
        { size: 10.5, weight: 600, color: '#c3c2b7' });

    const uniq = [...new Set(toks.map(o => o.t))];
    const pairs = [];
    for (let i = 0; i < uniq.length; i++)
      for (let j = i + 1; j < uniq.length; j++)
        pairs.push({ a: uniq[i], b: uniq[j], c: cosine(vec(uniq[i]), vec(uniq[j])) });
    pairs.sort((p, q) => q.c - p.c);
    const show = pairs.length >= 2 ? [pairs[0], pairs[pairs.length - 1]] : pairs;

    txt(ctx, B.x + 14, B.y + 22,
        'IDs are arbitrary labels — 11 is not "more" than 8.', { size: 11, color: '#c3c2b7' });
    txt(ctx, B.x + 14, B.y + 40,
        'Embeddings put related tokens near each other:', { size: 11, color: '#c3c2b7' });

    const a = U.clamp((t - 0.15) / 0.5, 0, 1);
    show.forEach((p, i) => {
      const y = B.y + 74 + i * 76;
      const good = p.c > 0.4;
      txt(ctx, B.x + 14, y, `"${p.a}"  vs  "${p.b}"`, { mono: true, size: 12, color: '#fff', weight: 600 });
      // similarity bar
      const bw = B.w - 130, bx = B.x + 14;
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.strokeRect(bx, y + 12, bw, 12);
      ctx.fillStyle = good ? '#0ca30c' : '#d95926';
      const frac = si === s0 ? a : 1;
      ctx.fillRect(bx, y + 12, bw * U.clamp((p.c + 1) / 2, 0, 1) * frac, 12);
      // zero mark
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.beginPath(); ctx.moveTo(bx + bw / 2, y + 10); ctx.lineTo(bx + bw / 2, y + 26); ctx.stroke();
      ctx.restore();
      txt(ctx, bx + bw + 12, y + 18, p.c.toFixed(2), { mono: true, size: 12, weight: 650,
          color: good ? '#0ca30c' : '#d95926' });
      txt(ctx, bx, y + 38, i === 0 ? 'most similar pair in this sentence' : 'least similar pair',
          { size: 9.5, color: '#5d5c58' });
    });
    txt(ctx, B.x + 14, B.y + B.h - 14, 'cosine similarity: −1 opposite · 0 unrelated · 1 identical',
        { size: 9, color: '#5d5c58' });
  }

  // ---------------- helpers ----------------
  function panel(ctx, x, y, w, h) {
    ctx.save();
    ctx.fillStyle = '#141414';
    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, 8); else ctx.rect(x, y, w, h);
    ctx.fill(); ctx.stroke();
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
  function caption(si, t) {
    const st = steps[si];
    const act = activeTok(si, t);
    switch (st.type) {
      case 'text':
        return `A model can't read letters. Everything below turns this sentence into <b>numbers</b> — that's the only thing a network ever consumes.`;
      case 'split': {
        const split = toks.some(o => o.kind === 'piece2');
        const unk = toks.some(o => o.kind === 'unk');
        return `<b>Tokenize</b>: chop the text into known pieces. ` +
          (split ? `Words the vocabulary doesn't have get split — <b>##</b> means "glued to the piece before me". ` : '') +
          (unk ? `A word with no usable split becomes <b>[?]</b>. ` : '') +
          `${toks.length} tokens here.`;
      }
      case 'ids':
        return `Each token is just a <b>row number</b> in the vocabulary. The sentence is now <b>[${toks.map(o => o.id).join(', ')}]</b> — and nothing else.`;
      case 'lookup': {
        if (act < 0) return '';
        const tk = toks[act];
        return `<b>Look up row ${tk.id}</b> ("${tk.t}") in the embedding table and copy out its <b>${D}</b> numbers. That vector <i>is</i> the token, as far as the model is concerned.`;
      }
      case 'expand':
        return `The flat row of tokens gains a second axis: every token becomes a <b>column of ${D} numbers</b>. A 1-D list of IDs has become a <b>2-D tensor</b> — shown here with thickness so you can see it as a block.`;
      case 'tensor':
        return `Shape <b>(${toks.length}, ${D})</b> — tokens × dimensions. Real models use d = 768 or more; the idea is identical, just wider.`;
      case 'similar':
        return `This is what IDs can't do. Token 8 and token 9 are unrelated as <i>numbers</i>, but their <b>vectors</b> can sit right next to each other.`;
      case 'done':
        return `That tensor is exactly what the <b>Attention</b> lesson starts from. One thing still missing: nothing here encodes <b>word order</b> — that's what positional encoding adds.`;
    }
    return '';
  }

  // ---------------- detail ----------------
  function chips(v) {
    return `<span style="display:inline-flex;gap:3px;vertical-align:middle">` +
      v.map((x, i) => {
        const bg = U.divColor(x, 1);
        return `<span title="${DIMS[i]}" style="min-width:42px;height:20px;border-radius:4px;background:${bg};color:${U.inkFor(bg)};font:600 10px var(--mono);display:flex;align-items:center;justify-content:center">${x.toFixed(1)}</span>`;
      }).join('') + `</span>`;
  }

  function buildDetail(el, si) {
    const st = steps[si];
    let h = '';
    if (st.type === 'text' || st.type === 'split') {
      h = `<div class="dp-title">tokenizing "${cfg.text}"</div>` +
          `<table class="dp-table"><thead><tr><th>#</th><th>token</th><th>id</th><th>came from</th></tr></thead><tbody>` +
          toks.map((o, i) => `<tr><td>${i}</td><td><b>${o.t}</b></td><td>${o.id}</td>` +
            `<td>${o.kind === 'unk' ? `"${o.from}" — not in vocabulary` :
                   o.kind === 'word' ? 'whole word' : `piece of "${o.from}"`}</td></tr>`).join('') +
          `</tbody></table>`;
    } else if (st.type === 'ids') {
      h = `<div class="dp-title">the whole sentence, as the model sees it</div>` +
          `<div class="dp-eq">[${toks.map(o => o.id).join(', ')}]<br>` +
          `<span style="color:var(--ink-muted)">${toks.length} integers. Every downstream layer — attention, MLP, everything — starts from this.</span></div>`;
    } else if (st.type === 'lookup' || st.type === 'expand' || st.type === 'tensor') {
      h = `<div class="dp-title">embedding lookup — table[id] for each token</div><div class="dp-eq" style="line-height:2.2">` +
          toks.map(o => `<b>${o.t}</b> <span style="color:var(--ink-muted)">(id ${o.id})</span> → ${chips(vec(o.t))}`).join('<br>') +
          `</div><div class="dp-note">These ${D} dimensions are hand-labelled (${DIMS.join(', ')}) so you can see what the numbers are doing. ` +
          `In a real model the embedding table is <b>learned</b> and individual dimensions have no tidy meaning — but the geometry does: related tokens end up close together.</div>`;
    } else if (st.type === 'similar') {
      const uniq = [...new Set(toks.map(o => o.t))];
      const rows = [];
      for (let i = 0; i < uniq.length; i++)
        for (let j = i + 1; j < uniq.length; j++)
          rows.push({ a: uniq[i], b: uniq[j], c: cosine(vec(uniq[i]), vec(uniq[j])) });
      rows.sort((p, q) => q.c - p.c);
      h = `<div class="dp-title">cosine similarity between every pair in this sentence</div>` +
          `<table class="dp-table"><thead><tr><th>a</th><th>b</th><th>similarity</th></tr></thead><tbody>` +
          rows.slice(0, 6).map(r => `<tr><td>${r.a}</td><td>${r.b}</td><td><b>${r.c.toFixed(3)}</b></td></tr>`).join('') +
          `</tbody></table>`;
    } else {
      h = `<div class="dp-title">what comes next</div><div class="dp-eq">` +
          `(${toks.length}, ${D}) tensor → <b>Attention</b><br>` +
          `<span style="color:var(--ink-muted)">Attention lets each of those ${toks.length} vectors look at the others and update itself with context. ` +
          `Right now "${toks[0] ? toks[0].t : 'the'}" has the same vector no matter what sentence it appears in — after attention, it doesn't.</span></div>`;
    }
    el.innerHTML = h;
  }
  function updateDetail() {}

  // ---------------- controls ----------------
  function init(controlsEl, legendEl) {
    controlsEl.innerHTML = '';
    const g0 = grp(controlsEl, 'Text');
    const sel = document.createElement('div');
    sel.className = 'ctl-row';
    sel.innerHTML = `<label>example</label><select class="ctl-select">` +
      SENTENCES.map((s, i) => `<option value="${i}">${s.split(' ').slice(0, 3).join(' ')}…</option>`).join('') +
      `</select>`;
    sel.querySelector('select').onchange = e => {
      cfg.text = SENTENCES[Number(e.target.value)];
      const inp = document.querySelector('[data-id=etext]');
      if (inp) inp.value = cfg.text;
      rebuild(); App.resetTimeline();
    };
    g0.appendChild(sel);

    const wrap = document.createElement('div');
    wrap.className = 'ctl-slider';
    wrap.innerHTML = `<div class="cs-top"><span class="cs-label">or type your own</span></div>` +
      `<input class="ctl-text" data-id="etext" type="text" maxlength="60" value="${cfg.text}">`;
    const inp = wrap.querySelector('input');
    inp.oninput = () => { cfg.text = inp.value; rebuild(); };
    inp.onchange = () => App.resetTimeline();
    g0.appendChild(wrap);

    const note = document.createElement('div');
    note.className = 'shape-note';
    note.innerHTML = `words outside the ${VOCAB.length}-token vocabulary get split into <b>##pieces</b>, or become <b>[?]</b> — exactly what real tokenizers do`;
    g0.appendChild(note);

    const g1 = grp(controlsEl, 'Vocabulary');
    const vl = document.createElement('div');
    vl.className = 'shape-note';
    vl.style.maxHeight = '150px';
    vl.style.overflowY = 'auto';
    vl.innerHTML = VOCAB.map((v, i) =>
      `<span style="color:${v.startsWith('##') ? '#d95926' : v === '[?]' ? '#e34948' : '#898781'}">${i}:${v}</span>`).join('  ');
    g1.appendChild(vl);

    legendEl.innerHTML =
      `<div class="legend-row"><span class="legend-swatch" style="background:#3987e5"></span><span>whole word</span></div>` +
      `<div class="legend-row"><span class="legend-swatch" style="background:#d95926"></span><span>##subword piece</span></div>` +
      `<div class="legend-row"><span class="legend-swatch" style="background:#e34948"></span><span>[?] unknown</span></div>` +
      `<div class="legend-row" style="margin-top:8px"><span>embedding value</span></div>` +
      (() => { const s = []; for (let i = 0; i <= 10; i++) s.push(U.divColor(U.lerp(-1, 1, i / 10), 1));
        return `<div class="legend-ramp" style="background:linear-gradient(90deg,${s.join(',')})"></div>` +
               `<div class="legend-cap"><span>−1</span><span>+1</span></div>`; })();
  }

  function grp(parent, title) {
    const g = document.createElement('div');
    g.className = 'ctl-group';
    g.innerHTML = `<h3>${title}</h3>`;
    parent.appendChild(g);
    return g;
  }

  rebuild();

  return {
    id: 'embed',
    title: 'Tokens & Embeddings',
    desc: 'How a sentence becomes numbers: split into tokens, look each one up as an integer ID, then swap that ID for a learned vector. The result is the (T × d) tensor every language model starts from.',
    VW, VH,
    get steps() { return steps; },
    init, regen: rebuild, render, caption, buildDetail, updateDetail
  };
})();
