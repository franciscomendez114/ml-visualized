'use strict';
/* ============ datasets + tiny MLP engine (real forward / backward / SGD) ============
 * Everything here is honest, checkable math — no hand-waving:
 *   forward  : z = W·a + b, a = act(z)
 *   loss     : softmax + cross-entropy (clf) | mean squared error (reg)
 *   backward : standard chain rule, gradient-checkable via ML.gradCheck()
 *   update   : plain SGD, w ← w − lr·∂L/∂w
 */
const ML = (() => {

  // ---------------- datasets ----------------
  // The built-ins are GENERATED LOCALLY (clearly labeled synthetic in the UI):
  // no network access here, and Boston Housing is retired upstream for ethical
  // reasons. Use "load CSV" for genuine data.
  const DATASETS = {
    moons: {
      name: 'two moons', task: 'clf', synthetic: true,
      feats: ['x₁', 'x₂'], classes: ['A', 'B'],
      gen: (rnd, n = 200) => {
        const X = [], y = [];
        for (let i = 0; i < n; i++) {
          const c = i % 2, t = rnd() * Math.PI;
          const nx = (rnd() - 0.5) * 0.32, ny = (rnd() - 0.5) * 0.32;
          if (c === 0) X.push([Math.cos(t) + nx, Math.sin(t) + ny]);
          else X.push([1 - Math.cos(t) + nx, 0.5 - Math.sin(t) + ny]);
          y.push(c);
        }
        return { X, y };
      }
    },
    circles: {
      name: 'concentric circles', task: 'clf', synthetic: true,
      feats: ['x₁', 'x₂'], classes: ['inner', 'outer'],
      gen: (rnd, n = 200) => {
        const X = [], y = [];
        for (let i = 0; i < n; i++) {
          const c = i % 2, t = rnd() * 6.283, r = (c ? 1.7 : 0.7) + (rnd() - 0.5) * 0.4;
          X.push([r * Math.cos(t), r * Math.sin(t)]);
          y.push(c);
        }
        return { X, y };
      }
    },
    xor: {
      name: 'XOR quadrants', task: 'clf', synthetic: true,
      feats: ['x₁', 'x₂'], classes: ['0', '1'],
      gen: (rnd, n = 200) => {
        const X = [], y = [];
        for (let i = 0; i < n; i++) {
          const a = rnd() * 2 - 1, b = rnd() * 2 - 1;
          X.push([a + (rnd() - 0.5) * 0.15, b + (rnd() - 0.5) * 0.15]);
          y.push((a > 0) !== (b > 0) ? 1 : 0);
        }
        return { X, y };
      }
    },
    spiral: {
      name: 'three spirals', task: 'clf', synthetic: true,
      feats: ['x₁', 'x₂'], classes: ['A', 'B', 'C'],
      gen: (rnd, n = 240) => {
        const X = [], y = [], K = 3;
        for (let c = 0; c < K; c++)
          for (let i = 0; i < n / K; i++) {
            const r = (i / (n / K)) * 2;
            const th = c * 6.283 / K + r * 1.9 + (rnd() - 0.5) * 0.3;
            X.push([r * Math.sin(th), r * Math.cos(th)]);
            y.push(c);
          }
        return { X, y };
      }
    },
    housing: {
      name: 'house prices', task: 'reg', synthetic: true,
      feats: ['area', 'rooms', 'age'], target: 'price ($k)',
      gen: (rnd, n = 200) => {
        const X = [], y = [];
        for (let i = 0; i < n; i++) {
          const area = 60 + rnd() * 200;            // m²
          const rooms = 1 + Math.floor(rnd() * 5);
          const age = rnd() * 60;                   // years
          // nonlinear-ish price with noise, in $k
          const price = 40 + 1.6 * area + 18 * rooms - 0.9 * age
                        + 0.004 * area * area / 10 + (rnd() - 0.5) * 40;
          X.push([area, rooms, age]);
          y.push(price);
        }
        return { X, y };
      }
    }
  };

  // z-score standardize columns; returns {Z, mu, sd}
  function standardize(X) {
    const n = X.length, d = X[0].length;
    const mu = Array(d).fill(0), sd = Array(d).fill(0);
    for (const row of X) for (let j = 0; j < d; j++) mu[j] += row[j] / n;
    for (const row of X) for (let j = 0; j < d; j++) sd[j] += (row[j] - mu[j]) ** 2 / n;
    for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j]) || 1;
    return { Z: X.map(r => r.map((v, j) => (v - mu[j]) / sd[j])), mu, sd };
  }

  function loadBuiltin(key, seed) {
    const spec = DATASETS[key];
    const rnd = U.mulberry32(seed);
    const { X, y } = spec.gen(rnd);
    const { Z, mu, sd } = standardize(X);
    const ds = {
      key, name: spec.name, task: spec.task, synthetic: true,
      feats: spec.feats, classes: spec.classes || null,
      targetName: spec.target || 'y',
      Xraw: X, X: Z, muX: mu, sdX: sd,
      nIn: X[0].length
    };
    if (spec.task === 'clf') {
      ds.y = y; ds.nOut = Math.max(...y) + 1;
    } else {
      const n = y.length;
      const muY = y.reduce((a, b) => a + b, 0) / n;
      const sdY = Math.sqrt(y.reduce((a, b) => a + (b - muY) ** 2, 0) / n) || 1;
      ds.yraw = y; ds.y = y.map(v => (v - muY) / sdY);
      ds.muY = muY; ds.sdY = sdY; ds.nOut = 1;
    }
    return ds;
  }

  // CSV: numeric columns, last column = target.
  // Few distinct integer targets → classification, otherwise regression.
  function parseCSV(text, name) {
    const lines = text.trim().split(/\r?\n/).filter(l => l.trim());
    if (lines.length < 4) throw new Error('need at least 4 rows');
    const split = l => l.split(/[,;\t]/).map(s => s.trim());
    let header = null, start = 0;
    const first = split(lines[0]);
    if (first.some(c => c !== '' && isNaN(Number(c)))) { header = first; start = 1; }
    const rows = [];
    for (let i = start; i < lines.length; i++) {
      const parts = split(lines[i]).map(Number);
      if (parts.length < 2 || parts.some(v => !isFinite(v))) continue;
      rows.push(parts);
    }
    if (rows.length < 4) throw new Error('no usable numeric rows');
    const d = rows[0].length - 1;
    const X = rows.map(r => r.slice(0, d));
    const rawY = rows.map(r => r[d]);
    const uniq = [...new Set(rawY)].sort((a, b) => a - b);
    const isClf = uniq.length <= 8 && uniq.every(v => Number.isInteger(v));
    const { Z, mu, sd } = standardize(X);
    const ds = {
      key: 'csv', name: name || 'CSV', task: isClf ? 'clf' : 'reg', synthetic: false,
      feats: header ? header.slice(0, d) : Array.from({ length: d }, (_, i) => 'f' + i),
      targetName: header ? header[d] : 'y',
      Xraw: X, X: Z, muX: mu, sdX: sd, nIn: d
    };
    if (isClf) {
      const map = new Map(uniq.map((v, i) => [v, i]));
      ds.y = rawY.map(v => map.get(v));
      ds.nOut = uniq.length;
      ds.classes = uniq.map(String);
    } else {
      const n = rawY.length;
      const muY = rawY.reduce((a, b) => a + b, 0) / n;
      const sdY = Math.sqrt(rawY.reduce((a, b) => a + (b - muY) ** 2, 0) / n) || 1;
      ds.yraw = rawY; ds.y = rawY.map(v => (v - muY) / sdY);
      ds.muY = muY; ds.sdY = sdY; ds.nOut = 1;
    }
    return ds;
  }

  // ---------------- network ----------------
  const ACTS = {
    relu:    { f: z => Math.max(0, z),          df: (z, a) => z > 0 ? 1 : 0,  name: 'ReLU' },
    tanh:    { f: z => Math.tanh(z),            df: (z, a) => 1 - a * a,      name: 'tanh' },
    sigmoid: { f: z => 1 / (1 + Math.exp(-z)),  df: (z, a) => a * (1 - a),    name: 'σ' },
  };

  function makeNet(sizes, actName, task, seed) {
    const rnd = U.mulberry32(seed);
    const gauss = () => {  // Box–Muller
      let u = 0, v = 0;
      while (u === 0) u = rnd();
      while (v === 0) v = rnd();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.283185307 * v);
    };
    const L = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const nin = sizes[l], nout = sizes[l + 1];
      const scale = actName === 'relu' ? Math.sqrt(2 / nin) : Math.sqrt(1 / nin);
      L.push({
        W: Array.from({ length: nout }, () => Array.from({ length: nin }, () => gauss() * scale)),
        b: Array(nout).fill(0), nin, nout
      });
    }
    return { sizes, L, act: actName, task };
  }

  // returns {z: [], a: []} with a[0] = input, a[l+1] = activation of layer l
  function forward(net, x) {
    const act = ACTS[net.act];
    const a = [x.slice()], z = [];
    for (let l = 0; l < net.L.length; l++) {
      const { W, b, nout, nin } = net.L[l];
      const zl = Array(nout).fill(0);
      for (let i = 0; i < nout; i++) {
        let s = b[i];
        for (let j = 0; j < nin; j++) s += W[i][j] * a[l][j];
        zl[i] = s;
      }
      z.push(zl);
      const last = l === net.L.length - 1;
      a.push(last ? (net.task === 'clf' ? softmax(zl) : zl.slice())
                  : zl.map(v => act.f(v)));
    }
    return { z, a, out: a[a.length - 1] };
  }

  function softmax(zs) {
    const m = Math.max(...zs);
    const e = zs.map(v => Math.exp(v - m));
    const S = e.reduce((p, q) => p + q, 0);
    return e.map(v => v / S);
  }

  function loss(net, cache, y) {
    const p = cache.out;
    if (net.task === 'clf') return -Math.log(Math.max(p[y], 1e-12));
    return 0.5 * (p[0] - y) ** 2;
  }

  function zeroGrads(net) {
    return net.L.map(l => ({
      W: Array.from({ length: l.nout }, () => Array(l.nin).fill(0)),
      b: Array(l.nout).fill(0)
    }));
  }

  // accumulate ∂L/∂W, ∂L/∂b for one sample; returns per-layer deltas for viz
  function backward(net, cache, y, grads) {
    const act = ACTS[net.act];
    const Ln = net.L.length;
    const deltas = Array(Ln);
    // output layer: softmax+CE and MSE+linear both give (p − target)
    let d;
    if (net.task === 'clf') {
      d = cache.out.slice();
      d[y] -= 1;
    } else {
      d = [cache.out[0] - y];
    }
    for (let l = Ln - 1; l >= 0; l--) {
      deltas[l] = d.slice();
      const { W, nin, nout } = net.L[l];
      for (let i = 0; i < nout; i++) {
        grads[l].b[i] += d[i];
        for (let j = 0; j < nin; j++) grads[l].W[i][j] += d[i] * cache.a[l][j];
      }
      if (l > 0) {
        const dPrev = Array(nin).fill(0);
        for (let j = 0; j < nin; j++) {
          let s = 0;
          for (let i = 0; i < nout; i++) s += W[i][j] * d[i];
          dPrev[j] = s * act.df(cache.z[l - 1][j], cache.a[l][j]);
        }
        d = dPrev;
      }
    }
    return deltas;
  }

  function scaleGrads(grads, k) {
    for (const g of grads) {
      for (let i = 0; i < g.b.length; i++) {
        g.b[i] *= k;
        for (let j = 0; j < g.W[i].length; j++) g.W[i][j] *= k;
      }
    }
  }

  function sgdStep(net, grads, lr) {
    net.L.forEach((l, li) => {
      for (let i = 0; i < l.nout; i++) {
        l.b[i] -= lr * grads[li].b[i];
        for (let j = 0; j < l.nin; j++) l.W[i][j] -= lr * grads[li].W[i][j];
      }
    });
  }

  // run one minibatch end-to-end; returns everything needed for the viz
  function trainBatch(net, X, Y, idx, lr, apply = true) {
    const grads = zeroGrads(net);
    const caches = [], losses = [];
    for (const i of idx) {
      const c = forward(net, X[i]);
      caches.push(c);
      losses.push(loss(net, c, Y[i]));
      backward(net, c, Y[i], grads);
    }
    scaleGrads(grads, 1 / idx.length);
    const meanLoss = losses.reduce((a, b) => a + b, 0) / losses.length;
    if (apply) sgdStep(net, grads, lr);
    return { grads, caches, losses, meanLoss };
  }

  function accuracy(net, X, Y) {
    let ok = 0;
    for (let i = 0; i < X.length; i++) {
      const p = forward(net, X[i]).out;
      let am = 0;
      for (let k = 1; k < p.length; k++) if (p[k] > p[am]) am = k;
      if (am === Y[i]) ok++;
    }
    return ok / X.length;
  }
  function meanLoss(net, X, Y) {
    let s = 0;
    for (let i = 0; i < X.length; i++) s += loss(net, forward(net, X[i]), Y[i]);
    return s / X.length;
  }

  // numeric gradient check — proof the backward pass is right
  function gradCheck(seed = 3) {
    const out = [];
    for (const [task, nOut] of [['clf', 3], ['reg', 1]]) {
      const net = makeNet([3, 5, nOut], 'tanh', task, seed);
      const x = [0.4, -1.1, 0.7], y = task === 'clf' ? 2 : 0.83;
      const grads = zeroGrads(net);
      backward(net, forward(net, x), y, grads);
      const eps = 1e-5;
      let worst = 0;
      net.L.forEach((l, li) => {
        for (let i = 0; i < l.nout; i++)
          for (let j = 0; j < l.nin; j++) {
            const orig = l.W[i][j];
            l.W[i][j] = orig + eps; const lp = loss(net, forward(net, x), y);
            l.W[i][j] = orig - eps; const lm = loss(net, forward(net, x), y);
            l.W[i][j] = orig;
            const num = (lp - lm) / (2 * eps), ana = grads[li].W[i][j];
            const den = Math.max(1e-8, Math.abs(num) + Math.abs(ana));
            worst = Math.max(worst, Math.abs(num - ana) / den);
          }
      });
      out.push({ task, worstRelErr: worst });
    }
    return out;
  }

  return { DATASETS, loadBuiltin, parseCSV, standardize, ACTS,
           makeNet, forward, softmax, loss, zeroGrads, backward, scaleGrads,
           sgdStep, trainBatch, accuracy, meanLoss, gradCheck };
})();
