'use strict';
/* ============ image datasets + a real (small) convnet ============
 * Layers:  conv1 → ReLU → maxpool2 → conv2 → ReLU → maxpool2 → flatten → dense → softmax
 * Everything is genuine: forward, backprop through convolution and pooling,
 * plain SGD. Verify the backward pass with CNet.gradCheck().
 *
 * Tensors are flat Float32Array in CHW order: t[(c*H + y)*W + x].
 */
const CNet = (() => {

  const SIDE = 16;              // input images are 1 × 16 × 16

  // ---------------- image datasets ----------------
  // Rendered locally on a canvas — no network here, so these are generated,
  // not MNIST/Fashion-MNIST. Labelled as such in the UI.
  const DATASETS = {
    shapes: {
      name: 'shapes', classes: ['cross', 'box', 'slash', 'ring'],
      blurb: 'four drawn shapes, jittered'
    },
    digits: {
      name: 'digits 0–5', classes: ['0', '1', '2', '3', '4', '5'],
      blurb: 'glyphs from the system font, jittered'
    },
  };

  function rasterize(draw) {
    const cv = document.createElement('canvas');
    cv.width = cv.height = SIDE;
    const c = cv.getContext('2d', { willReadFrequently: true });
    c.fillStyle = '#000';
    c.fillRect(0, 0, SIDE, SIDE);
    c.strokeStyle = c.fillStyle = '#fff';
    c.lineCap = 'round';
    draw(c);
    const d = c.getImageData(0, 0, SIDE, SIDE).data;
    const out = new Float32Array(SIDE * SIDE);
    for (let i = 0; i < SIDE * SIDE; i++) out[i] = d[i * 4] / 255;
    return out;
  }

  function drawShape(c, cls, rnd) {
    const jx = (rnd() - 0.5) * 3.4, jy = (rnd() - 0.5) * 3.4;
    const s = 4.2 + rnd() * 2.2;                 // half-size
    const cx = SIDE / 2 + jx, cy = SIDE / 2 + jy;
    c.lineWidth = 1.6 + rnd() * 0.9;
    c.beginPath();
    if (cls === 0) {                              // cross
      c.moveTo(cx - s, cy); c.lineTo(cx + s, cy);
      c.moveTo(cx, cy - s); c.lineTo(cx, cy + s);
    } else if (cls === 1) {                       // box
      c.rect(cx - s, cy - s, s * 2, s * 2);
    } else if (cls === 2) {                       // slash
      const flip = rnd() < 0.5 ? 1 : -1;
      c.moveTo(cx - s, cy - s * flip); c.lineTo(cx + s, cy + s * flip);
    } else {                                      // ring
      c.arc(cx, cy, s, 0, 6.2832);
    }
    c.stroke();
  }

  function drawDigit(c, cls, rnd) {
    const size = 12 + rnd() * 4;
    c.save();
    c.translate(SIDE / 2 + (rnd() - 0.5) * 2.6, SIDE / 2 + (rnd() - 0.5) * 2.6);
    c.rotate((rnd() - 0.5) * 0.5);
    c.font = `${rnd() < 0.5 ? '600' : '400'} ${size}px system-ui, -apple-system, sans-serif`;
    c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillText(String(cls), 0, 0);
    c.restore();
  }

  // build a labelled set, split into train / validation
  function loadImages(key, n, seed, valFrac = 0.25) {
    const spec = DATASETS[key];
    const K = spec.classes.length;
    const rnd = U.mulberry32(seed);
    const X = [], y = [];
    for (let i = 0; i < n; i++) {
      const cls = i % K;
      const img = rasterize(c => key === 'shapes' ? drawShape(c, cls, rnd) : drawDigit(c, cls, rnd));
      // a little pixel noise so it is not trivially separable
      for (let p = 0; p < img.length; p++) img[p] = U.clamp(img[p] + (rnd() - 0.5) * 0.16, 0, 1);
      X.push(img); y.push(cls);
    }
    // shuffle then split
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    const nVal = Math.floor(n * valFrac);
    const pick = ii => ({ X: ii.map(i => X[i]), y: ii.map(i => y[i]) });
    const val = pick(idx.slice(0, nVal)), train = pick(idx.slice(nVal));
    return {
      key, name: spec.name, blurb: spec.blurb, classes: spec.classes, K,
      side: SIDE, train, val, nTrain: train.X.length, nVal: val.X.length
    };
  }

  // ---------------- layers ----------------
  function convFwd(inp, Cin, H, W, Wt, bias, F, K, P) {
    const Ho = H + 2 * P - K + 1, Wo = W + 2 * P - K + 1;
    const out = new Float32Array(F * Ho * Wo);
    for (let f = 0; f < F; f++) {
      const fB = f * Cin * K * K;
      for (let oy = 0; oy < Ho; oy++)
        for (let ox = 0; ox < Wo; ox++) {
          let s = bias[f];
          for (let c = 0; c < Cin; c++) {
            const cB = fB + c * K * K, iB = c * H * W;
            for (let ky = 0; ky < K; ky++) {
              const iy = oy + ky - P;
              if (iy < 0 || iy >= H) continue;
              for (let kx = 0; kx < K; kx++) {
                const ix = ox + kx - P;
                if (ix < 0 || ix >= W) continue;
                s += Wt[cB + ky * K + kx] * inp[iB + iy * W + ix];
              }
            }
          }
          out[(f * Ho + oy) * Wo + ox] = s;
        }
    }
    return { data: out, C: F, H: Ho, W: Wo };
  }

  // returns dInput; accumulates into dW / db
  function convBwd(dout, inp, Cin, H, W, Wt, F, K, P, Ho, Wo, dW, db) {
    const dInp = new Float32Array(Cin * H * W);
    for (let f = 0; f < F; f++) {
      const fB = f * Cin * K * K;
      for (let oy = 0; oy < Ho; oy++)
        for (let ox = 0; ox < Wo; ox++) {
          const g = dout[(f * Ho + oy) * Wo + ox];
          if (g === 0) continue;
          db[f] += g;
          for (let c = 0; c < Cin; c++) {
            const cB = fB + c * K * K, iB = c * H * W;
            for (let ky = 0; ky < K; ky++) {
              const iy = oy + ky - P;
              if (iy < 0 || iy >= H) continue;
              for (let kx = 0; kx < K; kx++) {
                const ix = ox + kx - P;
                if (ix < 0 || ix >= W) continue;
                dW[cB + ky * K + kx] += g * inp[iB + iy * W + ix];
                dInp[iB + iy * W + ix] += g * Wt[cB + ky * K + kx];
              }
            }
          }
        }
    }
    return dInp;
  }

  function reluFwd(t) {
    const out = new Float32Array(t.data.length);
    for (let i = 0; i < out.length; i++) out[i] = t.data[i] > 0 ? t.data[i] : 0;
    return { data: out, C: t.C, H: t.H, W: t.W };
  }
  function reluBwd(dout, z) {
    const d = new Float32Array(dout.length);
    for (let i = 0; i < d.length; i++) d[i] = z[i] > 0 ? dout[i] : 0;
    return d;
  }

  function poolFwd(t) {                       // 2×2, stride 2
    const { C, H, W } = t, Ho = H >> 1, Wo = W >> 1;
    const out = new Float32Array(C * Ho * Wo);
    const arg = new Int32Array(C * Ho * Wo);
    for (let c = 0; c < C; c++)
      for (let oy = 0; oy < Ho; oy++)
        for (let ox = 0; ox < Wo; ox++) {
          let best = -Infinity, bi = 0;
          for (let dy = 0; dy < 2; dy++)
            for (let dx = 0; dx < 2; dx++) {
              const i = (c * H + oy * 2 + dy) * W + ox * 2 + dx;
              if (t.data[i] > best) { best = t.data[i]; bi = i; }
            }
          const o = (c * Ho + oy) * Wo + ox;
          out[o] = best; arg[o] = bi;
        }
    return { data: out, C, H: Ho, W: Wo, arg };
  }
  function poolBwd(dout, p, inLen) {
    const d = new Float32Array(inLen);
    for (let i = 0; i < dout.length; i++) d[p.arg[i]] += dout[i];
    return d;
  }

  function softmax(zs) {
    const m = Math.max(...zs);
    const e = zs.map(v => Math.exp(v - m));
    const S = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / S);
  }

  // ---------------- the network ----------------
  function makeNet(cfg, seed) {
    const rnd = U.mulberry32(seed);
    const gauss = () => {
      let u = 0, v = 0;
      while (u === 0) u = rnd();
      while (v === 0) v = rnd();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.283185307 * v);
    };
    const mk = (n, fanIn) => {
      const a = new Float32Array(n), s = Math.sqrt(2 / fanIn);
      for (let i = 0; i < n; i++) a[i] = gauss() * s;
      return a;
    };
    const { F1, F2, K, nOut } = cfg;
    const P = (K - 1) >> 1;
    const H1 = SIDE, H2 = SIDE >> 1, H3 = H2, H4 = H2 >> 1;
    const flat = F2 * H4 * H4;
    return {
      cfg: { ...cfg, P }, SIDE,
      W1: mk(F1 * 1 * K * K, K * K), b1: new Float32Array(F1),
      W2: mk(F2 * F1 * K * K, F1 * K * K), b2: new Float32Array(F2),
      Wd: mk(nOut * flat, flat), bd: new Float32Array(nOut),
      dims: { H1, H2, H3, H4, flat },
    };
  }

  function forward(net, img) {
    const { F1, F2, K, P, nOut } = net.cfg;
    const x = { data: img, C: 1, H: SIDE, W: SIDE };
    const z1 = convFwd(x.data, 1, SIDE, SIDE, net.W1, net.b1, F1, K, P);
    const a1 = reluFwd(z1);
    const p1 = poolFwd(a1);
    const z2 = convFwd(p1.data, F1, p1.H, p1.W, net.W2, net.b2, F2, K, P);
    const a2 = reluFwd(z2);
    const p2 = poolFwd(a2);
    const flat = p2.data;
    const logits = new Array(nOut);
    for (let o = 0; o < nOut; o++) {
      let s = net.bd[o];
      const B = o * flat.length;
      for (let i = 0; i < flat.length; i++) s += net.Wd[B + i] * flat[i];
      logits[o] = s;
    }
    const probs = softmax(logits);
    return { x, z1, a1, p1, z2, a2, p2, flat, logits, probs };
  }

  function zeroGrads(net) {
    return {
      W1: new Float32Array(net.W1.length), b1: new Float32Array(net.b1.length),
      W2: new Float32Array(net.W2.length), b2: new Float32Array(net.b2.length),
      Wd: new Float32Array(net.Wd.length), bd: new Float32Array(net.bd.length),
    };
  }

  function backward(net, c, label, g) {
    const { F1, F2, K, P, nOut } = net.cfg;
    // softmax + cross-entropy
    const dlogits = c.probs.slice();
    dlogits[label] -= 1;
    // dense
    const dflat = new Float32Array(c.flat.length);
    for (let o = 0; o < nOut; o++) {
      const gd = dlogits[o], B = o * c.flat.length;
      g.bd[o] += gd;
      if (gd === 0) continue;
      for (let i = 0; i < c.flat.length; i++) {
        g.Wd[B + i] += gd * c.flat[i];
        dflat[i] += gd * net.Wd[B + i];
      }
    }
    // pool2 → relu2 → conv2
    const da2 = poolBwd(dflat, c.p2, c.a2.data.length);
    const dz2 = reluBwd(da2, c.z2.data);
    const dp1 = convBwd(dz2, c.p1.data, F1, c.p1.H, c.p1.W, net.W2,
                        F2, K, P, c.z2.H, c.z2.W, g.W2, g.b2);
    // pool1 → relu1 → conv1
    const da1 = poolBwd(dp1, c.p1, c.a1.data.length);
    const dz1 = reluBwd(da1, c.z1.data);
    convBwd(dz1, c.x.data, 1, SIDE, SIDE, net.W1,
            F1, K, P, c.z1.H, c.z1.W, g.W1, g.b1);
    return { dlogits, dz1, dz2 };
  }

  const loss = (c, label) => -Math.log(Math.max(c.probs[label], 1e-12));

  function scaleGrads(g, k) {
    for (const key of ['W1', 'b1', 'W2', 'b2', 'Wd', 'bd'])
      for (let i = 0; i < g[key].length; i++) g[key][i] *= k;
  }
  function sgdStep(net, g, lr) {
    for (const key of ['W1', 'b1', 'W2', 'b2', 'Wd', 'bd'])
      for (let i = 0; i < net[key].length; i++) net[key][i] -= lr * g[key][i];
  }

  function trainBatch(net, X, Y, idx, lr, apply = true) {
    const g = zeroGrads(net);
    const caches = [], losses = [];
    for (const i of idx) {
      const c = forward(net, X[i]);
      caches.push(c);
      losses.push(loss(c, Y[i]));
      backward(net, c, Y[i], g);
    }
    scaleGrads(g, 1 / idx.length);
    const meanLoss = losses.reduce((a, b) => a + b, 0) / losses.length;
    if (apply) sgdStep(net, g, lr);
    return { g, caches, losses, meanLoss };
  }

  function evaluate(net, X, Y, cap = 1e9) {
    const n = Math.min(X.length, cap);
    let ok = 0, L = 0;
    for (let i = 0; i < n; i++) {
      const c = forward(net, X[i]);
      L += loss(c, Y[i]);
      let am = 0;
      for (let k = 1; k < c.probs.length; k++) if (c.probs[k] > c.probs[am]) am = k;
      if (am === Y[i]) ok++;
    }
    return { acc: ok / n, loss: L / n };
  }

  const paramCount = net =>
    net.W1.length + net.b1.length + net.W2.length + net.b2.length +
    net.Wd.length + net.bd.length;

  // ---------------- gradient check ----------------
  // Finite differences vs. backprop on a random image.
  //
  // Reported as median / p90 / worst rather than worst alone, on purpose:
  // ReLU and max-pooling are piecewise-linear, so a perturbation of ±eps can
  // flip a ReLU sign or a pool argmax and make the NUMERIC gradient wrong at
  // that kink. Those show up as a handful of large outliers while the bulk
  // agrees to ~1e-5. (Confirm it is kinks and not a bug by raising eps: the
  // error gets worse, which is the opposite of a truncation-error signature.)
  // Weights are Float32, so eps below ~1e-4 starts drowning in round-off.
  function gradCheck(seed = 4, eps = 1e-3) {
    const net = makeNet({ F1: 3, F2: 4, K: 3, nOut: 4 }, seed);
    const rnd = U.mulberry32(seed + 1);
    const img = new Float32Array(SIDE * SIDE);
    for (let i = 0; i < img.length; i++) img[i] = rnd();
    const label = 2;
    const g = zeroGrads(net);
    backward(net, forward(net, img), label, g);
    const report = {};
    for (const key of ['W1', 'W2', 'Wd', 'b1', 'b2', 'bd']) {
      const errs = [];
      const stride = Math.max(1, Math.floor(net[key].length / 60));
      for (let i = 0; i < net[key].length; i += stride) {
        const orig = net[key][i];
        net[key][i] = orig + eps; const lp = loss(forward(net, img), label);
        net[key][i] = orig - eps; const lm = loss(forward(net, img), label);
        net[key][i] = orig;
        const num = (lp - lm) / (2 * eps), ana = g[key][i];
        const den = Math.max(1e-7, Math.abs(num) + Math.abs(ana));
        errs.push(Math.abs(num - ana) / den);
      }
      errs.sort((a, b) => a - b);
      report[key] = {
        median: +errs[errs.length >> 1].toExponential(2),
        p90: +errs[Math.floor(errs.length * 0.9)].toExponential(2),
        worst: +errs[errs.length - 1].toExponential(2),
        okFrac: +(errs.filter(e => e < 1e-3).length / errs.length).toFixed(3),
      };
    }
    return report;
  }

  return { SIDE, DATASETS, loadImages, makeNet, forward, backward, zeroGrads,
           scaleGrads, sgdStep, trainBatch, evaluate, loss, paramCount,
           softmax, gradCheck };
})();
