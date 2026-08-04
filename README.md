# ML Visualizer

An interactive, animated visual explainer for core machine-learning operations.
Every module plays a real computation step by step — with the actual numbers —
on 3D isometric tensors, and a math panel at the bottom shows each
multiply-and-add as it happens.

It is organised as a **learning path**, not a pile of demos. The app opens on a
lesson map (press <kbd>Esc</kbd> to return to it any time); each lesson assumes
the ones before it, and finishing one offers the next. Progress is remembered
locally, so the map shows what you've already seen and offers to resume.

## The path

**Foundations** — what a network is made of, and what "learning" means
1. A Single Neuron → 2. MLP → 3. Gradient Descent → 4. Train an MLP

**Seeing images** — why images need a different kind of layer
5. Convolution 2D → 6. Max Pooling → 7. CNN → 8. Train a CNN

**Understanding language** — how text becomes numbers, and which words matter
9. Tokens & Embeddings → 10. Attention → 11. Multi-Head Attention → 12. Transformer Block

## Modules

| Module | What it shows |
|---|---|
| **A Single Neuron** | The atom: two or three inputs, a weight on each wire, a bias, an activation. Each product is formed on screen, they sum, the bias joins, and z is read off the activation curve to give `a`. Every slider is live. A side panel shows the straight line the neuron draws through its inputs — the limitation that motivates stacking layers. |
| **MLP** | A forward pass, neuron by neuron: incoming edges light up, the weighted-sum equation builds term by term, then the activation squashes it. |
| **Gradient Descent** | Training with the neural network taken away: a ball on a landscape, the slope under it, and one step downhill — `x ← x − lr·f′(x)` — repeated. Four landscapes (bowl, bumpy with local minima, 2-D bowl, narrow ravine), a learning-rate control that visibly produces crawling / converging / overshooting / diverging, and click-to-place so you can see that where you start decides which valley you land in. |
| **Convolution 2D** | A batch of `(B, C, H, W)` inputs as 3D cubes. Watch zero-padding grow around the image, a `K×K×C` filter window slide across it (configurable stride), every element-wise product per channel, the sum + bias collapse into a single number, and that number fly into the output feature map — one map per filter, stacking into the `(B, F, Ho, Wo)` output. Supports **grouped convolution** (`G` control): the depth splits into groups and each filter's window only spans its group's channel slice. |
| **Max Pooling** | A window sweeps each channel independently; the max (or average) pops out and flies to the shrunken output. |
| **CNN** | A whole backbone: `[conv → ReLU → max-pool] × L` with per-layer kernel/padding/filters. Conv outputs appear as signed pre-activations; the ReLU step sweeps over them zeroing the negatives. The tensor chain builds cube by cube — space shrinking, channels growing — with an architecture table tracking output shape, parameter count, and receptive field per stage, ending in flatten → MLP head. Impossible configs (kernel larger than the input) are flagged where the chain breaks. |
| **Train a CNN** | The CNN lesson with the placeholder textures replaced by a real convnet learning real 16×16 images, with a **held-out validation split**. One cycle = one SGD step: minibatch → conv1 → pool → conv2 → pool → flatten → softmax → loss → backprop → update. The conv1 filters are drawn from the live weights, so you watch them go from noise to structure as it learns. Two charts: batch loss, and train vs validation accuracy with the gap called out. |
| **Tokens & Embeddings** | Where every language model actually begins. A sentence is chopped into tokens by a real greedy subword tokenizer (`cats` → `cat` + `##s`, unknown words → `[?]`), each token becomes an integer ID, and each ID is swapped for a learned vector via an explicit row lookup in the embedding table. The flat horizontal row of tokens then **morphs from 2-D into a 3-D (T × d) slab** — the tensor attention consumes. Ends with cosine similarity between the sentence's own tokens, answering "why not just use the ID number?". Type your own sentence in the sidebar. |
| **Attention** | Self-attention on a small sentence: embeddings → q/k/v projections, the attention matrix computed dot-product by dot-product (with causal masking), softmax, weighted blend of values — plus a next-token-prediction head with weight tying. |
| **Multi-Head** | H heads in parallel, each with its own q/k/v projections into a d/H subspace and its own attention pattern. Head outputs concatenate (head-colored slices show provenance) and W_O mixes them. |
| **Transformer** | One full pre-LN block as a station chain: LayerNorm → multi-head attention → residual ⊕ → LayerNorm → position-wise MLP → residual ⊕, with the residual path arcing overhead and one focus token's actual numbers tracked through every stage. |
| **Train an MLP** | A real network really training. Each timeline cycle is one SGD step: draw a minibatch → animate the forward pass for the first 3 samples (the rest are computed, not drawn) → form the loss → watch gradients flow backward → apply `w ← w − lr·∂L/∂w`. It loops, so the loss curve falls and the decision boundary reshapes while you watch. |

### The training module

The math is genuine, not staged — `js/data.js` implements forward, softmax +
cross-entropy / MSE, backprop, and SGD from scratch. The backward pass is
verified against finite differences; run it yourself in the console:

```js
ML.gradCheck()     // MLP: worst relative error ~5e-10 for both clf and reg
CNet.gradCheck()   // convnet: median ~1e-5 per weight tensor
```

`CNet.gradCheck()` reports median / p90 / worst rather than worst alone. ReLU
and max-pooling are piecewise-linear, so a ±eps perturbation can flip a ReLU
sign or a pool argmax and make the *numeric* gradient wrong at that kink — a
handful of large outliers while the bulk agrees to ~1e-5. Raising eps makes it
worse (the opposite of a truncation-error signature), which is how you can tell
it is kinks and not a broken gradient.

**Datasets.** The built-ins (two moons, concentric circles, XOR, three spirals,
house prices) are **generated locally and labeled synthetic in the UI** — there
is no network access here, and Boston Housing specifically was retired from
scikit-learn over an ethically problematic feature. For genuine data, use
**↑ Load CSV**: numeric columns, last column is the target; few distinct integer
targets → classification, otherwise regression. Features are z-scored
automatically.

**Controls.** Hidden width, activation, learning rate, batch size, plus
`⚡ Train 200 / 2000 steps` to fast-forward without animation and `⟲ Reset
weights`. Two moons reaches ~99% train accuracy in ~2000 steps.

## Running it

No dependencies, no build step — plain HTML/CSS/JS with a hand-rolled isometric
canvas renderer. Serve the folder with any static server:

```bash
python3 -m http.server 8613
```

then open <http://localhost:8613>. (Opening `index.html` directly with a
double-click also works.)

## Controls

- **Esc** — open / close the lesson map · **[** and **]** — previous / next lesson
- **▶ / Space** — play / pause
- **⏭ / →** — finish the current step, or play the next one and pause
- **⏮ / ←** — replay the current step / go back one
- **⏩** — skip to the end · **⟲ / R** — reset
- **speed** slider — 0.25× to 8×, defaulting to **0.5×** (1× is too fast to follow)
- Sidebar — change shapes (batch, channels, filters, kernel, stride, padding,
  layer sizes, tokens, mask …); **⚄ Regenerate** rolls new random values.

All tensor values are small integers (or 1-decimal floats in attention) so the
on-screen arithmetic is exact and checkable by eye.

## Code layout

```
index.html        app shell
css/style.css     dark theme UI
js/util.js        rng, easing, validated color ramps (sequential / diverging / categorical)
js/iso.js         minimal isometric renderer (boxes, face cells, wireframes)
js/data.js        datasets + the real MLP engine (forward/backward/SGD, gradCheck)
js/convnet.js     image datasets + the real convnet engine (conv/pool backprop)
js/neuron.js      Lesson 1: a single neuron, every slider live
js/gd.js          Lesson 3: gradient descent on a loss landscape
js/conv.js        Convolution 2D module (incl. grouped conv)
js/pool.js        Max/avg pooling module
js/cnn.js         CNN pipeline: [conv → pool] × L architecture view
js/cnntrain.js    Live CNN training on images, with a train/validation split
js/mlp.js         MLP forward-pass module
js/embed.js       Tokenizer, embedding table, and the 2-D → 3-D tensor morph
js/attn.js        Self-attention + language-model head
js/mha.js         Multi-head attention: parallel heads, concat, W_O
js/tblock.js      Full transformer block: LN → MHA → ⊕ → LN → FFN → ⊕
js/train.js       Live training: minibatch → forward → loss → backprop → SGD (loops)
js/main.js        curriculum path, lesson map, animation loop, transport
```

Each module exposes the same interface (`steps`, `render(ctx, step, t)`,
`caption`, `buildDetail`, `updateDetail`, `init`, `regen`), so adding a new
concept (batch norm, softmax classifier, backprop through a conv…) means writing
one new file and adding it to the `PATH` in `main.js` (which defines the
chapters and lesson order). Two optional hooks:

- `loop: true` + `onLoop()` — the timeline wraps instead of stopping, so each
  cycle advances real state (one SGD step in *Train an MLP*, one descent step in
  *Gradient Descent*).
- `interactive: true` + `onPointer(x, y, type)` — receives clicks and drags in
  the module's own design coordinates (used for click-to-place in *Gradient
  Descent*).
