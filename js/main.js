'use strict';
/* ============ app shell: curriculum path, timeline, transport ============ */
const App = (() => {

  // ---------- the learning path ----------
  // Order matters: each lesson assumes the ones before it.
  const PATH = [
    {
      chapter: 'Foundations',
      blurb: 'What a network is made of, and what "learning" actually means.',
      lessons: [
        { mod: NeuronModule, blurb: 'Multiply, add, squash. The atom every network is built from.' },
        { mod: MLPModule,    blurb: 'Put neurons side by side and stack the layers.' },
        { mod: GDModule,     blurb: 'Roll a ball downhill — the idea behind all training.' },
        { mod: TrainModule,  blurb: 'Real data, real gradients: watch a network learn.' },
      ]
    },
    {
      chapter: 'Seeing images',
      blurb: 'Why images need a different kind of layer, and what a CNN does with one.',
      lessons: [
        { mod: ConvModule, blurb: 'Slide a filter over an image and multiply-and-add.' },
        { mod: PoolModule, blurb: 'Shrink the picture, keep the strongest signal.' },
        { mod: CNNModule,  blurb: 'Chain conv → ReLU → pool into a real backbone.' },
        { mod: CNNTrainModule, blurb: 'Train one on real images and watch its filters form.' },
      ]
    },
    {
      chapter: 'Understanding language',
      blurb: 'How a model decides which words matter to which — the transformer story.',
      lessons: [
        { mod: EmbedModule,  blurb: 'Turn a sentence into the tensor a model can read.' },
        { mod: AttnModule,   blurb: 'Every word asks every other word how relevant it is.' },
        { mod: MHAModule,    blurb: 'Run several attention patterns in parallel.' },
        { mod: TBlockModule, blurb: 'The complete block that stacks into GPT.' },
      ]
    },
  ];

  // flatten into an ordered lesson list
  const LESSONS = [];
  PATH.forEach((ch, ci) => ch.lessons.forEach(l => {
    l.chapter = ch.chapter; l.ci = ci; l.n = LESSONS.length + 1;
    LESSONS.push(l);
  }));
  const indexOfId = id => LESSONS.findIndex(l => l.mod.id === id);

  let cur = 0;
  let mod = LESSONS[0].mod;

  // ---------- progress ----------
  const LS_VISITED = 'mlviz.visited', LS_LAST = 'mlviz.last';
  function visited() {
    try { return new Set(JSON.parse(localStorage.getItem(LS_VISITED) || '[]')); }
    catch { return new Set(); }
  }
  function markVisited(id) {
    try {
      const v = visited(); v.add(id);
      localStorage.setItem(LS_VISITED, JSON.stringify([...v]));
      localStorage.setItem(LS_LAST, id);
    } catch { /* private mode — progress just won't persist */ }
  }

  // ---------- dom ----------
  const canvas = document.getElementById('stage');
  const ctx = canvas.getContext('2d');
  const holder = document.getElementById('stage-holder');
  const captionEl = document.getElementById('caption');
  const detailEl = document.getElementById('detail-inner');
  const controlsEl = document.getElementById('controls');
  const legendEl = document.getElementById('legend');
  const playBtn = document.getElementById('btn-play');
  const progressBar = document.getElementById('progress-bar');
  const stepCount = document.getElementById('step-count');
  const speedEl = document.getElementById('speed');
  const speedVal = document.getElementById('speed-val');
  const homeEl = document.getElementById('home');
  const nextLessonBtn = document.getElementById('btn-nextlesson');

  // 0.5× by default: at 1× the animations are hard to actually follow
  let si = 0, t = 0, playing = true, speed = 0.5, pauseAtEnd = false;
  let lastTime = 0, lastCaption = '', builtStep = -1;

  // ---------- sizing ----------
  let view = { scale: 1, ox: 0, oy: 0, dpr: 1 };
  function resize() {
    const r = holder.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
    const scale = Math.min(r.width / mod.VW, r.height / mod.VH);
    view = { scale, dpr, ox: (r.width - mod.VW * scale) / 2, oy: (r.height - mod.VH * scale) / 2 };
  }
  new ResizeObserver(resize).observe(holder);

  // ---------- timeline ----------
  function steps() { return mod.steps; }

  function setStep(i, tt = 0) {
    si = U.clamp(i, 0, steps().length - 1);
    t = tt;
    if (builtStep !== si) { mod.buildDetail(detailEl, si); builtStep = si; }
  }

  function resetTimeline() {
    builtStep = -1;
    setStep(0, 0);
    playing = true;
    pauseAtEnd = false;
    updatePlayBtn();
    nextLessonBtn.hidden = true;
  }

  function updatePlayBtn() { playBtn.textContent = playing ? '⏸' : '▶'; }

  // ---------- loop ----------
  function frame(now) {
    const dt = Math.min((now - lastTime) / 1000, 0.1);
    lastTime = now;

    const st = steps()[si];
    if (playing && st) {
      t += dt * speed / st.dur;
      while (t >= 1) {
        if (pauseAtEnd) { t = 1; playing = false; pauseAtEnd = false; updatePlayBtn(); break; }
        if (si >= steps().length - 1) {
          // looping modules (training, gradient descent) commit and start over
          if (mod.loop) { t -= 1; mod.onLoop(); builtStep = -1; setStep(0, t); continue; }
          t = 1; playing = false; updatePlayBtn();
          if (cur < LESSONS.length - 1) showNextLesson();
          break;
        }
        t -= 1;
        setStep(si + 1, t);
      }
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(view.dpr * view.scale, 0, 0, view.dpr * view.scale,
                     view.dpr * view.ox, view.dpr * view.oy);
    try {
      mod.render(ctx, si, U.clamp(t, 0, 1));
      mod.updateDetail(si, U.clamp(t, 0, 1));
      const cap = mod.caption(si, U.clamp(t, 0, 1));
      if (cap !== lastCaption) { captionEl.innerHTML = cap; lastCaption = cap; }
    } catch (e) {
      console.error(e);
    }

    const total = steps().length;
    progressBar.style.width = ((si + t) / total * 100).toFixed(2) + '%';
    stepCount.textContent = `step ${si + 1}/${total}`;

    requestAnimationFrame(frame);
  }

  function showNextLesson() {
    const nxt = LESSONS[cur + 1];
    nextLessonBtn.innerHTML = `Next lesson &nbsp;<b>${nxt.n}. ${nxt.mod.title}</b> &nbsp;→`;
    nextLessonBtn.hidden = false;
  }

  // ---------- navigation ----------
  function goTo(i, { fromHome = false, silent = false } = {}) {
    cur = U.clamp(i, 0, LESSONS.length - 1);
    const L = LESSONS[cur];
    mod = L.mod;
    if (!silent) markVisited(mod.id);   // boot pre-loads lesson 1 without "completing" it

    document.getElementById('module-title').textContent = mod.title;
    document.getElementById('module-desc').textContent = mod.desc;
    document.getElementById('crumb-ch').textContent = L.chapter;
    document.getElementById('crumb-title').textContent = `${L.n}. ${mod.title}`;
    document.getElementById('pathpos').textContent = `${L.n} / ${LESSONS.length}`;
    document.getElementById('btn-prev').disabled = cur === 0;
    document.getElementById('btn-next').disabled = cur === LESSONS.length - 1;
    canvas.style.cursor = mod.interactive ? 'crosshair' : 'default';

    mod.init(controlsEl, legendEl);
    hideHome();
    resize();
    resetTimeline();
    if (fromHome) renderHome();     // refresh visited ticks for next time
  }

  document.getElementById('btn-prev').onclick = () => goTo(cur - 1);
  document.getElementById('btn-next').onclick = () => goTo(cur + 1);
  nextLessonBtn.onclick = () => goTo(cur + 1);

  // ---------- home / map ----------
  const NUMWORD = ['', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight',
                   'Nine', 'Ten', 'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen'];

  function renderHome() {
    const v = visited();
    document.getElementById('home-sub').textContent =
      `${NUMWORD[LESSONS.length] || LESSONS.length} lessons, each one an animation you can ` +
      `pause, step through, and poke at. Every number on screen is really computed — ` +
      `nothing is faked. Start at the beginning and the path builds on itself.`;
    const wrap = document.getElementById('home-chapters');
    wrap.innerHTML = PATH.map((ch, ci) => `
      <section class="home-ch">
        <div class="home-ch-head">
          <h2>${ch.chapter}</h2>
          <p>${ch.blurb}</p>
        </div>
        <div class="home-cards">
          ${ch.lessons.map(l => `
            <button class="home-card${v.has(l.mod.id) ? ' seen' : ''}" data-i="${l.n - 1}">
              <span class="hc-num">${l.n}</span>
              <span class="hc-body">
                <span class="hc-title">${l.mod.title}</span>
                <span class="hc-blurb">${l.blurb}</span>
              </span>
              <span class="hc-tick">${v.has(l.mod.id) ? '✓' : ''}</span>
            </button>`).join('')}
        </div>
      </section>`).join('');
    wrap.querySelectorAll('.home-card').forEach(c =>
      c.onclick = () => goTo(Number(c.dataset.i), { fromHome: true }));

    // continue where you left off
    const last = (() => { try { return localStorage.getItem(LS_LAST); } catch { return null; } })();
    const btn = document.getElementById('home-resume');
    const li = last ? indexOfId(last) : -1;
    if (li > 0) {
      btn.hidden = false;
      btn.innerHTML = `Continue: <b>${LESSONS[li].n}. ${LESSONS[li].mod.title}</b>`;
      btn.onclick = () => goTo(li, { fromHome: true });
    } else {
      btn.hidden = true;
    }
  }

  function showHome() { renderHome(); homeEl.classList.add('open'); }
  function hideHome() { homeEl.classList.remove('open'); }

  document.getElementById('btn-map').onclick = () =>
    homeEl.classList.contains('open') ? hideHome() : showHome();
  document.getElementById('home-start').onclick = () => goTo(0, { fromHome: true });

  // ---------- transport ----------
  playBtn.onclick = () => {
    if (!playing && si >= steps().length - 1 && t >= 1) resetTimeline();
    else { playing = !playing; pauseAtEnd = false; updatePlayBtn(); }
  };
  document.getElementById('btn-step').onclick = () => {
    if (playing || t < 0.999) { playing = false; pauseAtEnd = false; t = 1; }
    else if (si < steps().length - 1) { setStep(si + 1, 0); playing = true; pauseAtEnd = true; }
    updatePlayBtn();
  };
  document.getElementById('btn-back').onclick = () => {
    if (t > 0.15) setStep(si, 0);
    else setStep(Math.max(0, si - 1), 0);
    t = 0;
    playing = true; pauseAtEnd = true;
    updatePlayBtn();
  };
  document.getElementById('btn-reset').onclick = () => resetTimeline();
  document.getElementById('btn-finish').onclick = () => {
    playing = false; pauseAtEnd = false; updatePlayBtn();
    setStep(steps().length - 1, 1); t = 1;
  };

  speedEl.oninput = () => {
    speed = Math.pow(2, Number(speedEl.value));
    const r = speed >= 1 ? Math.round(speed * 10) / 10 : Math.round(speed * 100) / 100;
    speedVal.textContent = r + '×';
  };
  speedEl.oninput();

  // ---------- pointer (modules that accept clicks) ----------
  function toDesign(e) {
    const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left - view.ox) / view.scale,
            (e.clientY - r.top - view.oy) / view.scale];
  }
  let dragging = false;
  canvas.addEventListener('pointerdown', e => {
    if (!mod.onPointer) return;
    dragging = true;
    canvas.setPointerCapture(e.pointerId);
    mod.onPointer(...toDesign(e), 'down');
  });
  canvas.addEventListener('pointermove', e => {
    if (!dragging || !mod.onPointer) return;
    mod.onPointer(...toDesign(e), 'move');
  });
  canvas.addEventListener('pointerup', e => {
    dragging = false;
    if (mod.onPointer) mod.onPointer(...toDesign(e), 'up');
  });

  // ---------- keys ----------
  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.key === 'Escape') {
      e.preventDefault();
      homeEl.classList.contains('open') ? hideHome() : showHome();
      return;
    }
    if (homeEl.classList.contains('open')) return;
    if (e.code === 'Space') { e.preventDefault(); playBtn.click(); }
    else if (e.code === 'ArrowRight') document.getElementById('btn-step').click();
    else if (e.code === 'ArrowLeft') document.getElementById('btn-back').click();
    else if (e.key === 'r' || e.key === 'R') resetTimeline();
    else if (e.key === '[') goTo(cur - 1);
    else if (e.key === ']') goTo(cur + 1);
  });

  // ---------- boot ----------
  goTo(0, { silent: true });
  showHome();                       // the path is the first thing you see
  requestAnimationFrame(now => { lastTime = now; requestAnimationFrame(frame); });

  return { resetTimeline, goTo, lessons: LESSONS };
})();
