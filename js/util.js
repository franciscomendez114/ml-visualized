'use strict';
/* ============ shared utilities: rng, colors, easing ============ */
const U = (() => {

  // deterministic rng so "regenerate" gives a fresh but repeatable tensor set
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
  function lerp(a, b, t) { return a + (b - a) * t; }

  // ---- easing ----
  function easeInOut(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }
  function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

  // ---- color ----
  function hexToRgb(h) {
    const n = parseInt(h.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function rgbToHex(r, g, b) {
    return '#' + ((1 << 24) | (r << 16) | (g << 8) | b).toString(16).slice(1);
  }
  function mix(h1, h2, t) {
    const a = hexToRgb(h1), b = hexToRgb(h2);
    return rgbToHex(
      Math.round(lerp(a[0], b[0], t)),
      Math.round(lerp(a[1], b[1], t)),
      Math.round(lerp(a[2], b[2], t)));
  }
  function shade(h, f) { // multiply brightness
    const [r, g, b] = hexToRgb(h);
    return rgbToHex(
      clamp(Math.round(r * f), 0, 255),
      clamp(Math.round(g * f), 0, 255),
      clamp(Math.round(b * f), 0, 255));
  }
  function withAlpha(h, a) {
    const [r, g, b] = hexToRgb(h);
    return `rgba(${r},${g},${b},${a})`;
  }
  function luminance(h) {
    const [r, g, b] = hexToRgb(h);
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  }
  function inkFor(bg) { return luminance(bg) > 0.56 ? '#0b0b0b' : '#ffffff'; }

  // sequential blue ramp (validated palette), dark -> light so on a dark
  // surface bigger value = brighter cell
  const SEQ = ['#0d366b', '#104281', '#184f95', '#1c5cab', '#256abf', '#2a78d6',
               '#3987e5', '#5598e7', '#6da7ec', '#86b6ef', '#9ec5f4', '#b7d3f6', '#cde2fb'];

  function rampColor(stops, t) {
    t = clamp(t, 0, 1);
    const x = t * (stops.length - 1);
    const i = Math.min(Math.floor(x), stops.length - 2);
    return mix(stops[i], stops[i + 1], x - i);
  }
  function seqColor(t) { return rampColor(SEQ, t); }

  // diverging: blue (negative) <-> neutral <-> red (positive), dark-surface steps
  const DIV_NEG = ['#383835', '#1c5cab', '#3987e5', '#86b6ef', '#cde2fb'];
  const DIV_POS = ['#383835', '#8f2a2a', '#c94b4b', '#e66767', '#f3a6a6'];
  function divColor(v, absMax) {
    if (absMax <= 0) absMax = 1;
    const t = clamp(Math.abs(v) / absMax, 0, 1);
    return v < 0 ? rampColor(DIV_NEG, t) : rampColor(DIV_POS, t);
  }

  // categorical slots (dark mode), used to tint filters / heads
  const CAT = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];

  function fmt(v) {
    if (Number.isInteger(v)) return String(v);
    const r = Math.round(v * 100) / 100;
    return String(r);
  }

  return { mulberry32, clamp, lerp, easeInOut, easeOut,
           mix, shade, withAlpha, inkFor, seqColor, divColor, rampColor,
           SEQ, CAT, fmt };
})();
