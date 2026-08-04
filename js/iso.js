'use strict';
/* ============ minimal isometric renderer ============
 * World axes: x → image columns (right-down on screen)
 *             y → up (image row r maps to y = H-1-r)
 *             z → depth/channels (left-down on screen)
 * Visible faces of a box: top (+y), right (+x), front (+z).
 */
const Iso = (() => {

  function make(ctx, s, ox, oy) {
    const ux = s * 0.90, uy = s * 0.45, h = s * 0.92;
    const iso = { ctx, s, ux, uy, h, ox, oy };
    iso.p = (x, y, z) => [ox + (x - z) * ux, oy + (x + z) * uy - y * h];
    return iso;
  }

  function poly(ctx, pts, fill, stroke, lw = 1) {
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
  }

  // face quads of a box spanning (x,y,z) .. (x+dx, y+dy, z+dz)
  function topFace(iso, x, y, z, dx, dy, dz) {
    const Y = y + dy;
    return [iso.p(x, Y, z), iso.p(x + dx, Y, z), iso.p(x + dx, Y, z + dz), iso.p(x, Y, z + dz)];
  }
  function rightFace(iso, x, y, z, dx, dy, dz) {
    const X = x + dx;
    return [iso.p(X, y, z), iso.p(X, y, z + dz), iso.p(X, y + dy, z + dz), iso.p(X, y + dy, z)];
  }
  function frontFace(iso, x, y, z, dx, dy, dz) {
    const Z = z + dz;
    return [iso.p(x, y, Z), iso.p(x + dx, y, Z), iso.p(x + dx, y + dy, Z), iso.p(x, y + dy, Z)];
  }

  // solid box with 3 shaded faces
  function box(iso, x, y, z, dx, dy, dz, color, opts = {}) {
    const ctx = iso.ctx;
    const a = opts.alpha != null ? opts.alpha : 1;
    ctx.save();
    ctx.globalAlpha = a;
    const st = opts.stroke || U.withAlpha(U.shade(color, 0.35), 0.9);
    poly(ctx, rightFace(iso, x, y, z, dx, dy, dz), U.shade(color, 0.62), st, opts.lw || 1);
    poly(ctx, frontFace(iso, x, y, z, dx, dy, dz), color, st, opts.lw || 1);
    poly(ctx, topFace(iso, x, y, z, dx, dy, dz), U.shade(color, 1.28), st, opts.lw || 1);
    ctx.restore();
  }

  // wireframe of visible edges of a box (used for selection volumes)
  function wire(iso, x, y, z, dx, dy, dz, color, lw = 2, alpha = 1) {
    const ctx = iso.ctx;
    ctx.save();
    ctx.globalAlpha = alpha;
    poly(ctx, topFace(iso, x, y, z, dx, dy, dz), null, color, lw);
    poly(ctx, rightFace(iso, x, y, z, dx, dy, dz), null, color, lw);
    poly(ctx, frontFace(iso, x, y, z, dx, dy, dz), null, color, lw);
    ctx.restore();
  }

  // front-face cell quad: grid col c (0..W-1), row r (0 = top), on plane z=Z.
  // (x0,y0) = world coords of the grid's bottom-left cell corner.
  function cellQuad(iso, x0, y0, Z, c, r, H) {
    const yb = y0 + (H - 1 - r);
    return [iso.p(x0 + c, yb, Z), iso.p(x0 + c + 1, yb, Z),
            iso.p(x0 + c + 1, yb + 1, Z), iso.p(x0 + c, yb + 1, Z)];
  }
  function cellCenter(iso, x0, y0, Z, c, r, H) {
    const yb = y0 + (H - 1 - r);
    return iso.p(x0 + c + 0.5, yb + 0.5, Z);
  }

  // draw one front-face cell (fill + hairline + optional text)
  function cell(iso, x0, y0, Z, c, r, H, fill, text, opts = {}) {
    const ctx = iso.ctx;
    const q = cellQuad(iso, x0, y0, Z, c, r, H);
    ctx.save();
    if (opts.alpha != null) ctx.globalAlpha = opts.alpha;
    if (opts.dashed) ctx.setLineDash([3, 3]);
    poly(ctx, q, fill, opts.stroke || 'rgba(0,0,0,0.35)', opts.lw || 1);
    ctx.setLineDash([]);
    if (text != null && iso.s >= 15) {
      const [cx, cy] = cellCenter(iso, x0, y0, Z, c, r, H);
      ctx.fillStyle = opts.ink || U.inkFor(fill);
      ctx.font = `600 ${Math.max(9, iso.s * 0.40)}px ui-monospace, Menlo, monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, cx, cy + 0.5);
    }
    ctx.restore();
  }

  // top-face cell (col c, depth d) at plane y=Y (world top)
  function topCell(iso, x0, z0, Y, c, d, fill, opts = {}) {
    const q = [iso.p(x0 + c, Y, z0 + d), iso.p(x0 + c + 1, Y, z0 + d),
               iso.p(x0 + c + 1, Y, z0 + d + 1), iso.p(x0 + c, Y, z0 + d + 1)];
    const ctx = iso.ctx;
    ctx.save();
    if (opts.alpha != null) ctx.globalAlpha = opts.alpha;
    poly(ctx, q, fill, opts.stroke || 'rgba(0,0,0,0.3)', 1);
    ctx.restore();
  }

  // right-face cell (depth d, row r from top) at plane x=X
  function rightCell(iso, y0, z0, X, d, r, H, fill, opts = {}) {
    const yb = y0 + (H - 1 - r);
    const q = [iso.p(X, yb, z0 + d), iso.p(X, yb, z0 + d + 1),
               iso.p(X, yb + 1, z0 + d + 1), iso.p(X, yb + 1, z0 + d)];
    const ctx = iso.ctx;
    ctx.save();
    if (opts.alpha != null) ctx.globalAlpha = opts.alpha;
    poly(ctx, q, fill, opts.stroke || 'rgba(0,0,0,0.3)', 1);
    ctx.restore();
  }

  // label under a box
  function label(iso, x, z, text, opts = {}) {
    const ctx = iso.ctx;
    const [px, py] = iso.p(x, -0.55, z);
    ctx.save();
    ctx.fillStyle = opts.color || '#898781';
    ctx.font = `${opts.weight || 500} ${opts.size || 12}px system-ui, sans-serif`;
    ctx.textAlign = opts.align || 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(text, px, py + (opts.dy || 0));
    ctx.restore();
  }

  return { make, poly, box, wire, cell, cellQuad, cellCenter, topCell, rightCell,
           topFace, rightFace, frontFace, label };
})();
