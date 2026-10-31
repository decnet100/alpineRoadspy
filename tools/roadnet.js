// Road network helpers for fix_picker.html: snapping points to road segments,
// shortest paths across segments (so a drive may change roads between fixes),
// and positions along a routed polyline. No DOM access, so it runs in node too.
'use strict';

const RoadNet = (() => {
  const R = 6371008.8;
  const RAD = Math.PI / 180;

  function haversine(lat1, lon1, lat2, lon2) {
    const dLat = (lat2 - lat1) * RAD, dLon = (lon2 - lon1) * RAD;
    const h = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * RAD) * Math.cos(lat2 * RAD) * Math.sin(dLon / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(h));
  }

  // Local equirectangular projection in metres; accurate enough over a few tens of km.
  function projection(lat0) {
    const kx = R * RAD * Math.cos(lat0 * RAD), ky = R * RAD;
    return {
      toXY: (lat, lon) => [lon * kx, lat * ky],
      toLatLon: (x, y) => [y / ky, x / kx],
    };
  }

  const nodeKey = (c) => c[0].toFixed(6) + ',' + c[1].toFixed(6);

  class MinHeap {
    constructor() { this.a = []; }
    get size() { return this.a.length; }
    push(item) {
      const a = this.a; a.push(item);
      let i = a.length - 1;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (a[p][0] <= a[i][0]) break;
        [a[p], a[i]] = [a[i], a[p]]; i = p;
      }
    }
    pop() {
      const a = this.a, top = a[0], last = a.pop();
      if (a.length) {
        a[0] = last;
        let i = 0;
        for (;;) {
          const l = 2 * i + 1, r = l + 1;
          let m = i;
          if (l < a.length && a[l][0] < a[m][0]) m = l;
          if (r < a.length && a[r][0] < a[m][0]) m = r;
          if (m === i) break;
          [a[m], a[i]] = [a[i], a[m]]; i = m;
        }
      }
      return top;
    }
  }

  // Segments: { id, coords: [[lat, lon], ...], name, cost, from, to, structure, level }.
  // With node ids (`from`/`to`, e.g. GIP links) segments connect exactly there
  // and nowhere else. Without them, topology comes from shared vertex
  // coordinates (rounded to ~0.1 m). Not every such dataset splits segments at
  // junctions, so ordinary segments connect at every shared vertex, but:
  //  - `structure: true` (bridge, tunnel): interior vertices stay private, so
  //    the road below/above an overpass does not become a junction;
  //  - `level: 'motorway'` vertices form a separate network, joined only by
  //    `level: 'ramp'` segments.
  // Segment ends lying on another segment (T-junction without a common vertex)
  // are connected too. `cost` (default 1) multiplies the routing weight, e.g.
  // to use footpaths only as a last resort.
  class Network {
    constructor() {
      this.segs = new Map();
      this.proj = null;
      this.dirty = true;
    }

    get size() { return this.segs.size; }

    addSegments(segs) {
      for (const s of segs) {
        if (s.coords.length < 2 || this.segs.has(s.id)) continue;
        if (!this.proj) this.proj = projection(s.coords[0][0]);
        this.segs.set(s.id, s);
        this.dirty = true;
      }
    }

    xy(c) { return this.proj.toXY(c[0], c[1]); }

    key(s, i) {
      const last = s.coords.length - 1;
      if (s.from != null) return i === 0 ? 'n:' + s.from : i === last ? 'n:' + s.to : s.id + '#' + i;
      if (s.structure && i > 0 && i < last) return s.id + '#' + i;
      return (s.level === 'motorway' ? 'M:' : '') + nodeKey(s.coords[i]);
    }

    build() {
      if (!this.dirty) return;
      this.adj = new Map();
      this.nodes = new Map();
      const link = (a, b, w, segId) => {
        if (!this.adj.has(a)) this.adj.set(a, []);
        this.adj.get(a).push([b, w, segId]);
      };
      for (const s of this.segs.values()) {
        s.xy = s.coords.map((c) => this.xy(c));
        for (let i = 0; i < s.coords.length; i++) {
          const k = this.key(s, i);
          this.nodes.set(k, s.coords[i]);
          if (s.level === 'ramp' && !k.includes('#')) {
            this.nodes.set('M:' + k, s.coords[i]);
            link(k, 'M:' + k, 0, s.id); link('M:' + k, k, 0, s.id);
          }
        }
        for (let i = 0; i + 1 < s.coords.length; i++) {
          const a = this.key(s, i), b = this.key(s, i + 1);
          if (a === b) continue;
          const w = Math.hypot(s.xy[i + 1][0] - s.xy[i][0], s.xy[i + 1][1] - s.xy[i][1]) * (s.cost || 1);
          link(a, b, w, s.id); link(b, a, w, s.id);
        }
      }
      this.connectTJunctions(link);
      this.dirty = false;
    }

    // Link dangling segment ends that lie within `tol` metres of another segment.
    connectTJunctions(link, tol = 1) {
      for (const s of this.segs.values()) {
        if (s.from != null) continue;
        for (const end of [0, s.coords.length - 1]) {
          const k = this.key(s, end);
          if ((this.adj.get(k) || []).length !== 1) continue;
          const [px, py] = s.xy[end];
          for (const o of this.segs.values()) {
            if (o === s || (o.level === 'motorway') !== (s.level === 'motorway')) continue;
            for (let i = 0; i + 1 < o.xy.length; i++) {
              const [ax, ay] = o.xy[i], [bx, by] = o.xy[i + 1];
              if (Math.max(ax, bx) < px - tol || Math.min(ax, bx) > px + tol ||
                  Math.max(ay, by) < py - tol || Math.min(ay, by) > py + tol) continue;
              const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
              const f = len2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2)) : 0;
              if (f === 0 || f === 1 || Math.hypot(ax + f * dx - px, ay + f * dy - py) > tol) continue;
              const len = Math.sqrt(len2) * (o.cost || 1);
              const u = this.key(o, i), v = this.key(o, i + 1);
              link(k, u, f * len, o.id); link(u, k, f * len, o.id);
              link(k, v, (1 - f) * len, o.id); link(v, k, (1 - f) * len, o.id);
            }
          }
        }
      }
    }

    // Nearest point on each segment within `radius` metres, closest first.
    snap(lat, lon, radius) {
      if (!this.proj) return [];
      this.build();
      const [px, py] = this.proj.toXY(lat, lon);
      const out = [];
      for (const s of this.segs.values()) {
        let best = null;
        for (let i = 0; i + 1 < s.xy.length; i++) {
          const [ax, ay] = s.xy[i], [bx, by] = s.xy[i + 1];
          const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
          const f = len2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2)) : 0;
          const qx = ax + f * dx, qy = ay + f * dy;
          const d = Math.hypot(px - qx, py - qy);
          if (d <= radius && (!best || d < best.dist)) best = { segId: s.id, i, f, x: qx, y: qy, dist: d };
        }
        if (best) {
          [best.lat, best.lon] = this.proj.toLatLon(best.x, best.y);
          best.name = s.name || '';
          out.push(best);
        }
      }
      return out.sort((a, b) => a.dist - b.dist);
    }

    // Shortest path between two snaps; returns { coords: [[lat, lon], ...],
    // roads: [names in driving order] } or null.
    route(a, b, maxDist = Infinity) {
      this.build();
      const overlay = new Map();
      const link = (u, v, w, segId) => {
        if (!overlay.has(u)) overlay.set(u, []);
        overlay.get(u).push([v, w, segId]);
      };
      const attach = (snap, tmp) => {
        const s = this.segs.get(snap.segId);
        const u = this.key(s, snap.i), v = this.key(s, snap.i + 1);
        const len = Math.hypot(s.xy[snap.i + 1][0] - s.xy[snap.i][0], s.xy[snap.i + 1][1] - s.xy[snap.i][1]) * (s.cost || 1);
        link(tmp, u, snap.f * len, s.id); link(u, tmp, snap.f * len, s.id);
        link(tmp, v, (1 - snap.f) * len, s.id); link(v, tmp, (1 - snap.f) * len, s.id);
        return len;
      };
      const len = attach(a, '#A');
      attach(b, '#B');
      if (a.segId === b.segId && a.i === b.i) {
        const w = Math.abs(a.f - b.f) * len;
        link('#A', '#B', w, a.segId); link('#B', '#A', w, a.segId);
      }

      const dist = new Map([['#A', 0]]), prev = new Map();
      const heap = new MinHeap();
      heap.push([0, '#A']);
      while (heap.size) {
        const [d, k] = heap.pop();
        if (d > (dist.get(k) ?? Infinity)) continue;
        if (k === '#B') break;
        if (d > maxDist) return null;
        for (const list of [this.adj.get(k), overlay.get(k)]) {
          if (!list) continue;
          for (const [n, w, segId] of list) {
            const nd = d + w;
            if (nd < (dist.get(n) ?? Infinity)) { dist.set(n, nd); prev.set(n, [k, segId]); heap.push([nd, n]); }
          }
        }
      }
      if (!dist.has('#B')) return null;

      const keys = ['#B'], segIds = [];
      while (keys[keys.length - 1] !== '#A') {
        const [k, segId] = prev.get(keys[keys.length - 1]);
        keys.push(k); segIds.push(segId);
      }
      keys.reverse(); segIds.reverse();
      const coords = [], roads = [];
      for (const k of keys) {
        const c = k === '#A' ? [a.lat, a.lon] : k === '#B' ? [b.lat, b.lon] : this.nodes.get(k);
        const last = coords[coords.length - 1];
        if (!last || last[0] !== c[0] || last[1] !== c[1]) coords.push(c);
      }
      for (const id of segIds) {
        const name = this.segs.get(id).name || '?';
        if (roads[roads.length - 1] !== name) roads.push(name);
      }
      return { coords, roads };
    }

    // Follow the network from a snap in travel direction `dir` (unit [east, north])
    // for up to `length` metres. At junctions keep to the same segment, then the
    // same road name, then the straightest continuation; stop at dead ends or
    // turns sharper than ~110°. Returns [[lat, lon], ...].
    extend(snap, dir, length) {
      this.build();
      const s = this.segs.get(snap.segId);
      const u = this.key(s, snap.i), v = this.key(s, snap.i + 1);
      const [ux, uy] = s.xy[snap.i], [vx, vy] = s.xy[snap.i + 1];
      const forward = (vx - ux) * dir[0] + (vy - uy) * dir[1] >= 0;
      let prevKey = forward ? u : v, curKey = forward ? v : u, segId = s.id;
      const name = s.name || '';
      const coords = [[snap.lat, snap.lon], this.nodes.get(curKey)];
      let px = snap.x, py = snap.y;
      let [cx, cy] = this.xy(this.nodes.get(curKey));
      let total = Math.hypot(cx - px, cy - py);

      while (total < length) {
        const din = Math.atan2(cy - py, cx - px);
        let best = null;
        for (const [n, w, sid] of this.adj.get(curKey) || []) {
          if (n === prevKey) continue;
          const [nx, ny] = this.xy(this.nodes.get(n));
          let turn = Math.abs(Math.atan2(ny - cy, nx - cx) - din);
          if (turn > Math.PI) turn = 2 * Math.PI - turn;
          if (turn > 1.9) continue;
          const sname = this.segs.get(sid).name || '';
          const score = turn + (sid === segId ? 0 : 0.2) + (sname === name ? 0 : 0.5);
          if (!best || score < best.score) best = { n, w, sid, nx, ny, score };
        }
        if (!best) break;
        prevKey = curKey; curKey = best.n; segId = best.sid;
        px = cx; py = cy; cx = best.nx; cy = best.ny;
        total += best.w;
        coords.push(this.nodes.get(curKey));
      }
      return coords;
    }
  }

  // Polyline helpers in a local metric frame ---------------------------------

  class Path {
    constructor(coords) {
      this.coords = coords;
      this.proj = projection(coords[0][0]);
      this.xy = coords.map((c) => this.proj.toXY(c[0], c[1]));
      this.cum = [0];
      for (let i = 1; i < this.xy.length; i++) {
        this.cum.push(this.cum[i - 1] + Math.hypot(this.xy[i][0] - this.xy[i - 1][0], this.xy[i][1] - this.xy[i - 1][1]));
      }
      this.length = this.cum[this.cum.length - 1];
    }

    // Unit direction (east, north) of the segment containing distance s.
    direction(s) {
      const n = this.xy.length;
      if (n < 2 || this.length === 0) return [0, 1];
      let i = 1;
      while (i < n - 1 && this.cum[i] < s) i++;
      while (i > 1 && this.cum[i] - this.cum[i - 1] === 0) i--;
      const dx = this.xy[i][0] - this.xy[i - 1][0], dy = this.xy[i][1] - this.xy[i - 1][1];
      const l = Math.hypot(dx, dy) || 1;
      return [dx / l, dy / l];
    }

    // Point at distance s plus a lateral offset (metres, left of travel positive).
    at(s, offset = 0) {
      s = Math.max(0, Math.min(this.length, s));
      let i = 1;
      while (i < this.xy.length - 1 && this.cum[i] < s) i++;
      const seg = this.cum[i] - this.cum[i - 1];
      const w = seg > 0 ? (s - this.cum[i - 1]) / seg : 0;
      const x = this.xy[i - 1][0] + w * (this.xy[i][0] - this.xy[i - 1][0]);
      const y = this.xy[i - 1][1] + w * (this.xy[i][1] - this.xy[i - 1][1]);
      const [ex, ey] = this.direction(s);
      const [lat, lon] = this.proj.toLatLon(x - ey * offset, y + ex * offset);
      return { lat, lon, heading: (Math.atan2(ex, ey) / RAD + 360) % 360 };
    }

    // Signed lateral distance of a point from the path at distance s (left positive).
    offsetOf(lat, lon, s) {
      const p = this.at(s);
      const [px, py] = this.proj.toXY(lat, lon), [qx, qy] = this.proj.toXY(p.lat, p.lon);
      const [ex, ey] = this.direction(s);
      return ex * (py - qy) - ey * (px - qx);
    }
  }

  return { haversine, Network, Path };
})();

if (typeof module !== 'undefined') module.exports = RoadNet;
