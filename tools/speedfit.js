// Speed along a section from the camera's turning: align the road heading theta(s)
// with the heading the camera turned through, Theta(t), between two fixes.
// Dynamic programming over a time x station grid: monotone s(t) from 0 to L,
// speed between 0 and vmax (default 110 km/h, adjustable; raised to 1.5x the
// section's mean speed if that is higher, so the end is always reachable),
// penalty `mu` on deviating from the mean speed (on
// straights the heading says nothing, so the speed stays even there). The end
// mismatch of Theta (drift, scale) is spread linearly over the section.
// Cross-validated on the Samnaun drive (fixes every ~37 s): position error at
// left-out fixes median 14 -> 9 m, 90 % 38 -> 21 m compared with constant speed.
// With a speed reference (vrefAt, e.g. from the engine sound; scaled so that it
// covers exactly the section) the penalty is on deviating from that instead of
// from the mean speed; turnAt may be omitted to use the reference alone.
'use strict';

const SpeedFit = (() => {
  function fitSection({ duration, length, thetaAt, turnAt, vrefAt = null, dt = 0.2, ds = 1.0, vmax = 110 / 3.6, mu = 300 }) {
    const N = Math.max(1, Math.round(duration / dt));
    const M = Math.max(1, Math.ceil(length / ds));
    const cell = length / M;
    const th = new Float64Array(M + 1);
    const th0 = thetaAt(0);
    for (let j = 0; j <= M; j++) th[j] = thetaAt(j * cell) - th0;
    const vid = new Float64Array(N + 1);
    if (turnAt) {
      const v0 = turnAt(0), v1 = turnAt(duration), endFix = isFinite(v0) && isFinite(v1) ? th[M] - (v1 - v0) : NaN;
      for (let i = 0; i <= N; i++) vid[i] = turnAt(i * duration / N) - v0 + endFix * i / N;
    }
    // expected cells per time step: the mean, or the speed reference scaled to the section
    const kref = new Float64Array(N + 1).fill(M / N);
    if (vrefAt) {
      let sum = 0;
      for (let i = 1; i <= N; i++) { const v = vrefAt((i - 0.5) * duration / N); kref[i] = isFinite(v) && v > 0 ? v : NaN; }
      const known = [...kref.slice(1)].filter(isFinite);
      const fill = known.length ? known.reduce((a, b) => a + b, 0) / known.length : 1;
      for (let i = 1; i <= N; i++) { if (!isFinite(kref[i])) kref[i] = fill; sum += kref[i]; }
      for (let i = 1; i <= N; i++) kref[i] *= M / sum;
    }
    const vlim = Math.max(vmax, 1.5 * length / duration);   // never below what the section needs
    let kmax = Math.max(1, Math.ceil(vlim * duration / N / cell));
    if (vrefAt) for (let i = 1; i <= N; i++) kmax = Math.max(kmax, Math.ceil(1.2 * kref[i]));
    kmax = Math.min(255, kmax);
    let cost = new Float64Array(M + 1).fill(Infinity);
    cost[0] = 0;
    const back = new Uint8Array((N + 1) * (M + 1));
    let next = new Float64Array(M + 1);
    for (let i = 1; i <= N; i++) {
      next.fill(Infinity);
      const row = i * (M + 1);
      const ref = kref[i];
      for (let k = 0; k <= kmax; k++) {
        const pen = mu * (k - ref) * (k - ref);
        for (let j = k; j <= M; j++) {
          const c = cost[j - k] + pen;
          if (c < next[j]) { next[j] = c; back[row + j] = k; }
        }
      }
      if (turnAt && isFinite(vid[i])) for (let j = 0; j <= M; j++) { const d = th[j] - vid[i]; next[j] += d * d; }   // no turn data: reference alone
      [cost, next] = [next, cost];
    }
    const steps = new Int32Array(N + 1);
    let j = M;
    for (let i = N; i >= 0; i--) { steps[i] = j; if (i > 0) j -= back[i * (M + 1) + j]; }
    // light smoothing (5 points), ends fixed
    const times = new Float64Array(N + 1), stations = new Float64Array(N + 1);
    for (let i = 0; i <= N; i++) {
      let sum = 0;
      for (let q = -2; q <= 2; q++) sum += steps[Math.min(N, Math.max(0, i + q))];
      times[i] = i * duration / N;
      stations[i] = sum / 5 * cell;
    }
    stations[0] = 0; stations[N] = length;
    for (let i = 1; i <= N; i++) if (stations[i] < stations[i - 1]) stations[i] = stations[i - 1];
    return { times, stations };
  }

  // station and speed at time t (seconds since the section start)
  function at(timing, t) {
    const { times, stations } = timing, n = times.length - 1;
    if (t <= 0) return { s: 0, v: n ? (stations[1] - stations[0]) / (times[1] - times[0]) : 0 };
    if (t >= times[n]) return { s: stations[n], v: (stations[n] - stations[n - 1]) / (times[n] - times[n - 1]) };
    const i = Math.min(n - 1, Math.floor(t / times[n] * n));
    const w = (t - times[i]) / (times[i + 1] - times[i]);
    const i0 = Math.max(0, i - 2), i1 = Math.min(n, i + 3);   // speed over ~1 s
    return { s: stations[i] + w * (stations[i + 1] - stations[i]),
             v: (stations[i1] - stations[i0]) / (times[i1] - times[i0]) };
  }

  return { fitSection, at };
})();

if (typeof module !== 'undefined') module.exports = SpeedFit;
