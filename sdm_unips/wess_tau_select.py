#!/usr/bin/env python3
"""Score a WESS tau grid against the pre-registered selection rule.

Stdlib only -- runs anywhere, including a box with no torch. Reads the JSONL
`wess_probe.py` writes for the real arm and for the `--shuffle_energy` arm and
reports, per tau:

    net_gain = mae_delta(real) - mae_delta(shuffled)

the MAE gap attributable to E's *alignment with geometry*, with the clustering
artifact measured by the falsification control subtracted at the matched
configuration. Selecting on raw `mae_delta` would reward peakedness itself: a
draw at ESS 0.100 costs +0.162 deg on randomly-placed clusters carrying no
geometric content at all (control of 2026-09-07, n=431).

The rule, fixed before the grid ran:

  1. keep tau with wess_ess >= --min_ess AND wess_tilt_int >= --min_tilt
  2. among survivors maximise net_gain
  3. ties within --tie_se standard errors -> take the LARGER tau

Usage:
    python sdm_unips/wess_tau_select.py \
        --real ~/result_overview/wess_tau_grid/wess_probe.jsonl \
        --shuffled ~/result_overview/wess_tau_grid_shuffled/wess_probe.jsonl
"""
import argparse, json, math, statistics as st


def load(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue          # console banners share the file in some runs
            if isinstance(r, dict) and 'per_tau' in r:
                recs.append(r)
    if not recs:
        raise SystemExit(f'no probe records in {path}')
    return recs


def by_tau(recs, key):
    out = {}
    for r in recs:
        for p in r['per_tau']:
            v = p.get(key)
            if v is None or v != v or abs(v) == float('inf'):
                continue
            out.setdefault(round(float(p['tau']), 6), []).append(float(v))
    return out


def mean_se(v):
    m = st.mean(v)
    se = st.pstdev(v) / math.sqrt(len(v)) if len(v) > 1 else float('nan')
    return m, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', required=True)
    ap.add_argument('--shuffled', required=True)
    ap.add_argument('--min_ess', type=float, default=0.25)
    ap.add_argument('--min_tilt', type=float, default=1.15)
    ap.add_argument('--tie_se', type=float, default=1.0)
    a = ap.parse_args()

    real, sh = load(a.real), load(a.shuffled)

    # Integrity gate. `uniform_sample` runs before the shuffle consumes the
    # generator, so the two arms' uniform draws must be bit-identical. If they
    # are not, the arms are not the same checkpoint or scene list and no
    # difference between them means anything.
    bys = {r['scene']: r for r in sh}
    common = [r for r in real if r['scene'] in bys]
    if not common:
        raise SystemExit('arms share no scenes -- different manifests?')
    dm = max(abs(r['mae_uniform_deg'] - bys[r['scene']]['mae_uniform_deg'])
             for r in common)
    print(f'integrity gate: {len(common)} shared scenes, '
          f'max |d mae_uniform| = {dm:.3e}', end='  ')
    if dm > 1e-9:
        print('** FAILED -- arms are not comparable, stopping **')
        raise SystemExit(2)
    print('OK')
    if not all(r.get('shuffled') for r in sh):
        print('** WARNING: --shuffled arm has records without "shuffled": true')
    if any(r.get('shuffled') for r in real):
        print('** WARNING: --real arm contains shuffled records')

    # The sampler gate. `wess_sample` (Phase-0) and `wess_sample_train`
    # (shipped) are different distributions: the shipped one excludes the
    # silhouette ring from E's statistics and from the softmax, which removes
    # what was inflating e.std() and makes the same tau peakier. A tau chosen
    # against the probe path does not transfer to training, so refuse to name
    # one unless the run used --shipped_sampler.
    shipped = (all(r.get('shipped_sampler') for r in real)
               and all(r.get('shipped_sampler') for r in sh))
    if not shipped:
        print('\n** These records were NOT produced with --shipped_sampler. **\n'
              '   They describe `wess_sample`, not the sampler B2 trains with,\n'
              '   so any tau read off them is for a distribution that will\n'
              '   never be trained. Re-run both arms with --shipped_sampler.\n'
              '   Reporting the table for reference; NO selection is made.')

    dr = by_tau(real, 'mae_delta_vs_uniform_deg')
    ds = by_tau(sh, 'mae_delta_vs_uniform_deg')
    ess = by_tau(real, 'wess_ess')
    tilt = by_tau(real, 'wess_tilt_int')
    if not tilt:
        tilt = by_tau(real, 'wess_tilt')
        print('note: no wess_tilt_int in these records, falling back to '
              'wess_tilt (full mask, inflated by the rim -- see below)')

    taus = sorted(set(dr) & set(ds))
    print(f'\n{"tau":>5} {"ESS":>7} {"tilt":>8} {"raw gain":>10} {"null":>8} '
          f'{"net gain":>10} {"SE":>7}  keep')
    rows = []
    for t in taus:
        m_r, se_r = mean_se(dr[t])
        m_s, se_s = mean_se(ds[t])
        net = m_r - m_s
        se = math.sqrt(se_r ** 2 + se_s ** 2)
        e = st.mean(ess.get(t, [float('nan')]))
        ti = st.mean(tilt.get(t, [float('nan')]))
        # A scene whose eroded interior is too thin falls back to a uniform
        # draw inside wess_train_probabilities, reading ESS exactly 1.0. Those
        # scenes dilute every mean here, so count them rather than let them
        # quietly inflate the ESS column.
        fb = sum(1 for v in ess.get(t, []) if v > 0.999)
        keep = e >= a.min_ess and ti >= a.min_tilt
        rows.append((t, net, se, e, ti, keep))
        print(f'{t:>5.2f} {e:>7.3f} {ti:>8.3f} {m_r:>+10.3f} {m_s:>+8.3f} '
              f'{net:>+10.3f} {se:>7.3f}  {"yes" if keep else "no"}'
              + (f'   [{fb} uniform-fallback scenes]' if fb else ''))

    ok = [r for r in rows if r[5]]
    print(f'\nrule: ESS >= {a.min_ess}, tilt >= {a.min_tilt}, max net gain, '
          f'ties within {a.tie_se} SE -> larger tau')
    if not ok:
        print('NO TAU SURVIVES. Do not launch on a grid point that failed the '
              'rule -- widen the grid and re-run instead.')
        return
    best = max(ok, key=lambda r: r[1])
    tied = [r for r in ok if r[1] >= best[1] - a.tie_se * best[2]]
    pick = max(tied, key=lambda r: r[0])
    if len(tied) > 1:
        print('tied within %g SE: %s -> larger tau wins'
              % (a.tie_se, ', '.join('%g' % r[0] for r in sorted(tied))))
    if not shipped:
        print(f'\nwould select tau {pick[0]:g} -- WITHHELD, wrong sampler '
              '(re-run with --shipped_sampler)')
        raise SystemExit(3)
    print(f'\nSELECTED  --wess_tau {pick[0]:g}  --wess_lam 0.25'
          f'   (net gain {pick[1]:+.3f} +- {pick[2]:.3f} deg, '
          f'ESS {pick[3]:.3f}, tilt {pick[4]:.3f})')


if __name__ == '__main__':
    main()
