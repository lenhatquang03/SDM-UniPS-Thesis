#!/usr/bin/env python3
"""Summarise a wess_site_probe.py run, apply the tap-site gate, and check S0
against the earlier wess_tau_grid_v2 records. Stdlib only -- runs anywhere.

Usage:
    python sdm_unips/probes/wess/wess_site_summary.py --run <out_dir>/site_probe.jsonl
    python sdm_unips/probes/wess/wess_site_summary.py --run <out_dir>/site_probe.jsonl \\
        --v2 <path>/wess_tau_grid_v2/wess_probe.jsonl

Columns, one row per configuration (site x fixed tau or target ESS):
  tau r/n   temperature of the real / null (shuffled-map) arm
  ESS r/n   mean effective sample size (fraction of the mask); null matched to real
  Etilt     mean wess_tilt_int: energy tilt inside the interior (1.0 = inert)
  rho       median rho(E, GT curvature) over the eroded interior (null value: 0)
  albedo    median rho(E, |grad I| | curvature) -- lower is better
  cTilt r   median curv_tilt_interior of the real draw (null value: 1.0)
  cTilt n   mean curv_tilt_interior of the null draw -- the validity check
  intF      mean wess_interior_frac: share of the draw inside the eroded interior
  raw/null  mean MAE gain (deg) of the drawn pixels over a uniform draw
  net       per-scene real - null MAE gain, mean +- SE (paired)

Gate, fixed before the screen ran; applied to a --target_ess run holding S0:
  valid   |cTilt n - 1| <= --null_band
  pass    cTilt r >= S0's + --tilt_margin  AND  albedo <= S0's + --albedo_margin
  choose  the valid, passing site with the highest cTilt r; none -> keep S0
The MAE columns are reported and never used to select.
"""
import argparse
import json
import math
import statistics as st

NAN = float('nan')


def load(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict):
                recs.append(r)
    if not recs:
        raise SystemExit(f'no records in {path}')
    return recs


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float('inf')


def mean_se(vals):
    vals = [v for v in vals if finite(v)]
    if not vals:
        return NAN, NAN, 0
    se = st.pstdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else NAN
    return st.mean(vals), se, len(vals)


def median(vals):
    vals = [v for v in vals if finite(v)]
    return st.median(vals) if vals else NAN


def summarise(recs):
    names = []
    for r in recs:
        for n in r.get('configs', {}):
            if n not in names:
                names.append(n)
    rows = []
    for name in names:
        cs = [r['configs'][name] for r in recs if name in r.get('configs', {})]
        c0 = cs[0]
        has_null = any('null' in c for c in cs)

        def col(arm, key):
            return [c[arm].get(key) for c in cs if arm in c]

        net = [c['real'].get('mae_delta_vs_uniform_deg') - c['null'].get('mae_delta_vs_uniform_deg')
               for c in cs if 'null' in c
               and finite(c['real'].get('mae_delta_vs_uniform_deg'))
               and finite(c['null'].get('mae_delta_vs_uniform_deg'))]
        rows.append(dict(
            name=name, site=c0['site'], mode=c0['mode'], n=len(cs),
            tau_r=c0.get('tau_real'), tau_n=c0.get('tau_null'),
            ess_r=mean_se(col('real', 'wess_ess'))[0],
            ess_n=mean_se(col('null', 'wess_ess'))[0] if has_null else NAN,
            etilt=mean_se(col('real', 'wess_tilt_int'))[0],
            rho=median([c.get('rho_E_curvature') for c in cs]),
            albedo=median([c.get('rho_E_imgrad_given_curvature') for c in cs]),
            ctilt_r=median(col('real', 'curv_tilt_interior')),
            ctilt_n=mean_se(col('null', 'curv_tilt_interior'))[0] if has_null else NAN,
            intf=mean_se(col('real', 'wess_interior_frac'))[0],
            raw=mean_se(col('real', 'mae_delta_vs_uniform_deg')),
            nullg=mean_se(col('null', 'mae_delta_vs_uniform_deg')) if has_null else (NAN, NAN, 0),
            net=mean_se(net)))
    return rows


def fmt(v, spec='.3f'):
    return format(v, spec) if finite(v) else '-'


def print_table(rows):
    print(f'{"config":18s} {"tau r":>6} {"tau n":>6} {"ESS r":>6} {"ESS n":>6} {"Etilt":>6} '
          f'{"rho":>6} {"albedo":>6} {"cTilt r":>7} {"cTilt n":>7} {"intF":>5} '
          f'{"raw":>7} {"null":>7} {"net +- SE":>15} {"n":>4}')
    for r in rows:
        net = f'{fmt(r["net"][0], "+.3f")} +- {fmt(r["net"][1])}' if r['net'][2] else '-'
        print(f'{r["name"]:18s} {fmt(r["tau_r"]):>6} {fmt(r["tau_n"]):>6} {fmt(r["ess_r"]):>6} '
              f'{fmt(r["ess_n"]):>6} {fmt(r["etilt"], ".2f"):>6} {fmt(r["rho"], "+.3f"):>6} '
              f'{fmt(r["albedo"], "+.3f"):>6} {fmt(r["ctilt_r"]):>7} {fmt(r["ctilt_n"]):>7} '
              f'{fmt(r["intf"], ".2f"):>5} {fmt(r["raw"][0], "+.3f"):>7} '
              f'{fmt(r["nullg"][0], "+.3f"):>7} {net:>15} {r["n"]:>4}')


def gate(rows, a):
    ess_rows = [r for r in rows if r['mode'] == 'ess']
    ref = next((r for r in ess_rows if r['site'] == 'S0'), None)
    print('\nGATE')
    if ref is None or len(ess_rows) < 2:
        print('  not applied: needs a --target_ess run holding S0 and at least one other site.')
        return
    ref_valid = finite(ref['ctilt_n']) and abs(ref['ctilt_n'] - 1) <= a.null_band
    print(f'  reference S0: cTilt r {fmt(ref["ctilt_r"])}, albedo {fmt(ref["albedo"], "+.3f")}, '
          f'null cTilt {fmt(ref["ctilt_n"])} ({"valid" if ref_valid else "INVALID"})')
    need_tilt = ref['ctilt_r'] + a.tilt_margin
    max_alb = ref['albedo'] + a.albedo_margin
    print(f'  pass needs: cTilt r >= {fmt(need_tilt)} and albedo <= {fmt(max_alb, "+.3f")}; '
          f'null cTilt within 1 +- {a.null_band}')
    passing = []
    for r in ess_rows:
        if r['site'] == 'S0':
            continue
        valid = finite(r['ctilt_n']) and abs(r['ctilt_n'] - 1) <= a.null_band
        tilt_ok = finite(r['ctilt_r']) and r['ctilt_r'] >= need_tilt
        alb_ok = finite(r['albedo']) and r['albedo'] <= max_alb
        why = []
        if not valid:
            why.append('null invalid' if finite(r['ctilt_n']) else 'no null arm')
        if not tilt_ok:
            why.append('tilt too low')
        if not alb_ok:
            why.append('albedo too high')
        verdict = 'PASS' if not why else 'fail (' + ', '.join(why) + ')'
        print(f'  {r["name"]:18s} cTilt r {fmt(r["ctilt_r"])}  albedo {fmt(r["albedo"], "+.3f")}  '
              f'null cTilt {fmt(r["ctilt_n"])}  -> {verdict}')
        if not why:
            passing.append(r)
    if not ref_valid:
        print('  ** S0 null arm is invalid: the pipeline did not collapse to its null; '
              'do not act on this gate. **')
    if passing:
        best = max(passing, key=lambda r: r['ctilt_r'])
        print(f'\n  DECISION: train {best["site"]} ({best["name"]}); '
              f'{len(passing)} site(s) passed.')
    else:
        print('\n  DECISION: no site passed -- keep S0; report the screen as a negative ablation.')


def regress(recs, v2_path, a):
    v2 = {r['scene']: r for r in load(v2_path)}
    diffs = {'mae_uniform_deg': [], 'rho_E_curvature': [], 'rho_E_imgrad_given_curvature': [],
             'wess_ess (tau=1)': []}
    matched = 0
    for r in recs:
        cfg = next((c for c in r.get('configs', {}).values()
                    if c['site'] == 'S0' and c['mode'] == 'tau' and abs(c['tau_real'] - 1.0) < 1e-9), None)
        old = v2.get(r.get('scene'))
        if cfg is None or old is None:
            continue
        p1 = next((p for p in old.get('per_tau', []) if abs(p['tau'] - 1.0) < 1e-9), None)
        if p1 is None:
            continue
        matched += 1
        for key, new, prev in [
                ('mae_uniform_deg', r.get('mae_uniform_deg'), old.get('mae_uniform_deg')),
                ('rho_E_curvature', cfg.get('rho_E_curvature'), old.get('rho_E_curvature')),
                ('rho_E_imgrad_given_curvature', cfg.get('rho_E_imgrad_given_curvature'),
                 old.get('rho_E_imgrad_given_curvature')),
                ('wess_ess (tau=1)', cfg['real'].get('wess_ess'), p1.get('wess_ess'))]:
            if finite(new) and finite(prev):
                diffs[key].append(abs(new - prev))
    print('\nREGRESSION: S0|tau=1 vs wess_tau_grid_v2, scene by scene')
    if not matched:
        print('  no matching scenes: run wess_site_probe.py --sites S0 --tau 1.0 --lam 0.25 '
              'with the same split as the v2 run.')
        return
    if any(abs(r.get('lam', 0.25) - 0.25) > 1e-9 for r in recs):
        print('  ** WARNING: this run is not lam=0.25, which is what v2 used **')
    tol = {'mae_uniform_deg': a.tol_mae, 'rho_E_curvature': a.tol_rho,
           'rho_E_imgrad_given_curvature': a.tol_rho, 'wess_ess (tau=1)': a.tol_ess}
    ok_all = True
    for key, vals in diffs.items():
        if not vals:
            print(f'  {key:30s} no finite pairs')
            ok_all = False
            continue
        worst = max(vals)
        ok = worst <= tol[key]
        ok_all &= ok
        print(f'  {key:30s} n={len(vals):4d}  max |d| {worst:.2e}  mean |d| {st.mean(vals):.2e}  '
              f'tol {tol[key]:g}  {"OK" if ok else "FAIL"}')
    print(f'  {matched} scenes matched -> {"PASS" if ok_all else "FAIL"}  '
          '(exact zeros on the same GPU and PyTorch; small differences are expected on another box. '
          'Draw-dependent fields are not compared: the draws use different seeds.)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='site_probe.jsonl from wess_site_probe.py')
    ap.add_argument('--v2', help='wess_tau_grid_v2/wess_probe.jsonl, for the S0 regression')
    ap.add_argument('--tilt_margin', type=float, default=0.05)
    ap.add_argument('--albedo_margin', type=float, default=0.05)
    ap.add_argument('--null_band', type=float, default=0.05)
    ap.add_argument('--tol_ess', type=float, default=0.01)
    ap.add_argument('--tol_rho', type=float, default=0.01)
    ap.add_argument('--tol_mae', type=float, default=0.05)
    a = ap.parse_args()
    recs = load(a.run)
    rows = summarise(recs)
    print(f'{len(recs)} scenes, lam {recs[0].get("lam")}\n')
    print_table(rows)
    gate(rows, a)
    if a.v2:
        regress(recs, a.v2, a)


if __name__ == '__main__':
    main()
