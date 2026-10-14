"""Fixed shared cohort at moderate severities (25/50/70%), tree + white-box DTA, 30 seeds. Run from repo root."""
import os, sys, importlib.util, numpy as np, polars as pl
os.environ['FRACS']='0.25,0.5,0.7'
spec=importlib.util.spec_from_file_location('rc','defect_expansion_experiment/robustness_checks.py'); rc=importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
s3=pl.DataFrame([rc.cell_joint(sd) for sd in rc.SEEDS]); s3.write_csv('defect_expansion_experiment/robustness_joint_cohort_moderate.csv')
base=s3['random_0.0'].to_numpy(); ci=lambda a:1.96*a.std(ddof=1)/np.sqrt(len(a))
print(f'TREE + white-box, 30 seeds, shared cohort {s3["n_cohort"].mean():.1f}/30')
for f in rc.FRACS[1:]:
    r=s3[f'random_{f}'].to_numpy()/base; sp=s3[f'spatial_{f}'].to_numpy()/base; dd=sp-r
    print(f'  frac {f:.2f} random {r.mean():.3f}x  spatial {sp.mean():.3f}x  paired spatial-random {dd.mean():+.3f} ±{ci(dd):.3f}  d {rc.cohens_d(sp,r):+.2f}')
