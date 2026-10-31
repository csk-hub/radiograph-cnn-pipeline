"""R1-9, R2-g: aggregate seeds.  Usage: python aggregate.py results [--csv summary.csv]
Groups results/*.json by args.name; reports mean +- SD and 95% t-CI over seeds."""
import sys, glob, json, argparse
import numpy as np, pandas as pd
from scipy import stats
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--csv", default="summary.csv"); a = ap.parse_args()
rows = []
for f in glob.glob(f"{a.dir}/*.json"):
    r = json.load(open(f))
    if "test_at_0.5" not in r: continue
    m = r["test_at_0.5"]
    rows.append(dict(name=r["args"]["name"], seed=r["args"]["seed"], acc=m["accuracy"], sens=m["sensitivity"],
                     spec=m["specificity"], auc=m["roc_auc"], f1=m["f1"], mcc=m["mcc"], FN=m["FN"], FP=m["FP"],
                     gap_acc=r.get("generalisation_gap_acc"), params=r["params"], macs=r.get("macs"),
                     lat_ms=(r.get("latency_ms") or {}).get("median"), epochs_run=len(r["history"]["loss"]),
                     t_epoch=r.get("epoch_time_median_excl_first")))
df = pd.DataFrame(rows)
out = []
for name, g in df.groupby("name"):
    n = len(g); d = {"name": name, "n_seeds": n, "params": int(g.params.iloc[0]),
                     "MACs(M)": None if g.macs.isna().all() else round(g.macs.iloc[0] / 1e6, 1)}
    for k in ["acc", "sens", "spec", "auc", "f1", "mcc", "gap_acc"]:
        v = g[k].dropna().values
        if len(v) == 0: continue
        sd = v.std(ddof=1) if len(v) > 1 else float("nan")
        h = stats.t.ppf(.975, len(v) - 1) * sd / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        d[k] = f"{v.mean():.4f} +- {sd:.4f} (CI +-{h:.4f})"
    d["FN per seed"] = ",".join(map(str, g.sort_values("seed").FN)); d["FP mean"] = round(g.FP.mean(), 1)
    d["lat_ms"] = None if g.lat_ms.isna().all() else round(g.lat_ms.mean(), 2)
    out.append(d)
res = pd.DataFrame(out); res.to_csv(a.csv, index=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 40)
print(res.T)
