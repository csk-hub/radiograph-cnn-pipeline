"""R1-3, R1-6: extended metrics with bootstrap CIs, ROC/PR curves, and validation-based threshold selection.

  python 01_metrics_and_threshold.py --preds results/shallow_orig_seed1_preds.npz \
         [--exclude audit/exclude_test.txt] [--sens_target 0.99] [--cost_fn 10]

Needs only the saved preds (fracture score s = 1 - p).  The threshold is chosen on the
VALIDATION scores only and then applied once to the test scores.
"""
import argparse, os, json
import numpy as np
from common import metrics, bootstrap_ci, cp_ci, confusion

ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
ap.add_argument("--exclude", default=None, help="test keys to drop (duplicates)")
ap.add_argument("--sens_target", type=float, default=0.99)
ap.add_argument("--cost_fn", type=float, default=10.0, help="cost of a missed fracture relative to a false alarm")
ap.add_argument("--out", default="metrics_out")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
z = np.load(a.preds, allow_pickle=True)
vy, vs = z["val_ypos"], z["val_score"]
ty, ts, tk = z["test_ypos"], z["test_score"], z["test_keys"]
if a.exclude:
    drop = set(open(a.exclude).read().split("\n")); keep = np.array([k not in drop for k in tk])
    print(f"excluding {int((~keep).sum())} test images listed in {a.exclude}")
    ty, ts = ty[keep], ts[keep]

# ---- operating points chosen on validation ---------------------------------
cand = np.unique(np.concatenate([vs, [0.0, 0.5, 1.0]]))
def val_stats(t):
    tp, fn, fp, tn = confusion(vy, vs, t); return tp, fn, fp, tn
ops = {"default_0.5": 0.5}
ok = [t for t in cand if val_stats(t)[0] / max(1, val_stats(t)[0] + val_stats(t)[1]) >= a.sens_target]
ops[f"val_sens>={a.sens_target:.2f}"] = float(max(ok)) if ok else 0.0
youden = [(val_stats(t)[0] / max(1, sum(val_stats(t)[:2])) + val_stats(t)[3] / max(1, sum(val_stats(t)[2:])) - 1, t) for t in cand]
ops["val_youden"] = float(max(youden)[1])
cost = [(a.cost_fn * val_stats(t)[1] + val_stats(t)[2], -t) for t in cand]
ops[f"val_cost_FN:FP={a.cost_fn:g}:1"] = float(-min(cost)[1])

res = {}
print(f"\nTEST n={len(ty)} (pos={int(ty.sum())}, neg={int((1-ty).sum())})")
print(f"{'operating point':28s} thr    sens [95% CI]          spec [95% CI]         FP/1000 normals  AUC")
for nm, t in ops.items():
    m = metrics(ty, ts, t); ci = bootstrap_ci(ty, ts, t)
    sci = cp_ci(m["TP"], m["TP"] + m["FN"]); pci = cp_ci(m["TN"], m["TN"] + m["FP"])
    res[nm] = dict(threshold=t, **m, boot_ci=ci, sens_cp_ci=sci, spec_cp_ci=pci)
    print(f"{nm:28s} {t:.3f}  {m['sensitivity']:.4f} [{sci[0]:.3f},{sci[1]:.3f}]  "
          f"{m['specificity']:.4f} [{pci[0]:.3f},{pci[1]:.3f}]  {1000*(1-m['specificity']):7.1f}          {m['roc_auc']:.4f}")
m = res["default_0.5"]
print("\nFull metrics at 0.5 (bootstrap 95% CI):")
for k in ["sensitivity", "specificity", "ppv", "npv", "accuracy", "f1", "balanced_acc", "mcc", "roc_auc", "pr_auc"]:
    lo, hi = m["boot_ci"][k]; print(f"  {k:13s} {m[k]:.4f}  [{lo:.4f}, {hi:.4f}]")
print(f"  confusion (rows true fractured/normal): TP={m['TP']} FN={m['FN']} FP={m['FP']} TN={m['TN']}")
json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
fpr, tpr, _ = roc_curve(ty, ts); ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "k--", lw=.7)
ax[0].set(xlabel="1 - specificity", ylabel="Sensitivity", title=f"ROC (AUC={m['roc_auc']:.3f})")
pr, rc, _ = precision_recall_curve(ty, ts); ax[1].plot(rc, pr)
ax[1].set(xlabel="Recall (sensitivity)", ylabel="Precision", title=f"PR (AP={m['pr_auc']:.3f})")
for nm, t in ops.items():
    mm = res[nm]; ax[0].plot(1 - mm["specificity"], mm["sensitivity"], "o", label=nm)
ax[0].legend(fontsize=6); plt.tight_layout(); plt.savefig(os.path.join(a.out, "roc_pr.png"), dpi=200)
