"""R1-5: controlled 1-GPU vs 2-GPU benchmark (calls train.py --bench in subprocesses).

Regime A (strong scaling): global batch 256 on 1 GPU vs 2 GPUs.
Regime B (weak scaling):   128 per GPU  -> 1 GPU (128) vs 2 GPUs (256).
Reports epoch time, throughput, peak memory per GPU, speedup S = T1/T2, efficiency E = S/2.
"""
import subprocess, json, glob, os, sys, numpy as np, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--extra", default="", help="extra args for train.py")
a = ap.parse_args()
cfgs = [("A_1gpu_b256", 1, 256), ("A_2gpu_b256", 2, 256), ("B_1gpu_b128", 1, 128), ("B_2gpu_b256", 2, 256)]
for rep in range(a.reps):
    for name, g, b in cfgs:       # same seed per rep -> identical init and data order
        nm = f"bench_{name}_rep{rep}"
        cmd = f"{sys.executable} train.py --name {nm} --gpus {g} --batch {b} --epochs {a.epochs} --seed {rep+1} --bench --out bench_results {a.extra}"
        print(">>", cmd); subprocess.run(cmd, shell=True, check=True)
R = {}
for f in glob.glob("bench_results/*.json"):
    r = json.load(open(f)); k = r["args"]["name"].split("_rep")[0].replace("bench_", "")
    R.setdefault(k, []).append(r)
def stat(k, key): v = [r[key] for r in R[k]]; return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
print(f"\n{'config':14s} {'epoch s (med excl. 1st)':>24s} {'img/s':>10s} {'peak MB/GPU0':>14s}")
for k in sorted(R):
    t, ts = stat(k, "epoch_time_median_excl_first"); th, ths = stat(k, "throughput_img_s")
    mem = np.mean([r["peak_gpu_mem"].get("gpu0_MB", np.nan) for r in R[k]])
    print(f"{k:14s} {t:10.2f} +- {ts:5.2f} {th:12.1f} {mem:12.0f}")
for reg in "AB":
    one = [k for k in R if k.startswith(f"{reg}_1gpu")][0]; two = [k for k in R if k.startswith(f"{reg}_2gpu")][0]
    t1, t2 = stat(one, "epoch_time_median_excl_first")[0], stat(two, "epoch_time_median_excl_first")[0]
    th1, th2 = stat(one, "throughput_img_s")[0], stat(two, "throughput_img_s")[0]
    S = (t1 / t2) if reg == "A" else (th2 / th1)      # weak scaling: throughput ratio
    print(f"Regime {reg}: speedup S = {S:.2f}, parallel efficiency E = S/2 = {100*S/2:.0f}%")
