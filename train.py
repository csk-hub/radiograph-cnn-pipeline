"""Train one configuration (one seed) and save everything needed by the other scripts.

Examples (Kaggle, 2xT4):
  python train.py --name shallow_orig --seed 1                       # paper configuration
  python train.py --name shallow_gap1 --head gap1 --seed 1           # R1-8
  python train.py --name aug_none --aug none --seed 1                # R2-d ablation
  python train.py --name mnv2_pre --model mobilenetv2 --pretrained --lr 1e-4 --seed 1   # R1-7
  python train.py --name es --early_stop 4 --reduce_lr --epochs 30   # R2-g
"""
import argparse, os, json, time
import numpy as np
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--model", default="shallow", choices=["shallow", "mobilenetv2", "efficientnetb0", "resnet18"])
ap.add_argument("--head", default="flatten256", choices=["flatten256", "flatten64", "gap256", "gap1"])
ap.add_argument("--pretrained", action="store_true")
ap.add_argument("--size", type=int, default=150)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--batch", type=int, default=256, help="GLOBAL batch size")
ap.add_argument("--gpus", type=int, default=2)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--dropout", type=float, default=0.5)
ap.add_argument("--l2", type=float, default=0.0)
ap.add_argument("--aug", default="rot,zoom,flip", help="comma list of rot,zoom,flip or 'none'")
ap.add_argument("--letterbox", action="store_true", help="aspect-preserving resize with padding")
ap.add_argument("--no_truncated", action="store_true", help="skip files that fail strict decode")
ap.add_argument("--exclude", default=None, help="text file of keys to drop (from audit)")
ap.add_argument("--early_stop", type=int, default=0, help="patience on val_loss (0=off)")
ap.add_argument("--reduce_lr", action="store_true")
ap.add_argument("--deterministic", action="store_true")
ap.add_argument("--bench", action="store_true", help="timing only: no eval/save")
ap.add_argument("--data_root", default=DEFAULT_ROOT)
ap.add_argument("--out", default="results")
ap.add_argument("--cache", default="cache")
a = ap.parse_args()

import tensorflow as tf
tf.keras.utils.set_random_seed(a.seed)
if a.deterministic:
    tf.config.experimental.enable_op_determinism()
os.makedirs(a.out, exist_ok=True)

excl = set(open(a.exclude).read().split("\n")) - {""} if a.exclude else set()
kw = dict(size=a.size, letterbox=a.letterbox, truncated_ok=not a.no_truncated, cache_dir=a.cache)
tr = load_split(a.data_root, "train", exclude=excl if a.exclude else None, **kw)
va = load_split(a.data_root, "val", **kw)
te = load_split(a.data_root, "test", **kw)
print({k: len(v["y"]) for k, v in dict(train=tr, val=va, test=te).items()},
      "skipped:", {k: len(v["skipped"]) for k, v in dict(train=tr, val=va, test=te).items()})

# ---- augmentation (applied on the fly, training only) -----------------------
augs = [] if a.aug == "none" else a.aug.split(",")
aug_layers = []
if "rot" in augs:  aug_layers.append(tf.keras.layers.RandomRotation(15 / 360, fill_mode="nearest"))
if "zoom" in augs: aug_layers.append(tf.keras.layers.RandomZoom(0.1, fill_mode="nearest"))
if "flip" in augs: aug_layers.append(tf.keras.layers.RandomFlip("horizontal"))
augment = tf.keras.Sequential(aug_layers) if aug_layers else None

def make_ds(d, train):
    ds = tf.data.Dataset.from_tensor_slices((d["X"], d["y"].astype("float32")))
    if train:
        ds = ds.shuffle(len(d["y"]), seed=a.seed, reshuffle_each_iteration=True)
    ds = ds.batch(a.batch).map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    if train and augment is not None:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

# ---- strategy / model -------------------------------------------------------
devs = [f"/gpu:{i}" for i in range(a.gpus)]
strategy = tf.distribute.MirroredStrategy(devices=devs) if a.gpus > 0 else tf.distribute.get_strategy()
print("replicas in sync:", strategy.num_replicas_in_sync)
with strategy.scope():
    model = build_model(a.model, a.size, a.head, a.dropout, a.lr, a.l2, a.pretrained)

class EpochTimer(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None): self.t, self.times = None, []
    def on_epoch_begin(self, e, logs=None): self.t = time.perf_counter()
    def on_epoch_end(self, e, logs=None): self.times.append(time.perf_counter() - self.t)
timer = EpochTimer()
cbs = [timer]
if a.early_stop: cbs.append(tf.keras.callbacks.EarlyStopping("val_loss", patience=a.early_stop, restore_best_weights=True))
if a.reduce_lr: cbs.append(tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=2, min_lr=1e-6))

t0 = time.perf_counter()
hist = model.fit(make_ds(tr, True), validation_data=make_ds(va, False), epochs=a.epochs,
                 callbacks=cbs, verbose=2)
total = time.perf_counter() - t0

peak = {}
for i in range(a.gpus):
    try: peak[f"gpu{i}_MB"] = tf.config.experimental.get_memory_info(f"GPU:{i}")["peak"] / 2**20
    except Exception: pass
et = timer.times
rec = dict(args=vars(a), n=dict(train=len(tr["y"]), val=len(va["y"]), test=len(te["y"])),
           skipped=dict(train=tr["skipped"], val=va["skipped"], test=te["skipped"]),
           epoch_times=et, total_train_s=total,
           epoch_time_median_excl_first=float(np.median(et[1:])) if len(et) > 1 else None,
           throughput_img_s=(len(tr["y"]) / float(np.median(et[1:]))) if len(et) > 1 else None,
           replicas=strategy.num_replicas_in_sync, peak_gpu_mem=peak,
           params=int(model.count_params()),
           history={k: [float(x) for x in v] for k, v in hist.history.items()})
fname = f"{a.name}_seed{a.seed}"

if not a.bench:
    rec["macs"] = count_macs(model, a.size)
    rec["latency_ms"] = latency_ms(model, a.size)
    # inference-mode train/val/test evaluation (no dropout, no augmentation) -> unbiased gap
    plain = lambda d: tf.data.Dataset.from_tensor_slices((d["X"], d["y"].astype("float32"))).batch(a.batch).map(
        lambda x, y: (tf.cast(x, tf.float32), y)).prefetch(tf.data.AUTOTUNE)
    ev = {}
    for nm, d in dict(train=tr, val=va, test=te).items():
        l, acc = model.evaluate(plain(d), verbose=0); ev[nm] = dict(loss=float(l), acc=float(acc))
    rec["eval_inference_mode"] = ev
    rec["generalisation_gap_acc"] = ev["train"]["acc"] - ev["val"]["acc"]
    if a.early_stop:
        rec["best_epoch"] = int(np.argmin(hist.history["val_loss"]) + 1)
    out = {}
    for nm, d in dict(val=va, test=te).items():
        p = model.predict(plain(d), verbose=0).reshape(-1)          # p = P(not fractured)
        out[f"{nm}_score"] = 1.0 - p                                 # fracture score
        out[f"{nm}_ypos"] = (d["y"] == 0).astype(int)
        out[f"{nm}_keys"] = np.array(d["keys"])
    np.savez(os.path.join(a.out, fname + "_preds.npz"), **out)
    rec["test_at_0.5"] = metrics(out["test_ypos"], out["test_score"], 0.5)
    rec["val_at_0.5"] = metrics(out["val_ypos"], out["val_score"], 0.5)
    model.save(os.path.join(a.out, fname + ".keras"))
    print("TEST @0.5:", {k: round(v, 4) if isinstance(v, float) else v for k, v in rec["test_at_0.5"].items()})

json.dump(rec, open(os.path.join(a.out, fname + ".json"), "w"), indent=1)
