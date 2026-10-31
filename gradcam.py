"""R1-10: Grad-CAM for the fracture score s = 1 - p on false positives, false negatives and random TP/TN.
Works for the shallow CNN and ResNet18 (last Conv2D layer found automatically).

  python gradcam.py --model results/shallow_orig_seed1.keras --preds results/shallow_orig_seed1_preds.npz \
                    --data_root <root> [--letterbox] [--size 150]
"""
import argparse, os, numpy as np
from common import *
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--preds", required=True)
ap.add_argument("--data_root", default=DEFAULT_ROOT); ap.add_argument("--size", type=int, default=150)
ap.add_argument("--letterbox", action="store_true"); ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--n_random", type=int, default=8); ap.add_argument("--out", default="gradcam_out")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
import tensorflow as tf, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
model = tf.keras.models.load_model(a.model)
conv = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)][-1]
gm = tf.keras.Model(model.inputs, [conv.output, model.output])
z = np.load(a.preds, allow_pickle=True); y, s, k = z["test_ypos"], z["test_score"], z["test_keys"]
pred = (s >= a.thr).astype(int)
groups = {"False positive": np.where((pred == 1) & (y == 0))[0], "False negative": np.where((pred == 0) & (y == 1))[0]}
rng = np.random.default_rng(0)
groups["True positive"] = rng.choice(np.where((pred == 1) & (y == 1))[0], a.n_random, replace=False)
groups["True negative"] = rng.choice(np.where((pred == 0) & (y == 0))[0], a.n_random, replace=False)

def cam(img):
    x = tf.convert_to_tensor(img[None].astype("float32"))
    with tf.GradientTape() as t:
        A, p = gm(x); score = 1.0 - p[:, 0]                      # fracture score
    g = t.gradient(score, A); w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
    c = tf.nn.relu(tf.reduce_sum(w * A, -1))[0].numpy()
    c = c / (c.max() + 1e-8)
    return np.array(tf.image.resize(c[..., None], (a.size, a.size)))[..., 0]

root = os.path.join(a.data_root, "test")
for gname, idx in groups.items():
    if len(idx) == 0: continue
    n = len(idx); cols = min(n, 6); rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(2 * rows, cols, figsize=(2.2 * cols, 4.6 * rows)); ax = np.atleast_2d(ax)
    for ax_ in ax.flat: ax_.axis("off")
    for j, i in enumerate(idx):
        r, c = divmod(j, cols)
        _, cls, fn = k[i].split("/", 2)
        img = load_image(os.path.join(root, cls, fn), a.size, a.letterbox)
        ax[2 * r, c].imshow(img); ax[2 * r, c].set_title(f"s={s[i]:.2f}", fontsize=7)
        ax[2 * r + 1, c].imshow(img); ax[2 * r + 1, c].imshow(cam(img), cmap="jet", alpha=.45)
    plt.suptitle(f"{gname} (n={n})  -  rows: image / Grad-CAM"); plt.tight_layout()
    plt.savefig(os.path.join(a.out, gname.lower().replace(" ", "_") + ".png"), dpi=150); plt.close()
    if gname == "False positive":
        open(os.path.join(a.out, "false_positive_keys.txt"), "w").write("\n".join(k[idx]))     # give to the clinician
print("saved to", a.out, "| last conv layer used:", conv.name, "feature map", conv.output.shape[1:3])
