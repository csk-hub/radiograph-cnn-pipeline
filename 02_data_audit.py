"""R1-2, R1-4, R2-c: dataset characterisation, truncated-image count, duplicate / leakage audit.

Outputs (in --out):
  manifest.csv         one row per file: split, class, size, mode, decode status, md5, pHash
  summary_counts.csv   class counts per split, image-size / mode statistics
  truncated.txt        keys of files that fail a STRICT decode
  cross_split_pairs.csv exact + near duplicates across splits (with label conflicts flagged)
  exclude_test.txt     test keys with an exact/near duplicate in train or val   -> train.py --exclude (for train side use exclude_train.txt)
  exclude_train.txt    train keys that duplicate a val/test image
  pairs_montage.png    closest non-identical pairs for visual confirmation
"""
import argparse, os, hashlib, csv
import numpy as np
from PIL import Image, ImageFile
from scipy.fft import dctn
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", default=DEFAULT_ROOT)
ap.add_argument("--out", default="audit")
ap.add_argument("--max_hamming", type=int, default=4, help="pHash distance (of 64 bits) for 'near duplicate'")
ap.add_argument("--embed", action="store_true", help="also cosine-similarity of MobileNetV2 features")
ap.add_argument("--cos", type=float, default=0.98)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

def phash(img):
    g = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float32)
    low = dctn(g, norm="ortho")[:8, :8].flatten()
    return (low > np.median(low[1:])).astype(np.uint8)          # 64 bits

rows, H, keys, labels, splits, paths = [], [], [], [], [], []
for split in ["train", "val", "test"]:
    for path, key, lab in list_split(a.data_root, split):
        ImageFile.LOAD_TRUNCATED_IMAGES = False                  # STRICT decode
        ok, err = True, ""
        try:
            im = Image.open(path); im.load()
        except Exception as e:
            ok, err = False, str(e)[:80]
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            im = Image.open(path); im.load()                     # tolerant decode for hashing
        md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
        h = phash(im)
        rows.append(dict(split=split, cls=CLASSES[lab], key=key, width=im.size[0], height=im.size[1],
                         mode=im.mode, filesize=os.path.getsize(path), decode_ok=ok, error=err,
                         md5=md5, phash="".join(map(str, h))))
        H.append(h); keys.append(key); labels.append(lab); splits.append(split); paths.append(path)

with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
import pandas as pd
df = pd.DataFrame(rows)
print("\n== Class counts per split ==\n", pd.crosstab(df.split, df.cls, margins=True))
print("\n== Image modes ==\n", pd.crosstab(df.split, df["mode"]))
print("\n== Size (w x h) percentiles ==\n", df[["width", "height"]].describe(percentiles=[.05, .5, .95]).round(0))
print("\n== Strict-decode failures (truncated/corrupt) ==\n", pd.crosstab([df.split, df.cls], df.decode_ok))
open(os.path.join(a.out, "truncated.txt"), "w").write("\n".join(df[~df.decode_ok].key))
pd.crosstab(df.split, df.cls, margins=True).to_csv(os.path.join(a.out, "summary_counts.csv"))

# ---- duplicates ----
H = np.stack(H).astype(np.float32); labels = np.array(labels); splits = np.array(splits); keys = np.array(keys)
md5 = df.md5.values
def hamming(A, B):  return A.sum(1)[:, None] + B.sum(1)[None, :] - 2 * A @ B.T

pairs = []
def cross(sa, sb):
    ia, ib = np.where(splits == sa)[0], np.where(splits == sb)[0]
    D = hamming(H[ia], H[ib])
    ii, jj = np.where(D <= a.max_hamming)
    for i, j in zip(ii, jj):
        A, B = ia[i], ib[j]
        pairs.append(dict(a=keys[A], b=keys[B], splits=f"{sa}-{sb}", hamming=int(D[i, j]),
                          exact_md5=bool(md5[A] == md5[B]), label_conflict=bool(labels[A] != labels[B])))
for sa, sb in [("test", "train"), ("test", "val"), ("val", "train")]:
    cross(sa, sb)
ia = np.where(splits == "test")[0]                                # within-test
D = hamming(H[ia], H[ia]); ii, jj = np.where(np.triu(D <= a.max_hamming, 1))
for i, j in zip(ii, jj):
    pairs.append(dict(a=keys[ia[i]], b=keys[ia[j]], splits="test-test", hamming=int(D[i, j]),
                      exact_md5=bool(md5[ia[i]] == md5[ia[j]]), label_conflict=bool(labels[ia[i]] != labels[ia[j]])))

if a.embed:                                                        # optional CNN-embedding check
    import tensorflow as tf
    base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", pooling="avg", input_shape=(160, 160, 3))
    feats = []
    for i in range(0, len(paths), 128):
        x = np.stack([load_image(p, 160) for p in paths[i:i + 128]]).astype("float32") / 127.5 - 1
        feats.append(base.predict(x, verbose=0))
    F = np.concatenate(feats); F /= np.linalg.norm(F, axis=1, keepdims=True)
    for sa, sb in [("test", "train"), ("test", "val")]:
        ia, ib = np.where(splits == sa)[0], np.where(splits == sb)[0]
        S = F[ia] @ F[ib].T
        for i, j in zip(*np.where(S >= a.cos)):
            pairs.append(dict(a=keys[ia[i]], b=keys[ib[j]], splits=f"{sa}-{sb}", hamming=-1,
                              exact_md5=False, label_conflict=bool(labels[ia[i]] != labels[ib[j]]),
                              cosine=float(S[i, j])))

P = pd.DataFrame(pairs); P.to_csv(os.path.join(a.out, "cross_split_pairs.csv"), index=False)
if len(P):
    cs = P[P.splits != "test-test"]
    print(f"\n== Duplicate audit (pHash Hamming <= {a.max_hamming}) ==")
    print(cs.groupby("splits").agg(pairs=("a", "size"), exact=("exact_md5", "sum"), label_conflicts=("label_conflict", "sum")))
    ex_test = sorted(set(cs[cs.splits.str.startswith("test")].a))
    ex_train = sorted(set(cs[cs.splits.isin(["val-train"])].b) | set(cs[cs.splits == "test-train"].b))
    open(os.path.join(a.out, "exclude_test.txt"), "w").write("\n".join(ex_test))
    open(os.path.join(a.out, "exclude_train.txt"), "w").write("\n".join(ex_train))
    print(f"test images with a near-duplicate in train/val: {len(ex_test)} / {(splits=='test').sum()}")
    # montage of the closest non-identical pairs
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pk = dict(zip(keys, paths)); sub = cs[~cs.exact_md5].sort_values("hamming").head(12)
    if len(sub):
        fig, ax = plt.subplots(len(sub), 2, figsize=(5, 2.5 * len(sub)))
        ax = np.atleast_2d(ax)
        for r, (_, row) in enumerate(sub.iterrows()):
            for c, k in enumerate([row.a, row.b]):
                ax[r, c].imshow(Image.open(pk[k]).convert("L"), cmap="gray"); ax[r, c].axis("off")
                ax[r, c].set_title(k.split("/", 1)[0] + " d=" + str(row.hamming), fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(a.out, "pairs_montage.png"), dpi=120)
else:
    print("No cross-split duplicates at this threshold.")
