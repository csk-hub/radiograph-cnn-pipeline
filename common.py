"""Shared utilities for the revision experiments.

Label convention (matches Keras flow_from_directory / alphabetical order):
    label 0 = 'fractured', label 1 = 'not fractured'.
The network output p = P(not fractured).  Everywhere in this package we use
    fracture score  s = 1 - p     and     positive class = fractured.
"""
import os, json, hashlib
import numpy as np
from PIL import Image, ImageFile

CLASSES = ["fractured", "not fractured"]
DEFAULT_ROOT = ("/kaggle/input/fracture-multi-region-x-ray-data/"
                "Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


# ----------------------------------------------------------------- data ----
def list_split(root, split):
    """Return sorted [(abs_path, key, label)] for one split. key = split/class/file."""
    items = []
    for li, cls in enumerate(CLASSES):
        d = os.path.join(root, split, cls)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"{d} not found (class folders must be {CLASSES})")
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMG_EXT):
                items.append((os.path.join(d, f), f"{split}/{cls}/{f}", li))
    return items


def load_image(path, size, letterbox=False):
    img = Image.open(path).convert("RGB")
    if letterbox:  # keep aspect ratio, pad with black
        w, h = img.size
        s = size / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
        return np.asarray(canvas, dtype=np.uint8)
    return np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.uint8)


def load_split(root, split, size=150, letterbox=False, truncated_ok=True,
               exclude=None, cache_dir=None):
    """Load a whole split into memory (uint8).  Returns dict(X, y, keys, skipped).

    truncated_ok=True  -> PIL LOAD_TRUNCATED_IMAGES=True (as in the submitted paper)
    truncated_ok=False -> files that fail a strict decode are skipped and reported
    exclude            -> set of keys (e.g. duplicates found by 02_data_audit.py)
    """
    exclude = set(exclude or [])
    tag = f"{split}_{size}_{int(letterbox)}_{int(truncated_ok)}_{len(exclude)}"
    cpath = os.path.join(cache_dir, tag + ".npz") if cache_dir else None
    if cpath and os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        return dict(X=z["X"], y=z["y"], keys=list(z["keys"]), skipped=list(z["skipped"]))
    ImageFile.LOAD_TRUNCATED_IMAGES = truncated_ok
    X, y, keys, skipped = [], [], [], []
    for path, key, label in list_split(root, split):
        if key in exclude:
            continue
        try:
            X.append(load_image(path, size, letterbox)); y.append(label); keys.append(key)
        except Exception as e:  # truncated/corrupt when truncated_ok=False
            skipped.append(key)
    out = dict(X=np.stack(X), y=np.array(y, dtype=np.int64), keys=keys, skipped=skipped)
    if cpath:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cpath, X=out["X"], y=out["y"], keys=np.array(keys),
                            skipped=np.array(skipped))
    return out


# --------------------------------------------------------------- metrics ----
def cp_ci(k, n, alpha=0.05):
    """Exact Clopper-Pearson interval for k successes in n trials."""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def confusion(y_pos, score, thr=0.5):
    pred = (np.asarray(score) >= thr).astype(int)
    y = np.asarray(y_pos).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fn, fp, tn


def metrics(y_pos, score, thr=0.5):
    """y_pos: 1 = fractured (positive).  score: fracture score s = 1 - p."""
    from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
    tp, fn, fp, tn = confusion(y_pos, score, thr)
    d = lambda a, b: a / b if b else float("nan")
    sens, spec = d(tp, tp + fn), d(tn, tn + fp)
    ppv, npv = d(tp, tp + fp), d(tn, tn + fn)
    pred = (np.asarray(score) >= thr).astype(int)
    return dict(TP=tp, FN=fn, FP=fp, TN=tn, sensitivity=sens, specificity=spec,
                ppv=ppv, npv=npv, accuracy=d(tp + tn, tp + tn + fp + fn),
                f1=d(2 * tp, 2 * tp + fp + fn), balanced_acc=(sens + spec) / 2,
                mcc=float(matthews_corrcoef(y_pos, pred)),
                roc_auc=float(roc_auc_score(y_pos, score)),
                pr_auc=float(average_precision_score(y_pos, score)))


def bootstrap_ci(y_pos, score, thr=0.5, n_boot=2000, seed=0, keys=None):
    """Percentile bootstrap (stratified by class) 95% CIs for all scalar metrics."""
    rng = np.random.default_rng(seed)
    y_pos = np.asarray(y_pos); score = np.asarray(score)
    pos, neg = np.where(y_pos == 1)[0], np.where(y_pos == 0)[0]
    keys = keys or ["sensitivity", "specificity", "ppv", "npv", "accuracy", "f1",
                    "balanced_acc", "mcc", "roc_auc", "pr_auc"]
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = metrics(y_pos[idx], score[idx], thr)
        for k in keys:
            acc[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
            for k, v in acc.items()}


# ---------------------------------------------------------------- models ----
def build_model(name="shallow", size=150, head="flatten256", dropout=0.5,
                lr=1e-3, l2=0.0, pretrained=False):
    import tensorflow as tf
    from tensorflow.keras import layers as L, Model, regularizers
    reg = regularizers.l2(l2) if l2 > 0 else None
    inp = L.Input((size, size, 3))               # float 0..255

    if name == "shallow":
        x = L.Rescaling(1 / 255.0)(inp)
        for f in (32, 64, 128):
            x = L.Conv2D(f, 3, activation="relu", kernel_regularizer=reg)(x)
            x = L.MaxPooling2D(2)(x)
        if head == "flatten256":
            x = L.Flatten()(x); x = L.Dense(256, activation="relu", kernel_regularizer=reg)(x)
        elif head == "flatten64":
            x = L.Flatten()(x); x = L.Dense(64, activation="relu", kernel_regularizer=reg)(x)
        elif head == "gap256":
            x = L.GlobalAveragePooling2D()(x); x = L.Dense(256, activation="relu", kernel_regularizer=reg)(x)
        elif head == "gap1":
            x = L.GlobalAveragePooling2D()(x)
        else:
            raise ValueError(head)
        x = L.Dropout(dropout)(x)
    elif name in ("mobilenetv2", "efficientnetb0"):
        w = "imagenet" if pretrained else None
        if name == "mobilenetv2":
            base = tf.keras.applications.MobileNetV2(input_shape=(size, size, 3), include_top=False, weights=w)
            x = L.Rescaling(1 / 127.5, offset=-1)(inp)          # MobileNetV2 expects [-1, 1]
        else:
            base = tf.keras.applications.EfficientNetB0(input_shape=(size, size, 3), include_top=False, weights=w)
            x = inp                                              # has its own Rescaling (expects 0..255)
        x = base(x); x = L.GlobalAveragePooling2D()(x); x = L.Dropout(dropout)(x)
    elif name == "resnet18":                                     # from scratch only (no Keras ImageNet weights)
        def block(t, f, s):
            sc = t
            t = L.Conv2D(f, 3, s, padding="same", use_bias=False)(t); t = L.BatchNormalization()(t); t = L.ReLU()(t)
            t = L.Conv2D(f, 3, padding="same", use_bias=False)(t); t = L.BatchNormalization()(t)
            if s != 1 or sc.shape[-1] != f:
                sc = L.Conv2D(f, 1, s, use_bias=False)(sc); sc = L.BatchNormalization()(sc)
            return L.ReLU()(L.Add()([t, sc]))
        x = L.Rescaling(1 / 255.0)(inp)
        x = L.Conv2D(64, 7, 2, padding="same", use_bias=False)(x); x = L.BatchNormalization()(x); x = L.ReLU()(x)
        x = L.MaxPooling2D(3, 2, padding="same")(x)
        for f, s in [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1)]:
            x = block(x, f, s)
        x = L.GlobalAveragePooling2D()(x); x = L.Dropout(dropout)(x)
    else:
        raise ValueError(name)

    out = L.Dense(1, activation="sigmoid")(x)                    # p = P(not fractured)
    model = Model(inp, out, name=f"{name}_{head}" if name == "shallow" else name)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def count_macs(model, size):
    """Multiply-accumulates per image via the TF profiler (FLOPs / 2). None on failure."""
    try:
        import tensorflow as tf
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
        f = tf.function(lambda x: model(x, training=False))
        conc = f.get_concrete_function(tf.TensorSpec([1, size, size, 3], tf.float32))
        _, gdef = convert_variables_to_constants_v2_as_graph(conc)
        with tf.Graph().as_default() as g:
            tf.graph_util.import_graph_def(gdef, name="")
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            fl = tf.compat.v1.profiler.profile(graph=g, run_meta=tf.compat.v1.RunMetadata(),
                                               cmd="op", options=opts)
        return fl.total_float_ops / 2.0
    except Exception as e:
        print("MAC counting failed:", e)
        return None


def latency_ms(model, size, n=200, warmup=30):
    """Batch-1 inference latency (ms): mean / median / p95 over n calls."""
    import time, tensorflow as tf
    fn = tf.function(lambda x: model(x, training=False))
    x = tf.random.uniform([1, size, size, 3], 0, 255)
    for _ in range(warmup):
        fn(x).numpy()
    t = []
    for _ in range(n):
        s = time.perf_counter(); fn(x).numpy(); t.append((time.perf_counter() - s) * 1e3)
    return dict(mean=float(np.mean(t)), median=float(np.median(t)), p95=float(np.percentile(t, 95)))
