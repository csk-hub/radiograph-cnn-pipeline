# Revision experiments - fracture screening CNN

Everything the reviewers asked for that needs code. Pipeline: PIL loads each split into memory once
(cached), augmentation runs on-the-fly in `tf.data`, training uses `tf.distribute.MirroredStrategy`.

**Label convention.** Keras assigns label 0 = *fractured*, 1 = *not fractured*; the network output
`p` is P(not fractured). All scripts save and analyse the **fracture score `s = 1 - p`** with
*fractured* as the positive class (answers R2-e).

| Script | Reviewer comments |
|---|---|
| `02_data_audit.py` | R1-2 dataset description (class counts, image sizes/modes), R2-c truncated-file count, R1-4 exact/near duplicates across splits (MD5, pHash, optional CNN embedding) |
| `train.py` | one run: seeds (R1-9), augmentation ablation (R2-d), letterbox (R2-d), heads (R1-8), early stopping / LR schedule / L2 / epochs (R2-g), baselines (R1-7), strict-decode (R2-c), MACs + latency + params + peak GPU memory |
| `01_metrics_and_threshold.py` | R1-3/R1-6: ROC-AUC, PR-AUC, balanced acc, MCC, bootstrap + Clopper-Pearson CIs, validation-selected thresholds, optional de-duplicated test set |
| `aggregate.py` | R1-9: mean +- SD, 95% t-CI, FN per seed, generalisation gap |
| `05_gpu_benchmark.py` | R1-5: epoch time, throughput, memory, speedup, parallel efficiency |
| `gradcam.py` | R1-10: Grad-CAM panels and the list of false-positive files for clinician review |
| `run_all.sh` | suggested run order (a few hours of GPU time; trim seeds if needed) |

## Read before running
- **These scripts were written against your notebook and syntax-checked, and the numpy-only parts
  (`01_...`, `02_...`) were tested on synthetic data. The TensorFlow parts (`train.py`, `05_...`,
  `gradcam.py`) have NOT been run** - expect to fix small version issues on first run
  (Keras 2.x vs 3.x layer names, memory-info API). Do a 1-epoch smoke test:
  `python train.py --name smoke --epochs 1 --seed 1`.
- New runs use bilinear resizing and a `tf.data` pipeline, whereas the paper used
  `ImageDataGenerator` (nearest-neighbour resize, different rotation/zoom implementation).
  Your headline numbers may shift slightly. **Re-run the paper configuration (`shallow_orig`) with
  this pipeline and report that as the reference for all comparisons**, and say so in the paper.
- `--exclude` in `train.py` removes training files only. For the test set, use `--exclude` in
  `01_metrics_and_threshold.py` (post-hoc, no retraining needed for the first look).
- Choose hyper-parameters, thresholds and the epoch budget on **validation** results only; touch the
  test set once per final configuration.
- ResNet18 has no ImageNet weights in Keras, so it is trained from scratch (state this in the paper);
  for a pretrained ResNet18 use torchvision with the same split lists (`audit/manifest.csv`).
- `from_tensor_slices` embeds the arrays in the graph; at 224 px the training set is ~1.3 GB, which
  is near TensorFlow's 2 GB limit. If it fails, switch `make_ds` to a generator.
- Grad-CAM works for the shallow CNN and ResNet18; pretrained backbones nest their conv layers
  and need a small adaptation.
- Record package versions (`pip freeze`) and GPU/driver details; the GPU benchmark must be run on
  an otherwise idle machine and the input pipeline can limit scaling (report it honestly).
