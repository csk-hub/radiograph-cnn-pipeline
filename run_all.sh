#!/usr/bin/env bash
# Suggested order (Kaggle, 2xT4).  Edit ROOT if the dataset is mounted elsewhere.
set -e
SEEDS="1 2 3 4 5 6 7 8 9 10"

# 0. Audit first: dataset description, truncated files, duplicates (R1-2, R1-4, R2-c)
python 02_data_audit.py --out audit --embed

# 1. Paper configuration, 10 seeds (R1-9)  -> results/
for s in $SEEDS; do python train.py --name shallow_orig --seed $s; done

# 2. Extended metrics + threshold analysis on seed 1 (R1-3, R1-6); then de-duplicated test set (R1-4)
python 01_metrics_and_threshold.py --preds results/shallow_orig_seed1_preds.npz --out metrics_out
python 01_metrics_and_threshold.py --preds results/shallow_orig_seed1_preds.npz --exclude audit/exclude_test.txt --out metrics_out_dedup
python train.py --name shallow_dedup --seed 1 --exclude audit/exclude_train.txt   # retrain without leaked train images

# 3. Truncated-image handling (R2-c): strict decode drops failing files
for s in 1 2 3 4 5; do python train.py --name shallow_strict_decode --no_truncated --seed $s; done

# 4. Augmentation ablation + aspect ratio (R2-d)
for s in 1 2 3 4 5; do
  python train.py --name aug_none      --aug none     --seed $s
  python train.py --name aug_rot       --aug rot      --seed $s
  python train.py --name aug_zoom      --aug zoom     --seed $s
  python train.py --name aug_flip      --aug flip     --seed $s
  python train.py --name letterbox150  --letterbox    --seed $s
  python train.py --name letterbox224  --letterbox --size 224 --seed $s
done

# 5. Classifier head (R1-8)
for s in 1 2 3 4 5; do for h in gap1 gap256 flatten64; do python train.py --name head_$h --head $h --seed $s; done; done

# 6. Regularisation / schedule / epoch budget (R2-g)
for s in 1 2 3 4 5; do
  python train.py --name es_patience4  --epochs 40 --early_stop 4 --seed $s
  python train.py --name es_reducelr   --epochs 40 --early_stop 6 --reduce_lr --seed $s
  python train.py --name l2_1e-4       --l2 1e-4 --seed $s
  for e in 10 20 30; do python train.py --name epochs_$e --epochs $e --seed $s; done
done

# 7. One-factor studies (R1-9): batch, dropout, LR   (select on VALIDATION metrics)
for s in 1 2 3; do
  for b in 64 128; do python train.py --name batch_$b --batch $b --seed $s; done
  for d in 0.3 0.7; do python train.py --name drop_$d --dropout $d --seed $s; done
  for lr in 1e-4 5e-4; do python train.py --name lr_$lr --lr $lr --seed $s; done
done

# 8. Baselines on identical splits (R1-7). Use 1e-4 for fine-tuning pretrained nets.
for s in 1 2 3 4 5; do
  python train.py --name mnv2_scratch  --model mobilenetv2 --seed $s
  python train.py --name mnv2_pre      --model mobilenetv2 --pretrained --lr 1e-4 --seed $s
  python train.py --name effb0_scratch --model efficientnetb0 --seed $s
  python train.py --name effb0_pre     --model efficientnetb0 --pretrained --lr 1e-4 --seed $s
  python train.py --name resnet18_scratch --model resnet18 --seed $s
  # optional second resolution: add --size 224 and a different --name
done

# 9. GPU benchmark (R1-5)
python 05_gpu_benchmark.py --reps 3 --epochs 6

# 10. Explainability (R1-10)
python gradcam.py --model results/shallow_orig_seed1.keras --preds results/shallow_orig_seed1_preds.npz

# 11. Summary tables
python aggregate.py results --csv summary.csv
