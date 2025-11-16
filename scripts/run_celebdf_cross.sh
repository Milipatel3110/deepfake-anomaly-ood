#!/bin/bash
# Cross-dataset: train on FF++ (already trained) and test on CelebDF

python ODIN/odin_run.py \
  --real_root datasets/celebdf/CelebDF\ Facial\ Images/images/real \
  --fake_root datasets/celebdf/CelebDF\ Facial\ Images/images/fake \
  --ckpt ODIN/resnet_ffpp_real_fake.pth \
  --frames_per_id 2 \
  --batch_size 32 \
  --workers 6 \
  --T 1 \
  --epsilon 0.006 \
  --out_csv outputs/celebdf_ood_results.csv
