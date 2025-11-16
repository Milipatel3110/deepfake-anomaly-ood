#!/bin/bash
# Run ODIN + Energy + Mahalanobis on full FaceForensics++ (in-distribution)

python ODIN/odin_run.py \
  --real_root datasets/faceforensics/original_sequences_face \
  --fake_root datasets/faceforensics/Deepfakes_face \
  --ckpt ODIN/resnet_ffpp_real_fake.pth \
  --frames_per_id 2 \
  --batch_size 32 \
  --workers 6 \
  --T 1 \
  --epsilon 0.006 \
  --out_csv outputs/ffpp_ood_results.csv
