# compute_maha_stats_ffpp.py
import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torchvision import models

from dataset_ffpp import FFPPFaces, default_transform


def extract_features(model, x, device):
    # penultimate features for ResNet-50
    x = x.to(device)
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_root", type=str, required=True,
                    help="FF++ original_sequences_face")
    ap.add_argument("--fake_root", type=str, required=True,
                    help="FF++ Deepfakes_face (used only to balance loader)")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="FF++ trained classifier (resnet_ffpp_real_fake.pth)")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--frames_per_id", type=int, default=2)
    ap.add_argument("--limit_per_class", type=int, default=800)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out_npz", type=str, default="../outputs/ffpp_maha_stats.npz")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # dataset – same loader as ODIN, but we only use real features to fit stats
    ds = FFPPFaces(
        real_root=args.real_root,
        fake_root=args.fake_root,
        frames_per_id=args.frames_per_id,
        transform=default_transform(args.img_size),
        limit_per_class=args.limit_per_class,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    print(f"Total FF++ samples (real+fake): {len(ds)}")

    # model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 2)
    print(f"Loading checkpoint: {args.ckpt}")
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.to(device).eval()

    feats_real = []

    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc="Extracting FF++ features"):
            feats = extract_features(model, imgs, device)
            feats = feats.cpu().numpy()
            labels_np = labels.numpy()
            # keep only REAL (label 0) for Mahalanobis stats
            feats_real.append(feats[labels_np == 0])

    feats_real = np.concatenate(feats_real, axis=0)
    print(f"Real feature matrix shape: {feats_real.shape}")

    mu = feats_real.mean(axis=0)
    cov = np.cov(feats_real.T) + 1e-6 * np.eye(feats_real.shape[1])
    prec = np.linalg.inv(cov)

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez(args.out_npz, mu=mu, prec=prec)
    print(f"Saved FF++ Mahalanobis stats to {args.out_npz}")


if __name__ == "__main__":
    main()
