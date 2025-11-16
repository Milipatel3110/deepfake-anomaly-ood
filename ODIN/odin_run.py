import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.covariance import EmpiricalCovariance  # kept, though not strictly needed

from dataset_ffpp import FFPPFaces, IMAGENET_STD, default_transform

# ---------- Image normalization helpers ----------
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _renorm(img01, device):
    return (img01.to(device) - IM_MEAN.to(device)) / IM_STD.to(device)


def _denorm(x_norm, device):
    return x_norm.to(device) * IM_STD.to(device) + IM_MEAN.to(device)


def fit_real_logit_stats(model, loader, device, max_real=2000):
    """Compute mean and std of logits on real (label=0) samples."""
    mus, sigmas, n = None, None, 0
    model.eval()
    with torch.no_grad():
        for imgs, labels, _ in loader:
            idx = (labels == 0)
            if idx.sum() == 0:
                continue
            x = imgs[idx].to(device)
            z = model(x)
            batch_mu = z.mean(0)
            batch_var = z.var(0, unbiased=False) + 1e-6
            batch_sigma = batch_var.sqrt()
            b = z.size(0)
            if mus is None:
                mus, sigmas, n = batch_mu, batch_sigma, b
            else:
                alpha = b / (n + b)
                mus = (1 - alpha) * mus + alpha * batch_mu
                sigmas = (1 - alpha) * sigmas + alpha * batch_sigma
                n += b
            if n >= max_real:
                break
    return mus.cpu(), sigmas.cpu()


def energy_both(model, x_norm, T=1.0, epsilon=0.006, device="cpu",
                real_mu=None, real_sigma=None):
    """
    Computes energy before and after ODIN perturbation.
    Uses real-data logit normalization (calibrated) for stability.
    """
    model.eval()
    x = x_norm.clone().detach().to(device)

    # energy before perturbation
    with torch.no_grad():
        logits0 = model(x)
        if (real_mu is not None) and (real_sigma is not None):
            logits0 = (logits0 - real_mu.to(device)) / (real_sigma.to(device) + 1e-8)
        E0 = -T * torch.logsumexp(logits0 / T, dim=1)

    # generate perturbed image (ODIN-style)
    x.requires_grad = True
    logits = model(x)
    preds = logits.argmax(1)
    loss = -F.log_softmax(logits / max(T, 1.0), dim=1).gather(
        1, preds.view(-1, 1)
    ).mean()
    loss.backward()
    grad = x.grad.data
    step_img = epsilon * torch.sign(grad / IM_STD.to(device))
    x_img = torch.clamp(_denorm(x, device), 0, 1)
    x_img_pert = torch.clamp(x_img - step_img, 0, 1)
    x_pert = _renorm(x_img_pert, device)

    # energy after perturbation
    with torch.no_grad():
        logits1 = model(x_pert)
        if (real_mu is not None) and (real_sigma is not None):
            logits1 = (logits1 - real_mu.to(device)) / (real_sigma.to(device) + 1e-8)
        E1 = -T * torch.logsumexp(logits1 / T, dim=1)

    return E0.cpu().numpy(), E1.cpu().numpy()


# ---------- ODIN confidence scoring ----------
def odin_confidence(model, x_norm, T=1000.0, epsilon=0.0014, device="cpu"):
    """
    ODIN: Input perturbation + temperature scaling to detect anomalies.
    """
    model.eval()
    x = x_norm.clone().detach().to(device)
    x.requires_grad = True

    logits = model(x)
    preds = logits.argmax(dim=1)
    log_probs = F.log_softmax(logits / T, dim=1)
    loss = -log_probs.gather(1, preds.view(-1, 1)).mean()
    loss.backward()

    grad = x.grad.data
    step_img = epsilon * torch.sign(grad / IM_STD.to(device))
    x_img = torch.clamp(_denorm(x, device), 0.0, 1.0)
    x_img_pert = torch.clamp(x_img - step_img, 0.0, 1.0)
    x_pert_norm = _renorm(x_img_pert, device)

    with torch.no_grad():
        logits_p = model(x_pert_norm)
        conf = F.softmax(logits_p / T, dim=1).max(dim=1).values
    return conf.detach().cpu().numpy()


# ---------- Delta confidence ----------
def delta_confidence(model, x_norm, T=1000.0, epsilon=0.0014, device="cpu"):
    """
    Delta confidence = drop in confidence after perturbation.
    """
    model.eval()
    x = x_norm.clone().detach().to(device)
    x.requires_grad = True

    logits = model(x)
    conf_before = F.softmax(logits / T, dim=1).max(dim=1).values

    log_probs = F.log_softmax(logits / T, dim=1)
    preds = logits.argmax(dim=1)
    loss = -log_probs.gather(1, preds.view(-1, 1)).mean()
    loss.backward()

    grad = x.grad.data
    step_img = epsilon * torch.sign(grad / IM_STD.to(device))
    x_img = torch.clamp(_denorm(x, device), 0.0, 1.0)
    x_img_pert = torch.clamp(x_img - step_img, 0.0, 1.0)
    x_pert_norm = _renorm(x_img_pert, device)

    with torch.no_grad():
        logits_p = model(x_pert_norm)
        conf_after = F.softmax(logits_p / T, dim=1).max(dim=1).values

    delta = conf_before - conf_after
    return delta.detach().cpu().numpy()


# ---------- Energy-based OOD scoring ----------
def energy_score(model, x, T=1.0, device="cpu"):
    """
    Energy-based OOD detection with normalized logits.
    Normalizing logits improves stability and prevents scale bias.
    """
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        logits = logits / (torch.norm(logits, dim=1, keepdim=True) + 1e-8)
        energy = -T * torch.logsumexp(logits / T, dim=1)
    return energy.cpu().numpy()


# ---------- Mahalanobis Distance ----------
def mahalanobis_confidence(model, loader, device, maha_stats_path=None):
    """
    Computes Mahalanobis distances using penultimate-layer features.

    If maha_stats_path is provided, uses precomputed (mu, prec) from FF++.
    Otherwise, fits Gaussian on REAL samples from the current loader.
    """
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for imgs, y, _ in tqdm(loader, desc="Extracting features"):
            x = imgs.to(device)
            # manual penultimate feature extraction for ResNet-50
            f = model.conv1(x)
            f = model.bn1(f)
            f = model.relu(f)
            f = model.maxpool(f)
            f = model.layer1(f)
            f = model.layer2(f)
            f = model.layer3(f)
            f = model.layer4(f)
            f = model.avgpool(f)
            f = torch.flatten(f, 1)
            feats.append(f.cpu().numpy())
            labels.extend(y.cpu().numpy())

    feats = np.concatenate(feats, axis=0)
    labels = np.array(labels)

    # load FF++ stats if available, else fit on current real samples
    if maha_stats_path and os.path.isfile(maha_stats_path):
        print(f"Loading Mahalanobis stats from {maha_stats_path}")
        d = np.load(maha_stats_path)
        mu = d["mu"]
        prec = d["prec"]
    else:
        print("No Mahalanobis stats file given – fitting on current REAL samples.")
        real_feats = feats[labels == 0]
        mu = real_feats.mean(axis=0)
        cov = np.cov(real_feats.T) + 1e-6 * np.eye(real_feats.shape[1])
        prec = np.linalg.inv(cov)

    dists = []
    for f in feats:
        diff = f - mu
        d = float(diff @ prec @ diff)
        dists.append(d)

    dists = np.array(dists)
    return dists, labels


# ---------- Main pipeline ----------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    # Dataset
    ds = FFPPFaces(
        real_root=args.real_root,
        fake_root=args.fake_root,
        frames_per_id=args.frames_per_id,
        transform=default_transform(args.img_size),
        limit_per_class=args.limit_per_class,
    )
    print(f"Total samples: {len(ds)} (real+fake)")

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 2)

    if args.ckpt and os.path.isfile(args.ckpt):
        print(f"Loading checkpoint: {args.ckpt}")
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    model.to(device).eval()

    # ---------- logit calibration for energy ----------
    print("\n--- Calibrating real-data logit stats ---")
    real_mu, real_sigma = fit_real_logit_stats(model, loader, device)
    print("Done computing mean/std for real samples.\n")

    # ---------- ODIN ----------
    print("\n--- Running ODIN scoring ---")
    all_conf, all_labels, all_paths = [], [], []
    for imgs, labels, paths in tqdm(loader, desc="ODIN scoring"):
        conf = odin_confidence(model, imgs, T=args.T, epsilon=args.epsilon, device=device)
        all_conf.extend(conf)
        all_labels.extend(labels.numpy().tolist())
        all_paths.extend(paths)

    all_conf = np.array(all_conf)
    y_true = np.array(all_labels)
    anomaly_scores = 1.0 - all_conf
    auroc_odin = roc_auc_score(y_true, anomaly_scores)
    print(f"AUROC (real vs deepfake, ODIN): {auroc_odin:.4f}")

    # ---------- Delta Confidence ----------
    print("\n--- Running Delta Confidence Scoring ---")
    delta_scores = []
    for imgs, labels, _ in tqdm(loader, desc="Delta confidence"):
        delta = delta_confidence(model, imgs, T=args.T, epsilon=args.epsilon, device=device)
        delta_scores.extend(delta)
    delta_scores = np.array(delta_scores)
    auroc_delta = roc_auc_score(y_true, delta_scores)
    print(f"AUROC (real vs deepfake, Delta Confidence): {auroc_delta:.4f}")

    # ---------- Improved Energy-based ----------
    print("\n--- Running Improved Energy-based OOD Scoring ---")
    all_E0, all_E1 = [], []
    for imgs, _, _ in tqdm(loader, desc="Energy (before/after)"):
        E0, E1 = energy_both(
            model, imgs, T=args.T, epsilon=args.epsilon,
            device=device, real_mu=real_mu, real_sigma=real_sigma
        )
        all_E0.extend(E0)
        all_E1.extend(E1)

    all_E0 = np.array(all_E0)
    all_E1 = np.array(all_E1)
    score_energy_after = all_E1
    score_energy_delta = all_E1 - all_E0

    auroc_energy_after = roc_auc_score(y_true, score_energy_after)
    auroc_energy_delta = roc_auc_score(y_true, score_energy_delta)
    print(f"AUROC (Energy after perturbation): {auroc_energy_after:.4f}")
    print(f"AUROC (ΔEnergy before→after): {auroc_energy_delta:.4f}")

    # Save per-sample ΔEnergy values for visualization
    print("\nSaving ΔEnergy details for visualization...")
    import csv
    delta_energy = score_energy_delta
    delta_csv_path = args.out_csv.replace(".csv", "_delta.csv")
    with open(delta_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label(fake=1)", "energy_before", "energy_after", "delta_energy"])
        for p, y, e0, e1 in zip(all_paths, y_true, all_E0, all_E1):
            w.writerow([p, y, f"{e0:.6f}", f"{e1:.6f}", f"{e1 - e0:.6f}"])
    print(f"Saved → {delta_csv_path}")

    # ---------- Plain Energy ----------
    print("\n--- Running Energy-based OOD Scoring ---")
    all_energy = []
    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc="Energy scoring"):
            energy = energy_score(model, imgs, T=1.0, device=device)
            all_energy.extend(energy)
    all_energy = np.array(all_energy)
    auroc_energy = roc_auc_score(y_true, all_energy)
    print(f"AUROC (real vs deepfake, Energy-based): {auroc_energy:.4f}")

    # ---------- Mahalanobis ----------
    print("\n--- Running Mahalanobis Scoring ---")
    maha_raw, y_maha = mahalanobis_confidence(
        model, loader, device, maha_stats_path=args.maha_stats
    )

    # auto sign-fix if inverted
    auroc_maha_raw = roc_auc_score(y_maha, maha_raw)
    if auroc_maha_raw < 0.5:
        print(f"Mahalanobis AUROC was {auroc_maha_raw:.4f} (<0.5), flipping sign.")
        maha_scores = -maha_raw
        auroc_maha = roc_auc_score(y_maha, maha_scores)
    else:
        maha_scores = maha_raw
        auroc_maha = auroc_maha_raw

    print(f"AUROC (real vs deepfake, Mahalanobis): {auroc_maha:.4f}")

    # ---------- Hybrid (Combined) ----------
    print("\n--- Combining All Methods ---")
    scaler = MinMaxScaler()
    # stack: ODIN anomaly, ΔConf, plain energy, Mahalanobis
    features = np.stack(
        [anomaly_scores, delta_scores, all_energy, maha_scores],
        axis=1
    )
    features_norm = scaler.fit_transform(features)
    combined = features_norm.mean(axis=1)

    auroc_combined = roc_auc_score(y_true, combined)
    print(f"\nCombined AUROC (Hybrid OOD): {auroc_combined:.4f}")

    # ---------- Save results ----------
    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "path", "label(fake=1)",
                "odin_conf", "anomaly_score(1-conf)",
                "delta_conf", "energy_plain",
                "mahalanobis"
            ])
            for p, y, c, d, e, m in zip(
                all_paths, y_true, all_conf,
                delta_scores, all_energy, maha_scores
            ):
                w.writerow([
                    p, y,
                    f"{c:.6f}",
                    f"{1.0 - c:.6f}",
                    f"{d:.6f}",
                    f"{e:.6f}",
                    f"{m:.6f}",
                ])
        print(f"Saved detailed results → {args.out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--real_root", type=str, required=True,
        help="e.g., ../faceforensics/original_sequences_face"
    )
    ap.add_argument(
        "--fake_root", type=str, required=True,
        help="e.g., ../faceforensics/Deepfakes_face"
    )
    ap.add_argument(
        "--ckpt", type=str, default="",
        help="Path to trained real/fake ResNet classifier"
    )
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--frames_per_id", type=int, default=2)
    ap.add_argument("--limit_per_class", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--T", type=float, default=1000.0)
    ap.add_argument("--epsilon", type=float, default=0.006)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument(
        "--out_csv", type=str,
        default="../outputs/ood_results.csv"
    )
    ap.add_argument(
        "--maha_stats", type=str, default="",
        help="optional npz file with mu,prec for Mahalanobis (e.g., from FF++)"
    )
    args = ap.parse_args()
    run(args)
