# train_ffpp.py
import os, argparse, random
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from dataset_ffpp import FFPPFaces, default_transform

def make_model(backbone="resnet50", freeze_to="layer3"):
    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        feat = model.fc.in_features
        # freeze all up to layer3
        for name, p in model.named_parameters():
            p.requires_grad = not (freeze_to in name or name.startswith("conv1") or name.startswith("bn1") or name.startswith("layer1") or name.startswith("layer2"))
        model.fc = nn.Linear(feat, 2)
        return model
    else:
        raise ValueError("Only resnet50 in this quick script")

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    tot, corr, n = 0.0, 0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        tot += loss.item()*x.size(0)
        corr += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return tot/n, corr/n

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    corr, n = 0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        corr += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return corr/n

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds = FFPPFaces(args.real_root, args.fake_root,
                   frames_per_id=args.frames_per_id,
                   transform=default_transform(args.img_size),
                   limit_per_class=args.limit_per_class)

    # 80/20 split
    n = len(ds)
    n_train = int(0.8*n)
    n_val = n - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = make_model().to(device)
    # class weights in case of slight imbalance
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc, best_path = 0.0, args.ckpt_out
    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_acc = eval_epoch(model, val_loader, device)
        sched.step()
        print(f"[{ep:03d}] loss={tr_loss:.4f} train_acc={tr_acc:.3f} val_acc={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ saved best to {best_path}")
    print(f"Best val acc = {best_acc:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_root", required=True)
    ap.add_argument("--fake_root", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--frames_per_id", type=int, default=2)
    ap.add_argument("--limit_per_class", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ckpt_out", type=str, default="resnet_ffpp_real_fake.pth")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)
