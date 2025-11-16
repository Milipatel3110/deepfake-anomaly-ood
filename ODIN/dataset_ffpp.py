# dataset_ffpp.py
import os, random, glob
from typing import List, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def default_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def list_frame_paths(root_dir: str, exts=(".jpg", ".jpeg", ".png")) -> List[str]:
    """Return a list of image file paths from all immediate subfolders."""
    paths = []
    # one folder per ID (e.g., 000/, 001/, 000_003/) with frames inside
    for sub in sorted(os.listdir(root_dir)):
        subdir = os.path.join(root_dir, sub)
        if not os.path.isdir(subdir):
            continue
        for f in os.listdir(subdir):
            if f.lower().endswith(exts):
                paths.append(os.path.join(subdir, f))
    return paths

def pick_k_per_id(root_dir: str, k: int, exts=(".jpg", ".jpeg", ".png")) -> List[str]:
    """Pick up to k frames per ID folder for quick experiments."""
    sel = []
    for sub in sorted(os.listdir(root_dir)):
        subdir = os.path.join(root_dir, sub)
        if not os.path.isdir(subdir):
            continue
        frames = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.lower().endswith(exts)]
        frames.sort()
        if not frames:
            continue
        if k > 0:
            frames = frames[:k]
        sel.extend(frames)
    return sel

class FFPPFaces(Dataset):
    """
    FaceForensics++ faces loader.
    - real_root: faceforensics/original_sequences_face (000/, 001/, ...)
    - fake_root: faceforensics/Deepfakes_face (000_003/, 001_870/, ...)
    - frames_per_id: how many frames to sample per ID folder (1 or 5 are good)
    """
    def __init__(self,
                 real_root: str,
                 fake_root: str,
                 frames_per_id: int = 1,
                 transform=None,
                 limit_per_class: int = None,
                 seed: int = 42):
        self.transform = transform or default_transform()
        self.samples: List[Tuple[str,int]] = []

        real_paths = pick_k_per_id(real_root, frames_per_id)
        fake_paths = pick_k_per_id(fake_root, frames_per_id)

        if limit_per_class is not None:
            real_paths = real_paths[:limit_per_class]
            fake_paths = fake_paths[:limit_per_class]

        self.samples.extend([(p, 0) for p in real_paths])  # 0 = real
        self.samples.extend([(p, 1) for p in fake_paths])  # 1 = fake

        random.Random(seed).shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long), path
