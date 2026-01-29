import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
OUT_ROOT = PROJECT_ROOT / "outputs" / "clipseg_finetuned"

OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def compute_iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 and inter == 0 else inter / max(union, 1)

def compute_dice(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    return 1.0 if total == 0 and inter == 0 else 2 * inter / max(total, 1)

# ---------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------

class DrywallDataset(Dataset):
    def __init__(self, img_dir, mask_dir, prompt, processor):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.prompt = prompt
        self.processor = processor

        assert self.img_dir.exists(), f"Missing images: {self.img_dir}"
        assert self.mask_dir.exists(), f"Missing masks: {self.mask_dir}"

        self.images = sorted(
            [p for p in self.img_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = self.mask_dir / f"{img_path.stem}_mask.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros(image.shape[:2], np.float32)

        inputs = self.processor(
            text=[self.prompt],
            images=[Image.fromarray(image)],
            padding=True,
            return_tensors="pt"
        )

        return {
            "pixel_values": inputs["pixel_values"][0],
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "labels": torch.tensor(mask),
            "filename": img_path.name
        }

# ---------------------------------------------------------------------
# COLLATE
# ---------------------------------------------------------------------

def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])

    max_len = max(b["input_ids"].shape[0] for b in batch)

    def pad(x):
        return torch.cat([x, torch.zeros(max_len - x.shape[0], dtype=x.dtype)])

    input_ids = torch.stack([pad(b["input_ids"]) for b in batch])
    attention_mask = torch.stack([pad(b["attention_mask"]) for b in batch])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "filenames": [b["filename"] for b in batch]
    }

# ---------------------------------------------------------------------
# LOSS
# ---------------------------------------------------------------------

def dice_loss(pred, target, eps=1e-5):
    p = pred.view(-1)
    t = target.view(-1)
    inter = (p * t).sum()
    return 1 - (2 * inter + eps) / (p.sum() + t.sum() + eps)

# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------

@torch.no_grad()
def validate(model, dataset, device):
    model.eval()
    ious, dices = [], []

    for sample in dataset:
        pv = sample["pixel_values"].unsqueeze(0).to(device)
        ids = sample["input_ids"].unsqueeze(0).to(device)
        am = sample["attention_mask"].unsqueeze(0).to(device)
        gt = sample["labels"].numpy()

        logits = model(pixel_values=pv, input_ids=ids, attention_mask=am).logits
        logits = torch.nn.functional.interpolate(
            logits.unsqueeze(1), size=gt.shape, mode="bilinear", align_corners=False
        ).squeeze()

        pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        ious.append(compute_iou(pred, gt))
        dices.append(compute_dice(pred, gt))

    return np.mean(ious), np.mean(dices), len(ious)

# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------

def train():
    print("\nPHASE 3B — HEAD-ONLY FINETUNING (TERMINAL LOGGING)")
    print("=" * 72)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

    # Freeze encoders
    for name, p in model.named_parameters():
        p.requires_grad = "decoder" in name

    model.to(device)
    model.train()

    cracks_train = DrywallDataset(
        DATA_ROOT / "cracks/train/images",
        DATA_ROOT / "cracks/train/masks",
        "drywall crack",
        processor
    )

    taping_train = DrywallDataset(
        DATA_ROOT / "taping/train/images",
        DATA_ROOT / "taping/train/masks",
        "drywall joint tape",
        processor
    )

    train_loader = DataLoader(
        ConcatDataset([cracks_train, taping_train]),
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )

    cracks_val = DrywallDataset(
        DATA_ROOT / "cracks/valid/images",
        DATA_ROOT / "cracks/valid/masks",
        "drywall crack",
        processor
    )

    taping_val = DrywallDataset(
        DATA_ROOT / "taping/valid/images",
        DATA_ROOT / "taping/valid/masks",
        "drywall joint tape",
        processor
    )

    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    best_iou = -1
    patience, wait = 3, 0

    for epoch in range(20):
        total_loss = 0

        for batch in train_loader:
            pv = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                logits = model(pixel_values=pv, input_ids=ids, attention_mask=am).logits
                logits = torch.nn.functional.interpolate(
                    logits.unsqueeze(1),
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                ).squeeze(1)

                bce = nn.BCEWithLogitsLoss()(logits, labels)
                dice = dice_loss(torch.sigmoid(logits), labels)
                loss = bce + dice

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()

        c_iou, c_dice, _ = validate(model, cracks_val, device)
        t_iou, t_dice, _ = validate(model, taping_val, device)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Loss {total_loss:.3f} | "
            f"Cracks IoU {c_iou:.3f} Dice {c_dice:.3f} | "
            f"Taping IoU {t_iou:.3f} Dice {t_dice:.3f}"
        )

        mean_iou = (c_iou + t_iou) / 2
        if mean_iou > best_iou:
            best_iou = mean_iou
            wait = 0
            torch.save(model.state_dict(), OUT_ROOT / "best_model.pt")
            print("  ↑ New best checkpoint saved")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered")
                break

    print("\nFINAL BEST mIoU:", best_iou)
    print("Checkpoint:", OUT_ROOT / "best_model.pt")

# ---------------------------------------------------------------------

if __name__ == "__main__":
    train()
