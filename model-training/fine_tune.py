"""
AASIST Fine-tuning 스크립트
공식 가중치(ASVspoof 2019 LA) 기반 transfer learning

Label convention (ASVspoof):
  - 0 = spoof (fake)
  - 1 = bonafide (real)
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.AASIST import Model


class AudioDataset(Dataset):
    def __init__(self, csv_path: Path, nb_samp: int = 64600, augment: bool = False):
        self.nb_samp = nb_samp
        self.augment = augment
        self.samples = []

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "filepath": row["filepath"],
                    "label": int(row["label"]),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        import torchaudio

        waveform, sr = torchaudio.load(item["filepath"])
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)  # (samples,)

        # Pad or crop
        if waveform.shape[0] < self.nb_samp:
            waveform = F.pad(waveform, (0, self.nb_samp - waveform.shape[0]))
        elif waveform.shape[0] > self.nb_samp:
            start = random.randint(0, waveform.shape[0] - self.nb_samp) if self.augment else 0
            waveform = waveform[start : start + self.nb_samp]

        # Simple augmentation
        if self.augment:
            # Random gain
            gain = random.uniform(0.8, 1.2)
            waveform = waveform * gain
            # Random noise injection
            if random.random() < 0.3:
                noise = torch.randn_like(waveform) * random.uniform(0.001, 0.005)
                waveform = waveform + noise

        return waveform, item["label"]  # (nb_samp,), label


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def compute_class_weights(dataset: AudioDataset) -> torch.Tensor:
    """클래스 불균형 보정 가중치 계산"""
    labels = [s["label"] for s in dataset.samples]
    counts = [0, 0]
    for label in labels:
        counts[label] += 1
    total = len(labels)
    weights = torch.tensor([total / (2 * c) if c > 0 else 1.0 for c in counts])
    return weights


def make_balanced_sampler(dataset: AudioDataset) -> WeightedRandomSampler:
    """불균형 데이터를 위한 weighted sampler"""
    labels = [s["label"] for s in dataset.samples]
    counts = [0, 0]
    for label in labels:
        counts[label] += 1
    weight_per_class = [1.0 / c if c > 0 else 0.0 for c in counts]
    sample_weights = [weight_per_class[label] for label in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


def freeze_early_layers(model: nn.Module):
    """GAT + output layer만 학습, 나머지 freeze"""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(
            ("out_layer", "pool_", "HtrgGAT", "GAT_layer")
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,}/{total:,} (GAT + output layers)")


def unfreeze_all(model: nn.Module):
    """모든 레이어 unfreeze"""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  All unfrozen: {trainable:,} trainable params")


def run_validation(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            _, logits, _ = model(waveforms)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total if total > 0 else 0


def train(args):
    device = get_device(args.device)
    print(f"Device: {device}")

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    # Build model
    model = Model(config["model_config"])
    ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=True)
    result = model.load_state_dict(ckpt, strict=False)
    print(f"Loaded pretrained weights: missing={len(result.missing_keys)}, "
          f"unexpected={len(result.unexpected_keys)}")
    model = model.to(device)

    # Datasets
    train_ds = AudioDataset(args.train_csv, nb_samp=64600, augment=True)
    val_ds = AudioDataset(args.val_csv, nb_samp=64600, augment=False)
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    sampler = make_balanced_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    # Class-weighted loss
    class_weights = compute_class_weights(train_ds).to(device)
    print(f"Class weights: spoof={class_weights[0]:.2f}, bonafide={class_weights[1]:.2f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        # Freeze strategy
        if epoch <= args.freeze_epochs:
            if epoch == 1:
                print(f"\n--- Freeze phase (epoch 1-{args.freeze_epochs}) ---")
                freeze_early_layers(model)
                # Re-init optimizer with only trainable params
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=args.lr, weight_decay=args.weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
                )
        elif epoch == args.freeze_epochs + 1:
            print(f"\n--- Unfreeze phase (epoch {epoch}-{args.epochs}) ---")
            unfreeze_all(model)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.freeze_epochs, eta_min=args.lr * 0.001
            )

        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for waveforms, labels in train_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            _, logits, _ = model(waveforms)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        val_loss, val_acc = run_validation(model, val_loader, criterion, device)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={lr:.2e}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_path = output_dir / "aasist_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

    # Save final
    final_path = output_dir / "aasist_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})")
    print(f"Best model: {output_dir / 'aasist_best.pth'}")
    print(f"Final model: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune AASIST model")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=str(Path(__file__).resolve().parent / "weights" / "official" / "AASIST.pth"),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "aasist.json"),
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "train.csv"),
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "val.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "weights" / "finetuned"),
    )
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-epochs", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
