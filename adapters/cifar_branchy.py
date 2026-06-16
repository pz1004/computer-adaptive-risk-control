"""
Build a per-exit cache for a branchy CIFAR ResNet.

Real run example:
  python -m adapters.cifar_branchy --dataset cifar100 --download --epochs 80 \
    --checkpoint checkpoints/cifar_branchy_resnet56.pt --out cache/cifar_branchy_resnet56.npz

Feasible neural benchmark example:
  python -m adapters.cifar_branchy --dataset cifar10 --download --epochs 30 \
    --checkpoint checkpoints/cifar10_branchy_resnet56.pt --out cache/cifar10_branchy_resnet56.npz

Smoke run example:
  python -m adapters.cifar_branchy --dataset fake --epochs 0 --allow-random-init \
    --max-train-samples 64 --max-cache-samples 64 --batch-size 16 \
    --threshold-count 8 --device cpu --out cache/smoke_cifar_branchy.npz
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from carc.chain import build_chain


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def git_sha() -> str | None:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except Exception:
        return None
    return proc.stdout.strip()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ExitHead(nn.Module):
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        return self.fc(x)


class BranchyResNet56(nn.Module):
    """CIFAR ResNet-56 backbone with exits after stem, stage1, stage2, and stage3."""

    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, 16, blocks=9, stride=1)
        self.layer2 = self._make_layer(16, 32, blocks=9, stride=2)
        self.layer3 = self._make_layer(32, 64, blocks=9, stride=2)
        self.exits = nn.ModuleList([
            ExitHead(16, num_classes),
            ExitHead(16, num_classes),
            ExitHead(32, num_classes),
            ExitHead(64, num_classes),
        ])

    @staticmethod
    def _make_layer(in_planes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_planes, planes, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = self.relu(self.bn1(self.conv1(x)))
        logits0 = self.exits[0](out)
        out = self.layer1(out)
        logits1 = self.exits[1](out)
        out = self.layer2(out)
        logits2 = self.exits[2](out)
        out = self.layer3(out)
        logits3 = self.exits[3](out)
        return [logits0, logits1, logits2, logits3]


def conv_macs(h: int, w: int, in_ch: int, out_ch: int, kernel: int, stride: int = 1) -> tuple[int, int, float]:
    out_h = h // stride
    out_w = w // stride
    return out_h, out_w, float(out_h * out_w * out_ch * in_ch * kernel * kernel)


def estimate_exit_costs(num_classes: int = 100) -> np.ndarray:
    """Analytical multiply-add estimate for the four exits."""
    h = w = 32
    total = 0.0
    h, w, mac = conv_macs(h, w, 3, 16, 3)
    total += mac
    stem_cost = total + 16 * num_classes

    for _ in range(9):
        _, _, mac = conv_macs(32, 32, 16, 16, 3)
        total += mac
        _, _, mac = conv_macs(32, 32, 16, 16, 3)
        total += mac
    stage1_cost = total + 16 * num_classes

    h, w, mac = conv_macs(32, 32, 16, 32, 3, stride=2)
    total += mac
    _, _, mac = conv_macs(16, 16, 32, 32, 3)
    total += mac
    _, _, mac = conv_macs(32, 32, 16, 32, 1, stride=2)
    total += mac
    for _ in range(8):
        _, _, mac = conv_macs(16, 16, 32, 32, 3)
        total += mac
        _, _, mac = conv_macs(16, 16, 32, 32, 3)
        total += mac
    stage2_cost = total + 32 * num_classes

    h, w, mac = conv_macs(16, 16, 32, 64, 3, stride=2)
    total += mac
    _, _, mac = conv_macs(8, 8, 64, 64, 3)
    total += mac
    _, _, mac = conv_macs(16, 16, 32, 64, 1, stride=2)
    total += mac
    for _ in range(8):
        _, _, mac = conv_macs(8, 8, 64, 64, 3)
        total += mac
        _, _, mac = conv_macs(8, 8, 64, 64, 3)
        total += mac
    stage3_cost = total + 64 * num_classes
    return np.array([stem_cost, stage1_cost, stage2_cost, stage3_cost], dtype=float)


def subset_dataset(dataset, max_samples: int | None, seed: int):
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=max_samples, replace=False)
    return Subset(dataset, idx.tolist())


def dataset_spec(name: str) -> tuple[type, tuple[float, float, float], tuple[float, float, float], int]:
    if name == "cifar100":
        return datasets.CIFAR100, CIFAR100_MEAN, CIFAR100_STD, 100
    if name == "cifar10":
        return datasets.CIFAR10, CIFAR10_MEAN, CIFAR10_STD, 10
    if name == "fake":
        return datasets.FakeData, CIFAR100_MEAN, CIFAR100_STD, 100
    raise ValueError(f"unknown dataset {name}")


def make_datasets(args):
    dataset_cls, mean, std, num_classes = dataset_spec(args.dataset)
    if args.dataset in {"cifar100", "cifar10"}:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        eval_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train = dataset_cls(args.data_root, train=True, download=args.download, transform=train_tf)
        cache = dataset_cls(args.data_root, train=False, download=args.download, transform=eval_tf)
    elif args.dataset == "fake":
        tf = transforms.ToTensor()
        train_size = args.max_train_samples if args.max_train_samples and args.max_train_samples > 0 else 128
        cache_size = args.max_cache_samples if args.max_cache_samples and args.max_cache_samples > 0 else 128
        train = dataset_cls(size=train_size, image_size=(3, 32, 32), num_classes=num_classes, transform=tf)
        cache = dataset_cls(size=cache_size, image_size=(3, 32, 32), num_classes=num_classes, transform=tf, random_offset=10_000)
    else:
        raise ValueError(f"unknown dataset {args.dataset}")
    return (
        subset_dataset(train, args.max_train_samples, args.seed),
        subset_dataset(cache, args.max_cache_samples, args.seed + 1),
    )


def save_checkpoint(path: Path, model: nn.Module, args, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, path)


def train(
    model: nn.Module,
    loader: DataLoader,
    args,
    device: torch.device,
    checkpoint_path: Path,
    start_epoch: int = 0,
) -> None:
    if args.epochs <= start_epoch:
        if start_epoch > 0 and args.epochs > 0:
            print(f"checkpoint already at epoch {start_epoch}; requested {args.epochs}, skipping training")
        return
    weights = torch.tensor([float(x) for x in args.exit_loss_weights.split(",")], device=device)
    if weights.numel() != 4:
        raise ValueError("--exit-loss-weights must contain 4 comma-separated values")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    remaining_epochs = args.epochs - start_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        total_loss = 0.0
        total = 0
        final_correct = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            losses = torch.stack([criterion(out, labels) for out in logits])
            loss = (weights * losses).sum() / weights.sum()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * labels.size(0)
            total += labels.size(0)
            final_correct += int((logits[-1].argmax(dim=1) == labels).sum())
        scheduler.step()
        print(f"epoch {epoch:03d}/{args.epochs} loss={total_loss / max(total, 1):.4f} "
              f"final_train_acc={final_correct / max(total, 1):.4f}")
        if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
            save_checkpoint(checkpoint_path, model, args, epoch)
            print(f"saved checkpoint {checkpoint_path} at epoch {epoch}")


@torch.no_grad()
def collect_cache(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    scores_all = []
    correct_all = []
    loss_all = []
    labels_all = []
    pred_all = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        batch_scores = []
        batch_correct = []
        batch_preds = []
        for out in logits:
            probs = torch.softmax(out, dim=1)
            scores, preds = probs.max(dim=1)
            correct = preds.eq(labels)
            batch_scores.append(scores.cpu().numpy())
            batch_correct.append(correct.cpu().numpy().astype(np.float32))
            batch_preds.append(preds.cpu().numpy())
        scores = np.stack(batch_scores, axis=1)
        correct = np.stack(batch_correct, axis=1)
        preds = np.stack(batch_preds, axis=1)
        scores_all.append(scores)
        correct_all.append(correct)
        loss_all.append(1.0 - correct)
        labels_all.append(labels.cpu().numpy())
        pred_all.append(preds)
    return (
        np.concatenate(scores_all, axis=0),
        np.concatenate(correct_all, axis=0),
        np.concatenate(loss_all, axis=0),
        np.concatenate(labels_all, axis=0),
        np.concatenate(pred_all, axis=0),
    )


def validate_cache(scores, loss, exit_costs, thresholds, loss_matrix, cost_matrix) -> dict:
    checks = {
        "n": int(scores.shape[0]),
        "num_exits": int(scores.shape[1]),
        "num_thresholds": int(thresholds.size),
        "scores_in_unit_interval": bool(np.all((scores >= 0.0) & (scores <= 1.0))),
        "loss_in_unit_interval": bool(np.all((loss >= 0.0) & (loss <= 1.0))),
        "exit_cost_strictly_increasing": bool(np.all(np.diff(exit_costs) > 0.0)),
        "chain_cost_non_decreasing": bool(np.all(np.diff(cost_matrix.mean(axis=0)) >= -1e-9)),
        "loss_matrix_shape": list(loss_matrix.shape),
        "cost_matrix_shape": list(cost_matrix.shape),
    }
    if not all(v for k, v in checks.items() if isinstance(v, bool)):
        raise ValueError(f"cache validation failed: {checks}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "fake"], default="cifar100")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out", default="cache/cifar_branchy_resnet56.npz")
    parser.add_argument("--checkpoint", default="checkpoints/cifar_branchy_resnet56.pt")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-cache-samples", type=int, default=0)
    parser.add_argument("--threshold-count", type=int, default=100)
    parser.add_argument("--threshold-low", type=float, default=0.0)
    parser.add_argument("--threshold-high", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--exit-loss-weights", default="0.2,0.3,0.5,1.0")
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()

    if args.dataset == "cifar10":
        if args.checkpoint == parser.get_default("checkpoint"):
            args.checkpoint = "checkpoints/cifar10_branchy_resnet56.pt"
        if args.out == parser.get_default("out"):
            args.out = "cache/cifar10_branchy_resnet56.npz"

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    _dataset_cls, _mean, _std, num_classes = dataset_spec(args.dataset)
    train_set, cache_set = make_datasets(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    cache_loader = DataLoader(cache_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = BranchyResNet56(num_classes=num_classes).to(device)
    ckpt_path = Path(args.checkpoint)
    loaded_checkpoint = False
    checkpoint_epoch = 0
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        checkpoint_epoch = int(state.get("epoch", 0)) if isinstance(state, dict) else 0
        loaded_checkpoint = True
        print(f"loaded checkpoint {ckpt_path} at epoch {checkpoint_epoch}")
    elif args.epochs <= 0 and args.dataset != "fake" and not args.allow_random_init:
        raise SystemExit(
            "No checkpoint found and --epochs <= 0. For a real CIFAR cache, provide a checkpoint "
            "or train with --epochs > 0. Use --allow-random-init only for smoke tests."
        )

    train(model, train_loader, args, device, ckpt_path, checkpoint_epoch)
    completed_epoch = int(args.epochs if args.epochs > checkpoint_epoch else checkpoint_epoch)

    scores, correct, loss, labels, preds = collect_cache(model, cache_loader, device)
    exit_costs = estimate_exit_costs(num_classes=num_classes)
    thresholds = np.linspace(args.threshold_low, args.threshold_high, args.threshold_count, dtype=float)
    loss_matrix, cost_matrix = build_chain(scores, loss, exit_costs, thresholds)
    validation = validate_cache(scores, loss, exit_costs, thresholds, loss_matrix, cost_matrix)
    per_exit_accuracy = correct.mean(axis=0)
    meta = {
        "dataset": args.dataset,
        "model": "branchy_resnet56_cifar",
        "num_classes": int(num_classes),
        "seed": args.seed,
        "git_sha": git_sha(),
        "device": str(device),
        "checkpoint": str(ckpt_path),
        "loaded_checkpoint": loaded_checkpoint,
        "loaded_checkpoint_epoch": checkpoint_epoch,
        "checkpoint_epoch": completed_epoch,
        "completed_epoch": completed_epoch,
        "epochs": args.epochs,
        "train_size": len(train_set),
        "cache_size": len(cache_set),
        "num_exits": int(scores.shape[1]),
        "threshold_count": args.threshold_count,
        "threshold_low": args.threshold_low,
        "threshold_high": args.threshold_high,
        "exit_cost_kind": "analytical_conv_linear_macs",
        "exit_macs": exit_costs.tolist(),
        "per_exit_accuracy": per_exit_accuracy.tolist(),
        "validation": validation,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        scores=scores.astype(np.float32),
        correct=correct.astype(np.float32),
        loss=loss.astype(np.float32),
        labels=labels.astype(np.int64),
        preds=preds.astype(np.int64),
        exit_cost=exit_costs.astype(np.float64),
        thresholds=thresholds.astype(np.float64),
        loss_matrix=loss_matrix.astype(np.float32),
        cost_matrix=cost_matrix.astype(np.float64),
        meta_json=json.dumps(meta, sort_keys=True),
    )
    print(f"wrote {out}")
    print(f"cache n={scores.shape[0]} exits={scores.shape[1]} thresholds={thresholds.size}")
    print("per-exit accuracy:", " ".join(f"{x:.4f}" for x in per_exit_accuracy))
    print("exit MACs:", " ".join(f"{x:.0f}" for x in exit_costs))


if __name__ == "__main__":
    main()
