import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms


CLASS_NAMES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
]

HARD_CLASSES = {"paper", "cardboard", "plastic", "trash", "glass"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path = self.samples[index][0]
        return image, target, path


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_grad_scaler(device):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        except TypeError:
            return torch.amp.GradScaler(enabled=device.type == "cuda")
    return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")


def autocast_context(device):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda")
        except TypeError:
            return torch.cuda.amp.autocast(enabled=device.type == "cuda")
    return torch.cuda.amp.autocast(enabled=device.type == "cuda")


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def find_data_dir(user_data_dir):
    if user_data_dir:
        data_dir = Path(user_data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data dir does not exist: {data_dir}")
        return data_dir

    for root in [Path("/kaggle/input"), Path("/kaggle/working"), Path("data"), Path(".")]:
        if not root.exists():
            continue
        for path in root.rglob("split_dataset"):
            if (path / "train").is_dir() and (path / "val").is_dir() and (path / "test").is_dir():
                return path

    raise FileNotFoundError("Could not find split_dataset. Pass --data-dir /path/to/split_dataset.")


def build_transforms(img_size, strong_aug=False):
    base_aug = [
        transforms.RandomResizedCrop(img_size, scale=(0.68 if strong_aug else 0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(25 if strong_aug else 15),
        transforms.ColorJitter(
            brightness=0.32 if strong_aug else 0.2,
            contrast=0.32 if strong_aug else 0.2,
            saturation=0.32 if strong_aug else 0.2,
            hue=0.08 if strong_aug else 0.05,
        ),
    ]
    if strong_aug:
        base_aug.extend(
            [
                transforms.RandomPerspective(distortion_scale=0.18, p=0.25),
                transforms.RandomAutocontrast(p=0.2),
            ]
        )
    base_aug.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.35 if strong_aug else 0.25, scale=(0.02, 0.16), ratio=(0.3, 3.3)),
        ]
    )

    train_transform = transforms.Compose(base_aug)
    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def load_hard_example_basenames(path):
    if not path:
        return set()
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Hard-example CSV does not exist: {csv_path}")
    basenames = set()
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("filename") or Path(row.get("path", "")).name
            if name:
                basenames.add(name)
    return basenames


def build_dataloaders(data_dir, img_size, batch_size, num_workers, hard_class_weight, hard_example_csv, supplement_dir=None):
    train_transform, eval_transform = build_transforms(img_size, strong_aug=True)
    base_train_dataset = ImageFolderWithPaths(data_dir / "train", transform=train_transform)
    val_dataset = ImageFolderWithPaths(data_dir / "val", transform=eval_transform)
    test_dataset = ImageFolderWithPaths(data_dir / "test", transform=eval_transform)

    if base_train_dataset.classes != CLASS_NAMES:
        print("Warning: class order from ImageFolder differs from CLASS_NAMES.")
        print("ImageFolder classes:", base_train_dataset.classes)

    train_parts = [base_train_dataset]
    all_samples = list(base_train_dataset.samples)
    supplement_count = 0
    if supplement_dir:
        supplement_path = Path(supplement_dir)
        if not supplement_path.exists():
            raise FileNotFoundError(f"Supplement dir does not exist: {supplement_path}")
        supplement_dataset = ImageFolderWithPaths(supplement_path, transform=train_transform)

        supplement_extra_classes = sorted(set(supplement_dataset.classes) - set(base_train_dataset.classes))
        if supplement_extra_classes:
            raise ValueError(
                f"Supplement contains classes not found in train data: {supplement_extra_classes}"
            )

        class_to_idx = base_train_dataset.class_to_idx
        remapped_samples = []
        for sample_path, _ in supplement_dataset.samples:
            class_name = Path(sample_path).parent.name
            remapped_samples.append((sample_path, class_to_idx[class_name]))

        supplement_dataset.classes = base_train_dataset.classes
        supplement_dataset.class_to_idx = class_to_idx
        supplement_dataset.samples = remapped_samples
        supplement_dataset.imgs = remapped_samples
        supplement_dataset.targets = [target for _, target in remapped_samples]

        train_parts.append(supplement_dataset)
        all_samples.extend(supplement_dataset.samples)
        supplement_count = len(supplement_dataset)
        print(f"Using supplement samples: {supplement_count} from {supplement_path}")

    train_dataset = ConcatDataset(train_parts) if len(train_parts) > 1 else base_train_dataset
    train_dataset.classes = base_train_dataset.classes
    train_dataset.samples = all_samples

    hard_basenames = load_hard_example_basenames(hard_example_csv)
    targets = [target for _, target in all_samples]
    class_counts = np.bincount(targets, minlength=len(base_train_dataset.classes))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = []
    for path, target in all_samples:
        class_name = base_train_dataset.classes[target]
        weight = class_weights[target]
        if class_name in HARD_CLASSES:
            weight *= hard_class_weight
        if Path(path).name in hard_basenames:
            weight *= 2.0
        sample_weights.append(weight)

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, class_counts, supplement_count


def build_model(model_name, num_classes, pretrained=True):
    model_name = model_name.lower()
    if model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if model_name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def build_criterion(loss_name, class_counts, label_smoothing, focal_gamma, device):
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    if loss_name == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights, label_smoothing=label_smoothing)
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_count += batch_size
    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_targets, all_preds, all_probs, all_paths = [], [], [], []
    for images, targets, paths in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == targets).sum().item()
        total_count += batch_size
        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_paths.extend(paths)
    return {
        "loss": total_loss / total_count,
        "acc": total_correct / total_count,
        "targets": all_targets,
        "preds": all_preds,
        "probs": all_probs,
        "paths": all_paths,
    }


def compute_auc_metrics(y_true, y_prob, class_names):
    y_prob = np.asarray(y_prob)
    labels = list(range(len(class_names)))
    metrics = {}
    metrics["roc_auc_macro_ovr"] = float(
        roc_auc_score(y_true, y_prob, labels=labels, multi_class="ovr", average="macro")
    )
    metrics["roc_auc_weighted_ovr"] = float(
        roc_auc_score(y_true, y_prob, labels=labels, multi_class="ovr", average="weighted")
    )
    per_class_auc = {}
    for class_idx, class_name in enumerate(class_names):
        binary_true = np.asarray(y_true) == class_idx
        per_class_auc[class_name] = float(roc_auc_score(binary_true.astype(int), y_prob[:, class_idx]))
    metrics["roc_auc_per_class_ovr"] = per_class_auc
    return metrics


def write_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def save_checkpoint(path, model, model_name, class_names, img_size, epoch, val_acc, version):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "class_names": class_names,
            "img_size": img_size,
            "epoch": epoch,
            "val_acc": val_acc,
            "version": version,
        },
        path,
    )


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix - v2",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_misclassified_csv(paths, y_true, y_pred, y_prob, class_names, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "true_label", "pred_label", "confidence"])
        writer.writeheader()
        for path, true_idx, pred_idx, probs in zip(paths, y_true, y_pred, y_prob):
            if true_idx != pred_idx:
                writer.writerow(
                    {
                        "path": path,
                        "true_label": class_names[true_idx],
                        "pred_label": class_names[pred_idx],
                        "confidence": float(np.max(probs)),
                    }
                )


def zip_output_dir(output_dir):
    zip_base = output_dir.parent / f"{output_dir.name}_outputs"
    return Path(shutil.make_archive(str(zip_base), "zip", root_dir=output_dir.parent, base_dir=output_dir.name))


def main():
    parser = argparse.ArgumentParser(description="Train optimized v2 garbage classifier.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/runs/convnext_tiny_v2")
    parser.add_argument("--model", type=str, default="convnext_tiny")
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--loss", choices=["focal", "cross_entropy"], default="focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--hard-class-weight", type=float, default=1.8)
    parser.add_argument("--hard-example-csv", type=str, default=None)
    parser.add_argument("--supplement-dir", type=str, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-zip-output", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = find_data_dir(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, _, _, train_loader, val_loader, test_loader, class_counts, supplement_count = build_dataloaders(
        data_dir=data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hard_class_weight=args.hard_class_weight,
        hard_example_csv=args.hard_example_csv,
        supplement_dir=args.supplement_dir,
    )
    class_names = train_dataset.classes
    write_json(output_dir / "class_names.json", class_names)

    model = build_model(args.model, len(class_names), pretrained=not args.no_pretrained).to(device)
    criterion = build_criterion(args.loss, class_counts, args.label_smoothing, args.focal_gamma, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = make_grad_scaler(device)

    best_val_acc = 0.0
    epochs_without_improvement = 0
    best_path = output_dir / "best_model_v2.pt"
    log_rows = []

    print(f"Device: {device}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Loss: {args.loss}")
    print(f"Hard classes: {sorted(HARD_CLASSES)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_result = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        improved = val_result["acc"] > best_val_acc + args.min_delta
        if improved:
            best_val_acc = val_result["acc"]
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, args.model, class_names, args.img_size, epoch, best_val_acc, "v2")
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_result["loss"],
            "val_acc": val_result["acc"],
            "lr": optimizer.param_groups[0]["lr"],
            "saved_best": improved,
        }
        log_rows.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_result['loss']:.4f} val_acc={val_result['acc']:.4f} "
            f"best_val_acc={best_val_acc:.4f}"
        )
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    with (output_dir / "train_log_v2.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    checkpoint = load_checkpoint(best_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_result = evaluate(model, test_loader, criterion, device)
    report_text = classification_report(
        test_result["targets"],
        test_result["preds"],
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        test_result["targets"],
        test_result["preds"],
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    auc_metrics = compute_auc_metrics(test_result["targets"], test_result["probs"], class_names)
    metrics = {
        "version": "v2",
        "best_val_acc": best_val_acc,
        "test_loss": test_result["loss"],
        "test_acc": test_result["acc"],
        **auc_metrics,
        "classification_report": report_dict,
        "model": args.model,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_ran": len(log_rows),
        "seed": args.seed,
        "loss": args.loss,
        "hard_class_weight": args.hard_class_weight,
        "label_smoothing": args.label_smoothing,
        "supplement_dir": args.supplement_dir,
        "supplement_count": supplement_count,
    }
    write_json(output_dir / "test_metrics_v2.json", metrics)
    write_json(output_dir / "metrics_v2.json", metrics)
    with (output_dir / "classification_report_v2.txt").open("w", encoding="utf-8") as f:
        f.write(report_text)
        f.write("\n")
    plot_confusion_matrix(test_result["targets"], test_result["preds"], class_names, output_dir / "confusion_matrix_v2.png")
    save_misclassified_csv(
        test_result["paths"],
        test_result["targets"],
        test_result["preds"],
        test_result["probs"],
        class_names,
        output_dir / "misclassified_samples_v2.csv",
    )

    print("\nV2 test result")
    print(f"test_loss={test_result['loss']:.4f} test_acc={test_result['acc']:.4f}")
    print(f"roc_auc_macro_ovr={auc_metrics['roc_auc_macro_ovr']:.4f}")
    print(f"roc_auc_weighted_ovr={auc_metrics['roc_auc_weighted_ovr']:.4f}")
    print(report_text)
    if not args.no_zip_output:
        print(f"Zipped artifacts saved to: {zip_output_dir(output_dir)}")


if __name__ == "__main__":
    main()
