import argparse
import csv
import json
import random
import shutil
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path = self.samples[index][0]
        return image, target, path


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
            return torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            )
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

    candidates = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path("data"),
        Path("."),
    ]

    for root in candidates:
        if not root.exists():
            continue
        for path in root.rglob("split_dataset"):
            if (path / "train").is_dir() and (path / "val").is_dir() and (path / "test").is_dir():
                return path

    raise FileNotFoundError(
        "Could not find split_dataset. Pass --data-dir /path/to/split_dataset."
    )


def build_transforms(img_size):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform


def build_dataloaders(data_dir, img_size, batch_size, num_workers, weighted_sampler):
    train_transform, eval_transform = build_transforms(img_size)

    train_dataset = ImageFolderWithPaths(data_dir / "train", transform=train_transform)
    val_dataset = ImageFolderWithPaths(data_dir / "val", transform=eval_transform)
    test_dataset = ImageFolderWithPaths(data_dir / "test", transform=eval_transform)

    if train_dataset.classes != CLASS_NAMES:
        print("Warning: class order from ImageFolder differs from CLASS_NAMES.")
        print("ImageFolder classes:", train_dataset.classes)

    sampler = None
    shuffle = True
    if weighted_sampler:
        targets = [target for _, target in train_dataset.samples]
        class_counts = np.bincount(targets, minlength=len(train_dataset.classes))
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = [class_weights[target] for target in targets]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_dataset, shuffle=shuffle, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def build_model(model_name, num_classes, pretrained=True):
    model_name = model_name.lower()

    if model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            model = models.convnext_tiny(weights=weights)
        except Exception as exc:
            print(f"Warning: failed to load pretrained weights: {exc}")
            print("Falling back to random initialization.")
            model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            model = models.efficientnet_b0(weights=weights)
        except Exception as exc:
            print(f"Warning: failed to load pretrained weights: {exc}")
            print("Falling back to random initialization.")
            model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            model = models.efficientnet_b3(weights=weights)
        except Exception as exc:
            print(f"Warning: failed to load pretrained weights: {exc}")
            print("Falling back to random initialization.")
            model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        try:
            model = models.resnet50(weights=weights)
        except Exception as exc:
            print(f"Warning: failed to load pretrained weights: {exc}")
            print("Falling back to random initialization.")
            model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def load_partial_weights(model, weights_path, device):
    checkpoint = load_checkpoint(weights_path, device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()

    matched = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            matched[key] = value
        else:
            skipped.append(key)

    model_state.update(matched)
    model.load_state_dict(model_state)
    print(f"Loaded {len(matched)} tensors from {weights_path}.")
    if skipped:
        print(f"Skipped {len(skipped)} tensors with missing keys or mismatched shapes.")


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
        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * batch_size
        total_correct += (preds == targets).sum().item()
        total_count += batch_size

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_targets = []
    all_preds = []
    all_probs = []
    all_paths = []

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


def save_checkpoint(path, model, model_name, class_names, img_size, epoch, val_acc):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "class_names": class_names,
        "img_size": img_size,
        "epoch": epoch,
        "val_acc": val_acc,
    }
    torch.save(checkpoint, path)


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
        title="Confusion Matrix",
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


def save_misclassified_csv(paths, y_true, y_pred, class_names, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "true_label", "pred_label"],
        )
        writer.writeheader()
        for path, true_idx, pred_idx in zip(paths, y_true, y_pred):
            if true_idx != pred_idx:
                writer.writerow(
                    {
                        "path": path,
                        "true_label": class_names[true_idx],
                        "pred_label": class_names[pred_idx],
                    }
                )


def compute_auc_metrics(y_true, y_prob, class_names):
    y_prob = np.asarray(y_prob)
    labels = list(range(len(class_names)))
    metrics = {}

    try:
        metrics["roc_auc_macro_ovr"] = float(
            roc_auc_score(
                y_true,
                y_prob,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
        metrics["roc_auc_weighted_ovr"] = float(
            roc_auc_score(
                y_true,
                y_prob,
                labels=labels,
                multi_class="ovr",
                average="weighted",
            )
        )
    except ValueError as exc:
        print(f"Warning: could not compute multiclass ROC-AUC: {exc}")
        metrics["roc_auc_macro_ovr"] = None
        metrics["roc_auc_weighted_ovr"] = None

    per_class_auc = {}
    for class_idx, class_name in enumerate(class_names):
        binary_true = np.asarray(y_true) == class_idx
        if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
            per_class_auc[class_name] = None
            continue
        per_class_auc[class_name] = float(
            roc_auc_score(binary_true.astype(int), y_prob[:, class_idx])
        )

    metrics["roc_auc_per_class_ovr"] = per_class_auc
    return metrics


def write_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def zip_output_dir(output_dir):
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        zip_base = kaggle_working / f"{output_dir.name}_outputs"
    else:
        zip_base = output_dir.parent / f"{output_dir.name}_outputs"

    zip_path = Path(
        shutil.make_archive(
            base_name=str(zip_base),
            format="zip",
            root_dir=output_dir.parent,
            base_dir=output_dir.name,
        )
    )
    return zip_path


def main():
    if '-f' in sys.argv:
        idx = sys.argv.index('-f')
        sys.argv.pop(idx)
        if idx < len(sys.argv):
            sys.argv.pop(idx)

    parser = argparse.ArgumentParser(description="Train a garbage image classifier on Kaggle.")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/runs/convnext_tiny")
    parser.add_argument("--model", type=str, default="convnext_tiny")
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--weights-path", type=str, default=None)
    parser.add_argument("--no-zip-output", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = find_data_dir(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(
        data_dir=data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        weighted_sampler=args.weighted_sampler,
    )

    class_names = train_dataset.classes
    write_json(output_dir / "class_names.json", class_names)

    model = build_model(
        args.model,
        num_classes=len(class_names),
        pretrained=not args.no_pretrained,
    ).to(device)
    if args.weights_path:
        load_partial_weights(model, args.weights_path, device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    scaler = make_grad_scaler(device)

    best_val_acc = 0.0
    best_path = output_dir / "best_model.pt"
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
        )
        val_result = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_result["loss"],
            "val_acc": val_result["acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        log_rows.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_result['loss']:.4f} val_acc={val_result['acc']:.4f}"
        )

        if val_result["acc"] > best_val_acc:
            best_val_acc = val_result["acc"]
            save_checkpoint(
                best_path,
                model,
                model_name=args.model,
                class_names=class_names,
                img_size=args.img_size,
                epoch=epoch,
                val_acc=best_val_acc,
            )
            print(f"Saved best model: {best_path} val_acc={best_val_acc:.4f}")

    with (output_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
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
    auc_metrics = compute_auc_metrics(
        test_result["targets"],
        test_result["probs"],
        class_names,
    )

    metrics = {
        "best_val_acc": best_val_acc,
        "test_loss": test_result["loss"],
        "test_acc": test_result["acc"],
        **auc_metrics,
        "classification_report": report_dict,
        "model": args.model,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
    }
    write_json(output_dir / "metrics.json", metrics)

    with (output_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(report_text)
        f.write("\n")

    plot_confusion_matrix(
        test_result["targets"],
        test_result["preds"],
        class_names,
        output_dir / "confusion_matrix.png",
    )
    save_misclassified_csv(
        test_result["paths"],
        test_result["targets"],
        test_result["preds"],
        class_names,
        output_dir / "misclassified_samples.csv",
    )

    print("\nTest result")
    print(f"test_loss={test_result['loss']:.4f} test_acc={test_result['acc']:.4f}")
    if auc_metrics["roc_auc_macro_ovr"] is not None:
        print(f"roc_auc_macro_ovr={auc_metrics['roc_auc_macro_ovr']:.4f}")
        print(f"roc_auc_weighted_ovr={auc_metrics['roc_auc_weighted_ovr']:.4f}")
    print(report_text)
    print(f"Artifacts saved to: {output_dir}")

    if not args.no_zip_output:
        zip_path = zip_output_dir(output_dir)
        print(f"Zipped artifacts saved to: {zip_path}")
        print("In a Kaggle notebook cell, run this to create a clickable download link:")
        print("from IPython.display import FileLink")
        print(f"FileLink('{zip_path}')")


if __name__ == "__main__":
    main()
