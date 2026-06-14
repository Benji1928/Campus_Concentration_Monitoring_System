"""
MobileViT-XXS Training Script.

V2: mobilevit_xxs backbone, AdaptiveAvgPool2d->Flatten->Linear head,
    CosineAnnealingLR, LR=2e-4, BS=32, dropout=0.1, weight_decay=1e-4,
    patience=5, optional class-weight loss, pre-split dataset support.

Usage:
    python src/classifiers/classification_pipeline/Transformer_training_mobilevit.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import timm
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent.parent

import sys
sys.path.insert(0, str(ROOT))

from src.classifiers.classification_pipeline.transformer_configs import MobileViTXXSConfig


CONFIG = MobileViTXXSConfig()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = ['ATTENTIVE', 'DROWSY', 'DISTRACTED']
CUSTOM_CLASS_TO_IDX = {'Attentive': 0, 'Drowsy': 1, 'Distracted': 2}


def load_datasets(train_transform, val_transform):
    if CONFIG.train_dir and CONFIG.val_dir:
        train_ds = datasets.ImageFolder(str(ROOT / CONFIG.train_dir), transform=train_transform)
        val_ds   = datasets.ImageFolder(str(ROOT / CONFIG.val_dir),   transform=val_transform)
        for ds in (train_ds, val_ds):
            ds.class_to_idx = CUSTOM_CLASS_TO_IDX
            ds.samples = [(p, CUSTOM_CLASS_TO_IDX[Path(p).parent.name]) for p, _ in ds.samples]
        return train_ds, val_ds

    dataset_dir = ROOT / CONFIG.dataset_dir
    probe = datasets.ImageFolder(root=str(dataset_dir))
    all_targets = np.array([s[1] for s in probe.samples])
    idx_remap = {probe.class_to_idx[k]: v
                 for k, v in CUSTOM_CLASS_TO_IDX.items()
                 if k in probe.class_to_idx}
    remapped = np.array([idx_remap.get(t, t) for t in all_targets])
    train_idx, val_idx = train_test_split(
        np.arange(len(probe)),
        test_size=CONFIG.test_size,
        stratify=remapped,
        random_state=42,
    )
    train_ds = Subset(datasets.ImageFolder(str(dataset_dir), transform=train_transform), train_idx)
    val_ds   = Subset(datasets.ImageFolder(str(dataset_dir), transform=val_transform),   val_idx)
    for ds in (train_ds.dataset, val_ds.dataset):
        ds.class_to_idx = CUSTOM_CLASS_TO_IDX
        ds.samples = [(p, CUSTOM_CLASS_TO_IDX[Path(p).parent.name]) for p, _ in ds.samples]
    return train_ds, val_ds


def main():
    out_dir = ROOT / 'models' / CONFIG.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Transforms ────────────────────────────────────────────────────
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    sz   = CONFIG.img_size  # 256 for MobileViT-XXS

    train_transform = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # ── Dataset ───────────────────────────────────────────────────────
    train_ds, val_ds = load_datasets(train_transform, val_transform)

    train_loader = DataLoader(train_ds, batch_size=CONFIG.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f'Train: {len(train_ds)}  |  Val: {len(val_ds)}')

    # ── Model ─────────────────────────────────────────────────────────
    print(f'\nLoading {CONFIG.model_name} (pretrained)...')
    model = timm.create_model(CONFIG.model_name, pretrained=True, num_classes=0)
    num_features = model.num_features
    model.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(num_features, CONFIG.num_classes),
    )
    model = model.to(DEVICE)
    print(f'  Features: {num_features} -> AdaptiveAvgPool2d -> Flatten -> {CONFIG.num_classes} classes')

    # ── Loss / Optimiser / Scheduler ──────────────────────────────────
    if CONFIG.use_class_weights:
        total = sum(CONFIG.class_counts)
        weights = [total / (CONFIG.num_classes * c) for c in CONFIG.class_counts]
        weight_tensor = torch.FloatTensor(weights).to(DEVICE)
        print(f'Class weights (Attentive, Sleepy, Distracted): {[round(w, 4) for w in weights]}')
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=CONFIG.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.lr,
                                  weight_decay=CONFIG.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
                     optimizer, T_max=CONFIG.epochs, eta_min=1e-6)
                 if CONFIG.use_cosine_lr else None)

    # ── Training loop ─────────────────────────────────────────────────
    best_val_loss = float('inf')
    patience_ctr  = 0
    best_val_acc  = 0.0
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    print(f'Training on {DEVICE}  |  epochs={CONFIG.epochs}  patience={CONFIG.patience}\n')

    for epoch in range(CONFIG.epochs):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{CONFIG.epochs}'):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * images.size(0)
            _, pred = torch.max(out, 1)
            total   += labels.size(0)
            correct += (pred == labels).sum().item()

        if scheduler:
            scheduler.step()

        train_loss = run_loss / len(train_ds)
        train_acc  = correct / total * 100
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        run_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                out  = model(images)
                loss = criterion(out, labels)
                run_loss += loss.item() * images.size(0)
                _, pred = torch.max(out, 1)
                total   += labels.size(0)
                correct += (pred == labels).sum().item()

        val_loss = run_loss / len(val_ds)
        val_acc  = correct / total * 100
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'   ↓ Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'   ↓ Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%')

        if val_loss < (best_val_loss - CONFIG.min_delta):
            best_val_loss = val_loss
            best_val_acc  = val_acc
            patience_ctr  = 0
            torch.save(model.state_dict(), out_dir / 'best_model.pt')
            print('   ★ Val loss improved — model saved.')
        else:
            patience_ctr += 1
            print(f'   ! Patience: {patience_ctr}/{CONFIG.patience}')

        if patience_ctr >= CONFIG.patience:
            print(f'\nEarly stopping at epoch {epoch+1}.')
            break

    print(f'\nBest val accuracy: {best_val_acc:.2f}%')

    # ── Training curves ───────────────────────────────────────────────
    n = len(train_losses)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(range(1, n+1), train_losses, label='Train')
    ax1.plot(range(1, n+1), val_losses,   label='Val')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()
    ax2.plot(range(1, n+1), train_accs, label='Train')
    ax2.plot(range(1, n+1), val_accs,   label='Val')
    ax2.set_title('Accuracy (%)'); ax2.set_xlabel('Epoch'); ax2.legend()
    plt.suptitle(f'{CONFIG.model_name} Training Curves')
    plt.tight_layout()
    plt.savefig(out_dir / 'training_curves.png', dpi=150)
    plt.close()

    # ── Confusion matrix on val set ───────────────────────────────────
    model.load_state_dict(torch.load(out_dir / 'best_model.pt', map_location=DEVICE))
    model.eval()
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            _, pred = torch.max(model(images.to(DEVICE)), 1)
            all_preds.extend(pred.cpu().numpy())
            all_tgts.extend(labels.numpy())

    cm = confusion_matrix(all_tgts, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'{CONFIG.model_name} Confusion Matrix')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrix.png', dpi=150)
    plt.close()

    report = classification_report(all_tgts, all_preds, target_names=CLASS_NAMES)
    print(report)

    # ── Stats file ────────────────────────────────────────────────────
    with open(out_dir / 'training_stats.txt', 'w') as f:
        f.write(f'Model:           {CONFIG.model_name}\n')
        f.write(f'Best val acc:    {best_val_acc:.4f}%\n')
        f.write(f'Best val loss:   {best_val_loss:.6f}\n')
        f.write(f'Epochs run:      {n}\n')
        f.write(f'LR:              {CONFIG.lr}\n')
        f.write(f'Batch size:      {CONFIG.batch_size}\n')
        f.write(f'Dropout:         {CONFIG.dropout}\n')
        f.write(f'Cosine LR:       {CONFIG.use_cosine_lr}\n')
        f.write(f'Label smoothing: {CONFIG.label_smoothing}\n')
        f.write(f'Class weights:   {CONFIG.use_class_weights}\n')
        f.write(f'\nClassification Report:\n{report}')

    print(f'\nOutputs saved to {out_dir}')


if __name__ == '__main__':
    main()
