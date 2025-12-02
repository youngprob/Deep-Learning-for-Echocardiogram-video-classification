import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import argparse
import numpy as np

from dataset_kfold import Config, EchoVideoDataset, collate_fn
from model_v4 import SwinTemporalClassifier 

# This is the oversampling weight from your K-Fold script (which worked)
OVERSAMPLE_WEIGHT = 7.0 

def plot_training_history(history, save_path):
    """Plots and saves the training loss and validation accuracy curves."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(history['train_loss'], label='Training Loss')
        ax1.set_title('Training Loss'); ax1.legend(); ax1.grid(True)
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Validation Accuracy'); ax2.legend(); ax2.grid(True)
        plt.savefig(save_path)
        plt.close(fig) # Close the figure to save memory
    except Exception as e:
        print(f"Error plotting history: {e}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plots and saves the confusion matrix."""
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix'); plt.ylabel('Actual'); plt.xlabel('Predicted')
        plt.savefig(save_path)
        plt.close() # Close the figure to save memory
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train new SOTA Swin-Temporal model with a simple train/val/test split.")
    parser.add_argument('--data_path', type=str, default=Config.SPLIT_DATA_PATH, help='Path to the split data directory')
    parser.add_argument('--output_path', type=str, default=Config.OUTPUT_DIR, help='Path to save models and plots')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transforms
    val_test_transform = T.Compose([
        T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=Config.MEAN, std=Config.STD)
    ])

    train_transform = T.Compose([
        T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=Config.MEAN, std=Config.STD)
    ])

    # --- Create the 3 Datasets ---
    # We use the (K-Fold compatible) dataset.py because it has the
    # 'is_train' flag, which we need for augmentation.
    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    test_dir = os.path.join(args.data_path, "test")

    train_dataset = EchoVideoDataset(
        data_dir=train_dir, 
        num_frames=Config.NUM_FRAMES, 
        transform=train_transform, 
        is_train=True  # <-- Turns on random sampling
    )
    val_dataset = EchoVideoDataset(
        data_dir=val_dir, 
        num_frames=Config.NUM_FRAMES, 
        transform=val_test_transform, 
        is_train=False # <-- Turns off random sampling
    )
    test_dataset = EchoVideoDataset(
        data_dir=test_dir, 
        num_frames=Config.NUM_FRAMES, 
        transform=val_test_transform, 
        is_train=False # <-- Turns off random sampling
    )

    # --- PERFORMANCE FIX 1: The Sampler ---
    labels = [sample[1] for sample in train_dataset.samples]
    class_counts = Counter(labels)
    print(f"Original class distribution: {class_counts}")
    
    class_sample_count = torch.tensor([class_counts.get(i, 0) for i in range(Config.NUM_CLASSES)])
    minority_class_idx = torch.argmin(class_sample_count).item()
    
    sampler_weights = [1.0] * Config.NUM_CLASSES
    sampler_weights[minority_class_idx] = OVERSAMPLE_WEIGHT
    print(f"Sampler weights: {sampler_weights} (Class {minority_class_idx} is oversampled by {OVERSAMPLE_WEIGHT}x)")

    samples_weight = torch.tensor([sampler_weights[t] for t in labels])
    sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, sampler=sampler,
                              num_workers=Config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                             num_workers=Config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    # --- PERFORMANCE FIX 2: The Loss Function ---
    # We use the Sampler, so we do NOT use a weighted loss.
    # This is the fix for the 25.62% accuracy bug.
    criterion = nn.CrossEntropyLoss()
    print("Using standard CrossEntropyLoss (balancing is done by the Sampler).")

    # --- Load the NEW SOTA Hybrid Model ---
    model = SwinTemporalClassifier(
        num_classes=Config.NUM_CLASSES,
        num_frames=Config.NUM_FRAMES,
        temporal_dim=512,
        num_temporal_layers=2, # You can try increasing this later (e.g., 4)
        num_heads=8
    ).to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_accuracy = 0.0
    start_epoch = 1
    history = {'train_loss': [], 'val_accuracy': []}

    best_model_path = os.path.join(args.output_path, "best_model.pth")
    latest_model_path = os.path.join(args.output_path, "latest_model.pth")

    # --- Simple HPC Resume Logic ---
    if os.path.exists(latest_model_path):
        print(f"Found latest checkpoint at {latest_model_path}. Resuming training.")
        checkpoint = torch.load(latest_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_accuracy = checkpoint['best_val_accuracy']
        history = checkpoint['history']
        print(f"Resuming from epoch {start_epoch} with best accuracy {best_val_accuracy:.2f}%")
    elif os.path.exists(best_model_path):
        print(f"Found best checkpoint at {best_model_path}, but no latest. Starting from best model.")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)

    print("Starting training...")
    for epoch in range(start_epoch, Config.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [Train]")
        for videos, labels in train_pbar:
            if videos.nelement() == 0:
                continue
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(videos)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{running_loss / (train_pbar.n + 1):.4f}"})
        
        history['train_loss'].append(running_loss / len(train_loader))
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [Val]")
        with torch.no_grad():
            for videos, labels in val_pbar:
                if videos.nelement() == 0:
                    continue
                videos, labels = videos.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(videos)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_pbar.set_postfix({'accuracy': f"{(100 * correct / total):.2f}%"})
        
        epoch_accuracy = 100 * correct / total
        history['val_accuracy'].append(epoch_accuracy)
        print(f"\nEpoch {epoch} Summary - Validation Accuracy: {epoch_accuracy:.2f}%\n")

        # --- Save Checkpoint Every Epoch ---
        current_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_accuracy': best_val_accuracy,
            'history': history
        }
        torch.save(current_checkpoint, latest_model_path)
        
        if epoch_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_accuracy
            current_checkpoint['best_val_accuracy'] = best_val_accuracy
            print(f"New best model found! Saving checkpoint to {best_model_path}")
            torch.save(current_checkpoint, best_model_path)

    print("Finished Training.")
    plot_training_history(history, args.output_path)

    print("\nStarting final evaluation on the test set...")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for videos, labels in tqdm(test_loader, desc="[Test]"):
                if videos.nelement() == 0:
                    continue
                videos, labels = videos.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(videos)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
        plot_confusion_matrix(all_preds, all_labels, class_names=test_dataset.classes, save_path=os.path.join(args.output_path, 'final_test_confusion_matrix.png'))
    else:
        print("No best model was saved. Skipping final evaluation.")

if __name__ == '__main__':
    main()
