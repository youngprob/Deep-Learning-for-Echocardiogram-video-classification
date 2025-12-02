import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import argparse
import numpy as np
import json # --- Added for master checkpoint ---

# import our helper scripts for the dataset and model
from dataset_kfold import Config, EchoVideoDataset, collate_fn
from model_v3 import SwinTemporalClassifier  # Use the new model

# --- Set to 7 to match your log file ---
K_FOLDS = 7

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


def load_all_samples(data_dir):
    """Loads all file paths and labels from a directory."""
    samples = []
    classes = ['normal_hearts', 'abnormal_hearts']
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    for target_class in classes:
        class_dir = os.path.join(data_dir, target_class)
        if not os.path.isdir(class_dir): continue
        target_idx = class_to_idx[target_class]
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith('.avi'):
                item = (os.path.join(class_dir, fname), target_idx)
                samples.append(item)
    return samples

def main():
    parser = argparse.ArgumentParser(description="Train Swin-Temporal model with K-Fold Cross-Validation.")
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

    # Load all data into a combined pool
    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    
    all_samples = load_all_samples(train_dir) + load_all_samples(val_dir)
    all_labels = [label for (path, label) in all_samples]
    print(f"Combined Train+Val dataset size: {len(all_samples)} videos")
    
    # Load held-out test set
    test_dir = os.path.join(args.data_path, "test")
    test_dataset = EchoVideoDataset(
        data_dir=test_dir, 
        num_frames=Config.NUM_FRAMES, 
        transform=val_test_transform, 
        is_train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    print(f"Held-out Test dataset size: {len(test_dataset)} videos")

    # K-Fold setup
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    X_dummy = np.zeros(len(all_labels)) 
    
    # --- RESUME LOGIC 1: Load master K-Fold state ---
    kfold_state_path = os.path.join(args.output_path, "kfold_state.json")
    start_fold = 1
    overall_best_val_accuracy = 0.0
    best_model_path_overall = ""
    fold_results = []

    if os.path.exists(kfold_state_path):
        print(f"Found K-Fold state at {kfold_state_path}. Resuming...")
        with open(kfold_state_path, 'r') as f:
            kfold_state = json.load(f)
        start_fold = kfold_state.get('start_fold', 1)
        overall_best_val_accuracy = kfold_state.get('overall_best_val_accuracy', 0.0)
        best_model_path_overall = kfold_state.get('best_model_path_overall', "")
        fold_results = kfold_state.get('fold_results', [])
        print(f"Resuming from Fold {start_fold}. Overall best accuracy so far: {overall_best_val_accuracy:.2f}%")
    # --- END RESUME LOGIC 1 ---

    print(f"\nStarting {K_FOLDS}-Fold Cross-Validation...")

    # Use range(start_fold - 1, K_FOLDS) to skip completed folds
    for fold in range(start_fold - 1, K_FOLDS):
        current_fold_num = fold + 1
        print(f"\n--- FOLD {current_fold_num}/{K_FOLDS} ---")
        
        # Get indices for this fold
        # We must regenerate the splits to get the right indices
        train_indices, val_indices = list(skf.split(X_dummy, all_labels))[fold]

        train_samples_fold = [all_samples[i] for i in train_indices]
        train_labels_fold = [all_labels[i] for i in train_indices]
        val_samples_fold = [all_samples[i] for i in val_indices]

        train_dataset = EchoVideoDataset(
            samples=train_samples_fold,
            num_frames=Config.NUM_FRAMES,
            transform=train_transform,
            is_train=True
        )
        val_dataset = EchoVideoDataset(
            samples=val_samples_fold,
            num_frames=Config.NUM_FRAMES,
            transform=val_test_transform,
            is_train=False
        )

        class_counts = Counter(train_labels_fold)
        print(f"Fold {current_fold_num} Train distribution: {class_counts}")
        class_sample_count = torch.tensor([class_counts.get(i, 0) for i in range(Config.NUM_CLASSES)])

        # Sampler weights based on K_FOLDS (e.g., 7x oversampling from your log)
        minority_class_idx = torch.argmin(class_sample_count).item()
        sampler_weights = [1.0] * Config.NUM_CLASSES
        sampler_weights[minority_class_idx] = K_FOLDS # Set to 7 (or 10 if you change K_FOLDS)
        
        print(f"Sampler weights: {sampler_weights} (Class {minority_class_idx} is oversampled by {K_FOLDS}x)")
        samples_weight = torch.tensor([sampler_weights[label] for label in train_labels_fold])
        train_sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

        # --- PERFORMANCE FIX: We use the Sampler, so we do NOT use a weighted loss ---
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss (balancing is done by the Sampler).")
        # --- END PERFORMANCE FIX ---

        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, sampler=train_sampler, num_workers=Config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

        # Initialize model for each fold
        model = SwinTemporalClassifier(
            num_classes=Config.NUM_CLASSES,
            num_frames=Config.NUM_FRAMES,
            temporal_dim=512,
            num_temporal_layers=2,
            num_heads=8
        ).to(device)
        
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        # --- RESUME LOGIC 2: Load this fold's latest epoch state ---
        start_epoch = 1
        best_val_accuracy_fold = 0.0
        history = {'train_loss': [], 'val_accuracy': []}
        
        best_model_path_fold = os.path.join(args.output_path, f"best_model_fold_{current_fold_num}.pth")
        latest_model_path_fold = os.path.join(args.output_path, f"latest_model_fold_{current_fold_num}.pth")

        if os.path.exists(latest_model_path_fold):
            print(f"Found latest checkpoint for Fold {current_fold_num} at {latest_model_path_fold}. Resuming...")
            checkpoint = torch.load(latest_model_path_fold, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_accuracy_fold = checkpoint['best_val_accuracy_fold']
            history = checkpoint['history']
            print(f"Resuming Fold {current_fold_num} from epoch {start_epoch} with best accuracy {best_val_accuracy_fold:.2f}%")
        # --- END RESUME LOGIC 2 ---

        # --- Epoch Training Loop (Inner Loop) ---
        for epoch in range(start_epoch, Config.EPOCHS + 1):
            model.train()
            running_loss = 0.0
            train_pbar = tqdm(train_loader, desc=f"Fold {current_fold_num} Epoch {epoch}/{Config.EPOCHS} [Train]")
            for videos, labels in train_pbar:
                if videos.nelement() == 0: continue
                videos, labels = videos.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(videos)
                    loss = criterion(outputs, labels) # Loss is no longer weighted
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
                train_pbar.set_postfix({'loss': f"{running_loss / (train_pbar.n + 1):.4f}"})
            
            history['train_loss'].append(running_loss / len(train_loader))
            scheduler.step()

            model.eval()
            correct, total = 0, 0
            val_pbar = tqdm(val_loader, desc=f"Fold {current_fold_num} Epoch {epoch}/{Config.EPOCHS} [Val]")
            with torch.no_grad():
                for videos, labels in val_pbar:
                    if videos.nelement() == 0: continue
                    videos, labels = videos.to(device), labels.to(device)
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        outputs = model(videos)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    val_pbar.set_postfix({'accuracy': f"{(100 * correct / total):.2f}%"})
            
            epoch_accuracy = 100 * correct / total
            history['val_accuracy'].append(epoch_accuracy)
            
            # --- SAVE LOGIC 1: Save latest state for this fold EVERY epoch ---
            current_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_accuracy_fold': best_val_accuracy_fold,
                'history': history
            }
            torch.save(current_checkpoint, latest_model_path_fold)
            # --- END SAVE LOGIC 1 ---

            if epoch_accuracy > best_val_accuracy_fold:
                best_val_accuracy_fold = epoch_accuracy
                print(f"  New best val accuracy for Fold {current_fold_num}: {best_val_accuracy_fold:.2f}%. Saving best model for fold...")
                # Save just the model state for the "best" model
                torch.save({'model_state_dict': model.state_dict()}, best_model_path_fold)

                if best_val_accuracy_fold > overall_best_val_accuracy:
                    overall_best_val_accuracy = best_val_accuracy_fold
                    best_model_path_overall = best_model_path_fold
                    print(f"    This is the new *OVERALL* best model so far!")
        
        # --- END OF EPOCH LOOP ---
        
        print(f"\nFold {current_fold_num} Summary - Best Validation Accuracy: {best_val_accuracy_fold:.2f}%\n")
        # Only add to fold_results if it wasn't there already (from resume)
        if len(fold_results) < current_fold_num:
            fold_results.append(best_val_accuracy_fold)
            
        plot_training_history(history, os.path.join(args.output_path, f'training_history_fold_{current_fold_num}.png'))
        
        # --- SAVE LOGIC 2: This fold is DONE. Update master checkpoint. ---
        kfold_state = {
            'start_fold': current_fold_num + 1, # Next time, start on the *next* fold
            'overall_best_val_accuracy': overall_best_val_accuracy,
            'best_model_path_overall': best_model_path_overall,
            'fold_results': fold_results
        }
        with open(kfold_state_path, 'w') as f:
            json.dump(kfold_state, f, indent=4)
        print(f"Completed Fold {current_fold_num}. Updated master checkpoint.")
        
        # Clean up the latest model checkpoint for this fold, as it's no longer needed
        if os.path.exists(latest_model_path_fold):
            os.remove(latest_model_path_fold)
        # --- END SAVE LOGIC 2 ---
    
    # --- END OF K-FOLD LOOP ---
    
    print("\n--- K-Fold Cross-Validation Summary ---")
    for i, acc in enumerate(fold_results):
        print(f"Fold {i+1} Best Validation Accuracy: {acc:.2f}%")
    
    mean_accuracy = np.mean(fold_results)
    std_accuracy = np.std(fold_results)
    print(f"\nAverage Validation Accuracy: {mean_accuracy:.2f}% +/- {std_accuracy:.2f}%")
    print(f"The best model overall was from {best_model_path_overall} with {overall_best_val_accuracy:.2f}% val accuracy.")

    # --- FINAL EVALUATION ON HELD-OUT TEST SET ---
    print("\nStarting final evaluation on the *held-out test set* using the best model...")
    if os.path.exists(best_model_path_overall):
        checkpoint = torch.load(best_model_path_overall, map_location=device)
        model = SwinTemporalClassifier(
            num_classes=Config.NUM_CLASSES,
            num_frames=Config.NUM_FRAMES,
            temporal_dim=512,
            num_temporal_layers=2,
            num_heads=8
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        all_preds, all_labels_test = [], []
        with torch.no_grad():
            for videos, labels in tqdm(test_loader, desc="[Final Test]"):
                if videos.nelement() == 0: continue
                videos, labels = videos.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(videos)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels_test.extend(labels.cpu().numpy())
        
        print("\n--- Final Test Set Classification Report ---")
        print(classification_report(all_labels_test, all_preds, target_names=test_dataset.classes))
        plot_confusion_matrix(all_labels_test, all_preds, class_names=test_dataset.classes, save_path=os.path.join(args.output_path, 'final_test_confusion_matrix.png'))
    else:
        print("No best model was saved. Skipping final evaluation.")
        
    # --- Clean up master checkpoint at the very end ---
    if os.path.exists(kfold_state_path):
        os.remove(kfold_state_path)

if __name__ == '__main__':
    main()

