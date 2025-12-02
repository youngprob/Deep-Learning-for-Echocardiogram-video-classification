import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

class Config:
    BASE_PROJECT_PATH = "/export/home/abhishek/V_BTP/" 
    BASE_PROJECT_PATH1 = "/export/home/abhishek/V_BTP/v4/" 
    DATA_PATH = os.path.join(BASE_PROJECT_PATH, "data")
    SPLIT_DATA_PATH = os.path.join(DATA_PATH, "split")
    OUTPUT_DIR = os.path.join(BASE_PROJECT_PATH1, "output")
    
    TRAIN_DIR = os.path.join(SPLIT_DATA_PATH, "train")
    VAL_DIR = os.path.join(SPLIT_DATA_PATH, "val")
    TEST_DIR = os.path.join(SPLIT_DATA_PATH, "test")
    
    NUM_CLASSES = 2
    EPOCHS = 20
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    
    # --- NUM_FRAMES is now 60 ---
    NUM_FRAMES = 60
    
    NUM_WORKERS = 2
    LSTM_HIDDEN_SIZE = 512
    LSTM_LAYERS = 1

    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

def extract_frames(video_path, num_frames, transform, is_train=False):
    """
    Extracts and transforms a fixed number of frames from a video file.
    
    Args:
        is_train (bool): If True, randomly samples a contiguous 60-frame clip (augmentation).
                         If False, uniformly samples 60 frames from the video.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1: return None
        
        indices = []
        
        if total_frames < num_frames:
            # Loop frames if video is shorter than num_frames
            indices = np.arange(total_frames)
            indices = np.tile(indices, (num_frames // total_frames) + 1)[:num_frames]
        elif is_train:
            # For training, randomly sample a starting point (data augmentation)
            start_frame = np.random.randint(0, total_frames - num_frames + 1)
            indices = np.arange(start_frame, start_frame + num_frames)
        else:
            # For validation/testing, uniformly sample (no augmentation)
            indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            processed_frame = transform(pil_img)
            frames.append(processed_frame)
            
        cap.release()
        
        if len(frames) != num_frames:
            return None
            
        return torch.stack(frames)
        
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None

class EchoVideoDataset(Dataset):
    """
    MODIFIED: This dataset is compatible with K-Fold.
    It can be initialized with a data_dir (for the test set)
    OR a list of samples (for the K-Fold train/val sets).
    """
    # --- THIS IS THE KEY CHANGE ---
    def __init__(self, num_frames, transform, is_train=False, data_dir=None, samples=None):
        self.num_frames = num_frames
        self.transform = transform
        self.is_train = is_train  # Flag for sampling logic
        self.classes = ['normal_hearts', 'abnormal_hearts']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        if samples:
            # This is used by the K-Fold script for train/val folds
            self.samples = samples
        elif data_dir:
            # This is used by the K-Fold script for the test set
            self.data_dir = data_dir # Set self.data_dir
            self.samples = self.load_samples_from_dir(self.data_dir)
        else:
            raise ValueError("Must provide either data_dir or samples")
    # --- END OF KEY CHANGE ---

    def load_samples_from_dir(self, data_dir):
        """Loads all video paths and labels from a directory."""
        samples = []
        for target_class in self.classes:
            class_dir = os.path.join(data_dir, target_class)
            if not os.path.isdir(class_dir): continue
            target_idx = self.class_to_idx[target_class]
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith('.avi'):
                    item = (os.path.join(class_dir, fname), target_idx)
                    samples.append(item)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        # --- Pass is_train flag to extract_frames ---
        frames = extract_frames(video_path, self.num_frames, self.transform, self.is_train)
        
        if frames is None:
            return torch.zeros((self.num_frames, 3, Config.IMG_SIZE, Config.IMG_SIZE)), -1
        return frames, label

def collate_fn(batch):
    """Custom collate function to filter out failed samples."""
    batch = list(filter(lambda x: x[1] != -1, batch))
    if len(batch) == 0: return torch.Tensor(), torch.Tensor()
    videos, labels = zip(*batch)
    videos = torch.stack(videos, dim=0)
    labels = torch.LongTensor(labels)
    return videos, labels
