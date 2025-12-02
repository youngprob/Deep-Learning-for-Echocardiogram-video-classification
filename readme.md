# **Deep Learning for Echocardiogram Video Classification**

## ** Project Overview**

This project implements a **State-of-the-Art (SoTA) Hybrid Video Transformer** to classify echocardiogram videos as **Normal** or **Abnormal** based on Ejection Fraction (EF).

Moving beyond standard global transformers, this project introduces a **Hybrid Local-Global Architecture** that combines **Multi-Head Self Attention (MHSA)** with **Depth-Wise Convolutions (DWConv)**. This dual-path design captures both the global cardiac rhythm and fine-grained, transient wall motion anomalies, significantly improving clinical sensitivity.

### ** Key Features**

* **Hybrid Architecture:** Custom HybridEncoderLayer combining Swin Transformer (Spatial) with parallel Attention/Convolution paths (Temporal).  
* **HPC-Native Engineering:** Robust **Checkpoint & Resume** system designed for High-Performance Computing clusters with strict runtime limits (e.g., 12-hour windows).  
* **Imbalance Handling:** Solved critical data imbalance (3:1 ratio) using a tuned **WeightedRandomSampler** strategy, boosting minority class recall from 26% to 63%.  
* **Rigorous Validation:** Implements **Stratified K-Fold Cross-Validation** (K=7) to ensure statistical reliability.

## **🛠️ Tech Stack**

* **Language:** Python 3.9+  
* **Frameworks:** PyTorch, Torchvision, TIMM  
* **Data Processing:** OpenCV, NumPy, Pandas, Pillow  
* **Evaluation:** Scikit-Learn (Classification Report, Confusion Matrix)

## ** Project Structure**

.  
├── data/                   \# Dataset directory (created by pre\_data.py)  
├── dataset.py              \# Custom "Smart" Dataset with Augmentation & K-Fold logic  
├── model.py                \# The SOTA Hybrid Architecture (Swin \+ DWConv)  
├── pre\_data.py             \# Script to download, extract, and sort EchoNet-Dynamic data  
├── train\_fast\_test.py      \# Script for a single Train/Val/Test run (Proof of Concept)  
├── train\_kfold.py          \# HPC-Ready script for full 7-Fold Cross Validation  
├── requirements.txt        \# Python dependencies  
└── README.md               \# This file

## **⚡ Quick Start**

### **1\. Setup Environment**

pip install \-r requirements.txt

### **2\. Prepare Data**

Run the preprocessing script to download the **EchoNet-Dynamic** dataset and organize it into a stratified split structure.

python pre\_data.py

### **3\. Run a "Fast Test"**

To verify the architecture and environment without running a 40+ hour experiment, use the fast test script. This runs a single split with the new Hybrid model.

python train\_fast\_test.py \--output\_path ./output\_fast

### **4\. Run Full K-Fold Validation (HPC Mode)**

To run the rigorous 7-Fold validation. This script includes **Auto-Resume logic**. If the job is killed (e.g., 12-hour limit), simply re-run the command, and it will pick up exactly where it left off (skipping completed folds and resuming epochs).

python train\_kfold.py \--output\_path ./output\_kfold

## ** Model Architecture details**

### **The "Recall Bottleneck" Problem**

Standard Video Transformers use **Global Average Pooling** to condense frames into a prediction. We found this "drowned out" short, transient anomalies (like a flutter lasting 3 frames), resulting in a low **26% Recall** for abnormal hearts.

### **The Hybrid Solution**

We replaced the standard Temporal Encoder with a custom **Hybrid Encoder** inspired by HLG-ViT logic.

1. **Global Path (MHSA):** Looks at the entire video sequence to understand the overall heart cycle.  
2. **Local Path (DWConv):** Uses **Depth-Wise Convolutions** as a sliding window to detect fast, frame-to-frame local motion changes.  
3. **Feature Mixing:** The outputs are summed, preserving both global rhythm and local details.

**Result:** Abnormal Recall improved from **26%** $\\rightarrow$ **63%**.

## ** Results Summary**

| Model Phase | Approach | Accuracy | Abnormal Recall |
| :---- | :---- | :---- | :---- |
| **Phase 1** | CNN \+ LSTM | 79% | 46% |
| **Phase 2** | Swin \+ Global Transformer | 78% | 26% |
| **Phase 3** | **Swin \+ Hybrid (Ours)** | **74%** | **63%** |

*Note: Phase 3 trades a small amount of global accuracy for a massive **2.5x improvement** in sensitivity to disease, making it more clinically valuable.*

## **📝 Acknowledgments**

* Dataset provided by **Stanford (EchoNet-Dynamic)**.  
* Pre-trained weights via **TIMM (PyTorch Image Models)**.
