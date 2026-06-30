# RAIDKL

A comprehensive framework for training, distilling, and deploying 1D-CNN models for device classification and attack detection using Knowledge Distillation techniques.

## Features

- **Teacher-Student Knowledge Distillation** with multiple methods:
  - Standard KD (KL Divergence)
  - Active Learning
  - Feature Matching
  - Gradient Matching
  - Combined methods
  - Coreset selection
  - Generative (VAE-based)
  - **Adaptive Temperature Online Distillation** (new)

- **Multi-output Support** (`device` + `attack` classification)
- **SMOTE-based class balancing** on training data only
- **FLOPs & Parameter counting**
- **Comprehensive SHAP explanations** (summary, bar, beeswarm, waterfall, decision, force, etc.)
- **Fine-tuning** pre-trained models on new datasets
- **Standalone Inference script**
- Resource monitoring (time, memory, CPU)

## Project Structure
IoMT-Distillation/
├── main.py                    # Main training orchestration
├── training.py                # Training loops (teacher, student, KD)
├── models.py                  # 1D-CNN architectures (large/small)
├── data_preprocessing.py      # Data loading, sampling, scaling
├── evaluation.py              # Metrics, SHAP explanations
├── distillation_techniques.py # All KD methods
├── utils.py                   # Utilities (SMOTE, FLOPs, etc.)
├── inference.py               # Standalone inference
├── checkpoints/               # Model weights & scalers
├── results/                   # Training results & plots
└── logs/                      # Profiling logs


## Setup

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow scikit-learn pandas numpy psutil matplotlib seaborn imbalanced-learn joblib h5py shap

```

### Project Setup
```bash

# Clone or navigate to project directory
cd IoMT-Distillation

# Create required directories
mkdir -p checkpoints results logs
```

# Usage
### 1. Training

```bash
python main.py --csv_path "path/to/NIMLABIoT_processed.csv" \
               --output_mode device \
               --distillation_method standard \
               --n_splits 1
```

# Distillation Method Options

Available `--distillation_method` options:

- **standard**
- **active**
- **feature_matching**
- **gradient_matching**
- **combined**
- **coreset**
- **generative**
- **adaptive_temperature** — Online with adaptive temperature

---

# Output Modes

- **device** — Device type classification  
- **traffic** — Attack category classification  
- **multi** — Both device + attack

---

# File Requirements

- **Training CSV**: Must contain feature columns + label column(s)  
- **New Dataset CSV (for fine-tuning/inference)**: Same structure as training data

---

# Outputs

- **Model weights** → `checkpoints/`  
- **Training results** → `results/*.csv`  
- **SHAP plots** → `checkpoints/`  
- **Performance history (adaptive method)** → JSON files
