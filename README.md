# 🎧 CLAP Fine-Tuning for Speech Emotion Recognition

This repository provides code for fine-tuning the [CLAP](https://github.com/LAION-AI/CLAP) model on the IEMOCAP dataset for Speech Emotion Recognition (SER).  
We adopt a linear probing setup where CLAP audio embeddings are fed into a lightweight classifier.

## 🛠️ Environment Setup

We recommend using **Anaconda** for environment management:
## 🛠️ Environment Setup
```bash
conda env create -f train_linear/environment.yml
conda activate clap-ser
```
## 📁 Project Structure
```CLAP_fintuning/
├── assets/                  # Pre-trained CLAP weights
├── class_labels/            # Label mappings
├── experiment_scripts/      # Additional experiments
├── LICENSE
├── MANIFEST.in
├── pyproject.toml
├── README.md
├── requirements.txt
└── src/                     # CLAP modules and utilities

train_linear/
├── dassl/                   # DASSL training engine
├── config/                  # Training config files
├── environment.yml
├── scripts/
│   └── run_clap_iemocap_ps.sh   # One-click training script
├── train.py                 # Main training script
└── README.md
```
## Run CLAP Finetune
```
bash CLAP_finetune/experiment_script/train-pann-roberta.sh
```

## Run Linear Training
```
bash scripts/run_clap_iemocap_ps.sh
```
