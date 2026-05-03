# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Hetairos (H&E Trained AI for Recognition of Oncology Slides) is a multiple instance learning (MIL) model for predicting methylation-based CNS tumour subtypes from H&E slide images. The project uses PyTorch Lightning for training.

## Common Commands

### Environment Setup
```bash
conda create -n hetairos python=3.10
conda activate hetairos
pip install -r requirements.txt
```

### Preprocessing: Tiling
```bash
cd preprocessing
python -m tiling.main_create_tiles --source_dir <WSIs_path> --source_list <slide_list.txt> --save_dir <tiles_path> --patch_size 256 --step_size 256 --mag 20
```

### Preprocessing: Feature Extraction
```bash
python preprocessing/feature_extraction/get_features.py --split <tile_path_list.txt> --feature_dir <features_path> --batchsize 384
```

### Model Training
```bash
python -m aggregator_train_val.model_run --dataset <features_path> --label <label.csv> --label_map <label_mapping.yaml> --split <split.yaml> --mode train --data_aug --soft_labels --exp_name <exp_name>
```

### Model Testing
```bash
python -m aggregator_train_val.model_run --dataset <features_path> --label <label.csv> --label_map <label_mapping.yaml> --split <split.yaml> --mode test --exp_name <exp_name>
```

### End-to-End Pipeline
```bash
python pipeline.py --tiling --slide_dir <WSIs_path> --tile_savedir <tiles_path> --feature_extraction --feature_dir <features_path> --model_run --dataset <dataset_path> --label <label.csv> --label_map <label_mapping.yaml> --split <split.yaml> --mode train
```

## Architecture

### Pipeline Flow
1. **Tiling** (`preprocessing/tiling/`) - Segments WSIs into 256x256 patches at 20x magnification
2. **Feature Extraction** (`preprocessing/feature_extraction/get_features.py`) - Extracts tile features using Prov-GigaPath foundation model, saves as .h5 then converts to .pt
3. **Model Training/Inference** (`aggregator_train_val/`) - ATransMIL model with contrastive learning

### Core Model: ATransMIL (`aggregator_train_val/model_module.py`)
- Divides slide feature matrix into `group_num` (default 3) sub-bags
- Uses Nyström attention-based transformer layers with PPEG positional encoding
- Integrates patient age (positional encoding) and tumor location (7 anatomical regions) with tile embeddings
- Contrastive loss with EMA-updated class templates guides hidden space clustering
- Outputs both sub-bag and slide-level predictions

### Data Format
- **Labels CSV**: columns `slide | family | prob_vector (optional) | age | location`
- **Split YAML**: `{"train": [slide_ids], "test": [slide_ids]}`
- **Label mapping YAML**: `{tumor_name: integer_id}`
- Location values: Extracranial, Infratentorial, Intra- or Peri-Ventricular, Intra- or Supra-Sellar, Pineal, Spinal, Supratentorial

### Key Configuration (`aggregator_train_val/config.yaml`)
- `n_classes`: 186 (output dimension, can exceed actual classes)
- `embedding_size`: 1536 (matches Prov-GigaPath output)
- `group_num`: 3 (sub-bag splits)
- `cl_w`: 20 (contrastive loss weight)
- `age_loc_drop_prob`: 0.7 (dropout for age/location during augmentation)

### External Dependencies
Foundation model for feature extraction must be configured in `preprocessing/feature_extraction/get_features.py:75` - default uses `hf_hub:prov-gigapath/prov-gigapath`.
