## Team 3TD
Performed by the following members
- Nguyễn Hoàng Minh (Leader)
- Huỳnh Vĩ Trung
- Lê Quý Hoàng Tùng

# Dataset
This Dataset/CMOSE, processed_data is NOT pushed to GitHub due to its large size.

## How to download data
- Google Drive link: ...
- Kaggle: ...
- After downloading Dataset.zip, extract it to the `Dataset/` folder.
- After downloading processed_data, extract it to the 3TD_KLTN_AI folder.

## Structure
Dataset/
 ├─ CMOSE/
models/
processed_data/

......

## Problem
Binary classification (Engage / Disengage) from time-series behavioral features.

## Model
- CNN 1D for local temporal feature extraction
- BiLSTM for long-range bidirectional dependencies

## Pipeline
- Build sliding window sequences
- Normalize features
- Train CNN-BiLSTM model
- Evaluate with precision / recall / F1-score

## Results
Current results:
 Accuracy: 85.3% Engage recall: 94.4%

## Requirements
Python 3.10+
Install dependencies: pip install -r requirements.txt 
Note: TensorFlow manages its internal dependencies automatically.