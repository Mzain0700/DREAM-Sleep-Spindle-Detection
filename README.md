# DREAM Sleep Spindle Detection

Machine learning pipeline for detecting sleep spindles from EEG recordings in the DREAMS Sleep Spindles Database.


## Problem Statement

Sleep spindles are short bursts of oscillatory brain activity that appear during non-REM sleep. They are clinically important because they are associated with sleep quality, brain maturation, memory consolidation, and several neurological conditions.

Manual spindle annotation is slow, subjective, and difficult to scale across long EEG recordings. This project addresses that problem by building an automated spindle detection pipeline that can classify short EEG windows as spindle or non-spindle events.

## Solution Overview

This notebook implements a classical machine learning pipeline for spindle detection on the DREAMS dataset. The workflow is:

1. Load EEG excerpts and spindle annotations.
2. Split the signal into overlapping windows.
3. Label each window using annotation overlap.
4. Extract handcrafted time-domain and frequency-domain features.
5. Balance the training data with SMOTE-ENN.
6. Train and compare multiple classifiers.
7. Evaluate with leave-one-excerpt-out validation.
8. Visualize model behavior and feature discrimination.
9. Save the best model and scaler for reuse.

## What Is Implemented

### Data Loading

- Loads EEG excerpts from the DREAMS Sleep Spindles Database.
- Handles missing files and recording inconsistencies.
- Summarizes each excerpt by duration, spindle count, and spindle density.

### Window Generation

- Uses a sliding window approach with:
  - window size: 0.5 seconds
  - stride: 0.4 seconds
- Labels a window as spindle if annotation overlap exceeds the configured threshold.
- Reduces class imbalance by keeping a controlled spindle-to-non-spindle ratio.

### Feature Extraction

Each EEG window is transformed into a compact feature vector containing:

- RMS
- Peak-to-peak amplitude
- Zero-crossing rate
- Teager energy
- Spindle-band power (11 to 16 Hz)
- Alpha-band power (7 to 10 Hz)
- Spectral entropy
- Hjorth mobility
- Hjorth complexity

### Machine Learning Models

The notebook compares four classifiers:

- Random Forest
- Gradient Boosting
- Support Vector Machine
- Logistic Regression

Hyperparameter tuning is performed with randomized search where applicable.

### Class Imbalance Handling

- Applies SMOTE-ENN during training folds to improve class balance.
- Uses class weighting for models that support it.

### Validation Strategy

- Uses Leave-One-Group-Out cross-validation.
- Treats each excerpt as a group so evaluation stays excerpt-independent.
- Reports Accuracy, F1 score, and ROC-AUC for each fold and model.

### Final Model Saving

- Trains the best-performing model on the full balanced dataset.
- Saves the trained model with joblib.
- Saves the fitted StandardScaler for future inference.

## Visualizations

The notebook includes several visual analyses:

- EEG excerpts with spindle annotations
- Windowing process illustration
- Feature distributions by class
- Feature separation power comparison
- Feature correlation heatmap
- ROC curves for all models
- Precision-Recall curves for all models
- Confusion matrix for the best model
- Model comparison bar charts
- Per-excerpt performance analysis

These visualizations help inspect signal structure, feature usefulness, class separability, and model performance.

## Dataset Used

This project uses the DREAMS Sleep Spindles Database.

Dataset details:

- 8 excerpts
- 30 minutes each
- central EEG channel
- expert spindle annotations
- derived from whole-night PSG recordings

Download link:

- [DREAMS Sleep Spindles Database on Zenodo](https://zenodo.org/record/2650142)

Original dataset description:

- [The DREAMS Databases and Assessment algorithm](https://github.com/numediart/The-DREAMS-Databases-and-Assessment-algorithm/)

## Project Structure

- `DREAM_ML_Models.ipynb` - main notebook containing the full pipeline
- `README.md` - project overview and setup instructions

## How to Run

### In Google Colab

1. Upload or open `DREAM_ML_Models.ipynb` in Colab.
2. Mount Google Drive when prompted.
3. Place the dataset in the expected folder path used by the notebook:
   - `/content/drive/MyDrive/DREAM_Database_Spindles`
4. Run the notebook cells from top to bottom.

### Local Environment

If running locally, update the DATA_DIR path in the notebook to match your machine and ensure the DREAMS EDF files and annotations are available.

## Dependencies

The notebook uses the following Python libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scipy
- scikit-learn
- imbalanced-learn
- mne
- tqdm
- joblib

## Notes

- The notebook is designed for excerpt-level generalization rather than random window splitting.
- Feature engineering is intentionally lightweight and interpretable.
- The pipeline is suitable for benchmarking classical ML approaches before moving to deep learning models.

## Citation and Acknowledgment

This project is based on the DREAMS Sleep Spindles Database and the original DREAMS database release by the Numediart group.

If you use this repository, please also cite the DREAMS database source and any related publications used in your research.
