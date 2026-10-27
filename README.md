# Indian Traffic Sign Recognition Using ResNet50

## Overview

This project evaluates ResNet50 for classifying Indian traffic signs using three training strategies: training from scratch, transfer learning with a frozen backbone, and fine-tuning a pretrained model.

The objective is to analyse how ImageNet pretraining and fine-tuning affect classification performance on the Indian Traffic Sign Dataset.

## Dataset

- **Dataset:** Indian Traffic Sign Dataset
- **Number of classes:** 58
- **Images after cleaning:** 12,925
- **Preprocessing:** Duplicate removal, resizing to 224 × 224, and ImageNet normalization
- **Data split:** Stratified training, validation, and test sets

The dataset was cleaned to remove exact duplicates and files involved in label conflicts. Class-weighted cross-entropy was used to address class imbalance.

## Models and Experiments

Three ResNet50 configurations were evaluated:

1. **Training from scratch:** ResNet50 initialized without pretrained ImageNet weights.
2. **Frozen backbone:** ImageNet-pretrained ResNet50 with its backbone frozen while training a new classification head.
3. **Fine-tuning:** ImageNet-pretrained ResNet50 with `layer4` unfrozen and adapted to the traffic-sign dataset.

All three approaches were evaluated using the same held-out test set.

## Results

| Method | Test Accuracy | Macro F1 | Weighted F1 | Top-5 Accuracy |
|---|---:|---:|---:|---:|
| ResNet50 from scratch | 93.14% | 90.35% | 92.99% | 99.59% |
| Pretrained, frozen backbone | 79.53% | 75.20% | 79.17% | 96.03% |
| Pretrained, fine-tuned | 92.32% | 89.24% | 92.24% | 99.54% |

## Key Findings

- Training ResNet50 from scratch achieved the highest test accuracy in this run.
- The frozen-backbone model performed substantially worse than the other two configurations.
- Fine-tuning improved test accuracy by 12.79 percentage points compared with the frozen-backbone result.
- Fine-tuning recovered most of the performance of the scratch-trained model, indicating that adapting pretrained features is important for this dataset.

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision and recall
- Macro F1-score
- Weighted F1-score
- Top-5 accuracy
- Confusion matrix
- Per-class classification metrics

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Google Colab

## Limitations

- Results are from a single experimental run and random seed.
- The small difference between the scratch-trained and fine-tuned models may not be statistically significant.
- Fine-tuning followed the frozen-backbone training stage, so the experiment involved additional training exposure.
- Further testing under varied lighting, blur, occlusion, and other real-world conditions is needed.

## Reproducibility

The Jupyter notebook contains the main data preparation, model training, evaluation, and inference workflow. Access to the dataset and a compatible Python environment are required to reproduce the experiments.
