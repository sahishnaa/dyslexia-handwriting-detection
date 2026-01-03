# Dyslexia Handwriting Pattern Detection using YOLOv8 and CNN
Hybrid ML model for detecting dyslexia-related handwriting patterns using:
- YOLOv8 for letter-level detection
- ResNet18 CNN for page-level classification
- Explainable AI (Grad-CAM)
- Person-level handwriting aggregation

## Problem statement
Dyslexia often shows itself through letter reversals, overwriting corrections, and inconsistent handwriting. Traditional assessment is manual and subjective. This system provides AI-assisted, explainable approach to analyze handwriting patterns.

## Methodology
1. YOLOv8 detects individual letters and classifies them as:
   - Normal
   - Reversal
   - Corrected
2. Confidence-weighted aggregation is used to determine dominant handwriting type.
3. A CNN (ResNet18 with transfer learning) classifies full handwriting sheets.
4. Grad-CAM visualizations explain CNN predictions.
5. Final decision uses hybrid fusion for robustness.

## Models Used
- YOLOv8 (Ultralytics)
- ResNet18 (PyTorch, Transfer Learning)

## Dataset
Synthetic dyslexia handwriting dataset (Kaggle):
https://www.kaggle.com/datasets/michaelfink0923/synthetic-dyslexia-handwriting-dataset

## Installation
```bash
pip install ultralytics torch torchvision opencv-python matplotlib scikit-learn
```

## Explainability
Grad-CAM heatmaps show which handwriting regions influence CNN predictions.

## Limitations
Dataset is synthetic
Real handwriting causes domain shift

## Future Work
Train on real dyslexic handwriting
Multimodal learning (audio + eye tracking)
Classroom assistive deployment

## Authors
- Sahishna Rajesh
- Anshu Priya
