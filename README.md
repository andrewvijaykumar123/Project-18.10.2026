# Wine Quality Prediction

This project applies machine learning to the UCI Wine dataset for multi-class classification.

## Overview

The enhanced version trains and evaluates a Decision Tree model, compares it with a Random Forest, and highlights the most important features driving the predictions.

## Files

- wine_quality_prediction.py — basic Decision Tree workflow.
- wine_quality_comparison.py — direct comparison between Decision Tree and Random Forest.
- wine_quality_enhanced.py — enhanced solution with dataset summary, feature importance, confusion matrix, and cross-validation.

## Run

```bash
python wine_quality_prediction.py
python wine_quality_comparison.py
python wine_quality_enhanced.py
```

## Result

The enhanced solution shows that the Decision Tree performs strongly on the wine dataset and remains interpretable, while Random Forest gives a slightly stronger test accuracy on this split.
