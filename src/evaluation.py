# evaluation.py - “test the models”
# Author: Bernadette Burks
# Created: Sept. 14, 2025

# run cross-validation
# compute metrics
# generate confusion matrices
# compare model performance

# evaluation.py

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from statistics import mode
import matplotlib.pyplot as plt
import seaborn as sns
from src.training import models, rf_model, dt_model, svm_model
from src.preprocess import X_resampled, y_resampled

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation
for name, model in models.items():
    scores = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring="accuracy")
    print(name)
    print(scores)
    print(scores.mean())


def plot_confusion(model, X, y, title):
    preds = model.predict(X)
    cm = confusion_matrix(y, preds)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(title)
    plt.show()

    print(title, accuracy_score(y, preds))
    return preds


svm_preds = plot_confusion(svm_model, X_resampled, y_resampled, "SVM")
dt_preds = plot_confusion(dt_model, X_resampled, y_resampled, "Decision Tree")
rf_preds = plot_confusion(rf_model, X_resampled, y_resampled, "RF")

final_preds = [mode([i, j, k]) for i, j, k in zip(svm_preds, dt_preds, rf_preds)]

cm = confusion_matrix(y_resampled, final_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Ensemble Model")
plt.show()