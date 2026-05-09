# preprocessing.py
# Author: Bernadette Burks
# Created: Sept. 14, 2025

# Library imports
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler

# Dataset loading
data = pd.read_csv(r'C:\improved_disease_dataset.csv')

# Data encoding
encoder = LabelEncoder()
data["disease"] = encoder.fit_transform(data["disease"])

# Symptom index creation
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Symptom index mapping
symptom_index = {col: idx for idx, col in enumerate(X.columns)}

# Resampling to address class imbalance
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Ensure no NaN values and correct shape
X_resampled = X_resampled.fillna(0)
if len(y_resampled.shape) > 1:
    y_resampled = y_resampled.values.ravel()