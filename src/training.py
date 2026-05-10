# training.py - “teach the models”
# Author: Bernadette Burks
# Created: Sept. 14, 2025

# define models
# fit/train models
# produce trained objects

from src.preprocess import X_resampled, y_resampled
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC

rf_model = RandomForestClassifier(random_state=42)
nb_model = GaussianNB()
svm_model = SVC()

rf_model.fit(X_resampled, y_resampled)
nb_model.fit(X_resampled, y_resampled)
svm_model.fit(X_resampled, y_resampled)

models = {
    "Random Forest": rf_model,
    "Naive Bayes": nb_model,
    "SVM": svm_model
}