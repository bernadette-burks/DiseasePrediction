# prediction.py - “use the models”
# Author: Bernadette Burks
# Created: Sept. 14, 2025

# symptom input is processed
# prediction happens
# final output is returned

# prediction.py

import pandas as pd
from statistics import mode

from src.training import rf_model, nb_model, svm_model
from src.preprocessing import symptom_index, encoder, X_resampled

symptoms = X_resampled.columns


def predict_disease(input_symptoms):
    input_symptoms = input_symptoms.split(",")

    input_data = [0] * len(symptom_index)

    for symptom in input_symptoms:
        symptom = symptom.strip()
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1

    input_df = pd.DataFrame([input_data], columns=symptoms)

    rf_pred = encoder.classes_[rf_model.predict(input_df)[0]]
    nb_pred = encoder.classes_[nb_model.predict(input_df)[0]]
    svm_pred = encoder.classes_[svm_model.predict(input_df)[0]]

    final_pred = mode([rf_pred, nb_pred, svm_pred])

    return {
        "Random Forest": rf_pred,
        "Naive Bayes": nb_pred,
        "SVM": svm_pred,
        "Final Prediction": final_pred
    }