# main.py - “test the models”
# Author: Bernadette Burks
# Created: Sept. 14, 2025


from src.training import models
from src.evaluation import plot_confusion
from src.prediction import predict_disease
from src.preprocessing import X_resampled, y_resampled

def main():

    print("Training and evaluation complete.")

    # Example prediction
    result = predict_disease(
        "itching, skin_rash, nodal_skin_eruptions"
    )

    print("\nPrediction Results:")
    
    for model, prediction in result.items():
        print(f"{model}: {prediction}")


if __name__ == "__main__":
    main()