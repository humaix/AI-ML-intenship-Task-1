
import argparse
import json
import joblib
import pandas as pd

from src.data_cleaner import DataCleaner
from src.feature_engineer import FeatureEngineer


def predict(input_data):
    # laoding the trained model and predict kidney disease.
    model_path = "models/kidney_disease_model.joblib"
    pipeline = joblib.load(model_path)
    patient_df = pd.DataFrame([input_data])
    # apply the same preprocessing used during training
    cleaner = DataCleaner()
    patient_df = cleaner.clean_data(patient_df)
    engineer = FeatureEngineer()
    patient_df = engineer.engineer_features(patient_df)
    prediction = pipeline.predict(patient_df)[0]
    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(patient_df)[0]
    return prediction, probabilities


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Kidney Disease Prediction")

    parser.add_argument("--input",type=str,required=True, help="Patient information in JSON format")

    args = parser.parse_args()
    try:
        patient_data = json.loads(args.input)
        prediction, probabilities = predict(patient_data)
        if prediction == 1:
            diagnosis = "CKD (Chronic Kidney Disease)"
        else:
            diagnosis = "Healthy (Not CKD)"

        print(f"\nDiagnosis: {diagnosis}")

        if probabilities is not None:
            print(f"Probability of CKD: {probabilities[1]:.4f}")

    except Exception as error:
        print(f"Error: {error}")