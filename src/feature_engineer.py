
import logging
import numpy as np
import pandas as pd


class FeatureEngineer:
    # creating additional features from the cleaned kidney disease dataset.
    def __init__(self):
        self.logger = logging.getLogger("FeatureEngineer")
    def engineer_features(self, data):
        self.logger.info("Starting feature engineering...")
        data = data.copy()
        data = self.create_age_group(data)
        data = self.create_bgr_hemo_ratio(data)
        data = self.create_kidney_function_score(data)
        self.logger.info("Feature engineering completed.")
        return data
    def create_age_group(self, data):
        if "age" in data.columns:
            age_bins = [0, 18, 35, 55, 100]
            age_labels = [0, 1, 2, 3]
            age_group = pd.cut(
                data["age"],
                bins=age_bins,
                labels=age_labels,
                include_lowest=True
            )
            age_group = age_group.astype(float)
            age_group = age_group.fillna(2)
            data["age_group"] = age_group
            self.logger.info("Age group feature created.")
        else:
            data["age_group"] = 2

        return data

    def create_bgr_hemo_ratio(self, data):
        if "bgr" in data.columns and "hemo" in data.columns:
            hemoglobin = data["hemo"].replace(0, np.nan)
            ratio = data["bgr"] / hemoglobin
            ratio = ratio.fillna(0)
            data["bgr_hemo_ratio"] = ratio
            self.logger.info("BGR/Hemoglobin ratio created.")

        else:
            data["bgr_hemo_ratio"] = 0
        return data

    def create_kidney_function_score(self, data):
        required_columns = ["sc", "bu", "sod", "pot"]
        if all(column in data.columns for column in required_columns):
            sc = (data["sc"] - data["sc"].min()) / (
                data["sc"].max() - data["sc"].min() + 1e-8
            )
            bu = (data["bu"] - data["bu"].min()) / (
                data["bu"].max() - data["bu"].min() + 1e-8
            )

            sod = 1 - (
                (data["sod"] - data["sod"].min()) /
                (data["sod"].max() - data["sod"].min() + 1e-8)
            )

            pot = (data["pot"] - data["pot"].min()) / (
                data["pot"].max() - data["pot"].min() + 1e-8
            )

            score = (sc + bu + sod + pot) / 4
            score = score.fillna(0)

            data["kidney_function_score"] = score

            self.logger.info("Kidney function score created.")

        else:
            data["kidney_function_score"] = 0

        return data

