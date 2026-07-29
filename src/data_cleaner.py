






import logging
import pandas as pd
import numpy as np


class DataCleaner:
    # Cleans the kidney disease dataset before training.
    def __init__(self):
        self.logger = logging.getLogger("DataCleaner")

    def clean_data(self, data):
        # Apply all cleaning steps on the dataset.
        self.logger.info("Cleaning dataset...")
        cleaned_data = data.copy()
        # Removing id column
        if "id" in cleaned_data.columns:
            cleaned_data.drop(columns=["id"], inplace=True)
            self.logger.info("ID column removed.")
        # replace invalid values with NaN
        invalid_entries = ["?", "\t?", "\t"]
        cleaned_data.replace(invalid_entries, np.nan, inplace=True)
        # remove spaces from text columns
        text_columns = cleaned_data.select_dtypes(include="object").columns
        for column in text_columns:
            updated_values = []
            for value in cleaned_data[column]:
                if isinstance(value, str):
                    updated_values.append(value.strip())
                else:
                    updated_values.append(value)
            cleaned_data[column] = updated_values
        # encode target column
        self.clean_target(cleaned_data)
        # convert numeric columns
        self.convert_numeric(cleaned_data)
        # remove duplicate rows
        duplicate_rows = cleaned_data.duplicated().sum()
        if duplicate_rows > 0:
            cleaned_data.drop_duplicates(inplace=True)
            self.logger.info(f"Removed {duplicate_rows} duplicate rows.")
        # Fill missing values
        self.fill_missing_values(cleaned_data)
        self.logger.info("Data cleaning completed.")
        self.logger.info(f"Final dataset shape: {cleaned_data.shape}")
        return cleaned_data

    def clean_target(self, data):
        # convert target labels into numeric values.
        if "classification" in data.columns:
            data["classification"] = (data["classification"].astype(str).str.strip())
            target_mapping = {"ckd": 1,"notckd": 0}
            data["classification"] = data["classification"].map(target_mapping)
            most_common = data["classification"].mode()[0]
            data["classification"] = (data["classification"].fillna(most_common).astype(int))
            self.logger.info("Target column encoded successfully.")
    def convert_numeric(self, data):
        # Convert selected columns to numeric datatype.
        numeric_columns = [
            "age", "bp", "sg", "al", "su",
            "bgr", "bu", "sc", "sod",
            "pot", "hemo", "pcv", "wc", "rc"
        ]
        for column in numeric_columns:
            if column not in data.columns:
                continue
            data[column] = pd.to_numeric(data[column],errors="coerce")
        self.logger.info("Numeric columns converted.")
    def fill_missing_values(self, data):
        # Fill missing values in numerical and categorical columns.
        numerical_columns = data.select_dtypes(include=np.number).columns
        for column in numerical_columns:
            if data[column].isnull().sum() > 0:
                median_value = data[column].median()
                data[column] = data[column].fillna(median_value)
        categorical_columns = data.select_dtypes(
            include=["object", "string"]
        ).columns
        for column in categorical_columns:
            if data[column].isnull().sum() > 0:
                if not data[column].mode().empty:
                    replacement_value = data[column].mode()[0]
                else:
                    replacement_value = "Unknown"
                data[column] = data[column].fillna(replacement_value)
        self.logger.info("Missing values handled.")