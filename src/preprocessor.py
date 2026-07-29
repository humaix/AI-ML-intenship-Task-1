


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import logging


class DataPreprocessor:
    def __init__(self,target_column,numerical_cols,categorical_cols,test_size=0.2,random_state=42):
        self.target_column = target_column
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.test_size = test_size
        self.random_state = random_state

        self.logger = logging.getLogger("DataPreprocessor")
        self.preprocessor = None

    def build_preprocessor(self):
        self.logger.info("Building preprocessing pipeline...")

        numerical_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])

        categorical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))
            ,("encoder", OneHotEncoder(handle_unknown="ignore",sparse_output=False))])

        transformer = ColumnTransformer(transformers=[("num",numerical_pipeline, self.numerical_cols),
                ("cat",categorical_pipeline, self.categorical_cols)],remainder="drop")

        self.preprocessor = transformer
        self.logger.info("Preprocessor created successfully.")
        return transformer

    def split_data(self, df):
        self.logger.info("Splitting dataset into train and test sets...")

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=self.test_size,random_state=self.random_state,stratify=y)

        self.logger.info(f"Training data: {X_train.shape}")
        self.logger.info(f"Testing data: {X_test.shape}")
        return X_train, X_test, y_train, y_test

