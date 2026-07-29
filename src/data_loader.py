import os
import logging
import pandas as pd


class DataLoader:

    # Simple class to load the kidney disease dataset and show some basic information about it.


    def __init__(self, file_path):
        self.file_path = file_path
        self.logger = logging.getLogger("DataLoader")

    def load_data(self):
        # read the csv file and return a dataframe.

        if not os.path.isfile(self.file_path):
            self.logger.error("dataset file not found.")
            raise FileNotFoundError(f"cannot find file: {self.file_path}")

        try:
            data = pd.read_csv(self.file_path)
            self.logger.info(f"dataset loaded successfully ({data.shape[0]} rows, {data.shape[1]} columns)")
            return data

        except Exception as error:
            self.logger.error(f"failed to load dataset: {error}")
            raise

    def display_basic_info(self, data):
        # Display basic information about the dataset.

        self.logger.info("\n")
        self.logger.info("basic dataset information")
        self.logger.info(f"\nShape : {data.shape}")
        self.logger.info("\ncolumns and data types:")
        self.logger.info(data.dtypes)
        self.logger.info("\nmissing values:")
        self.logger.info(data.isnull().sum())

        self.logger.info("\nsummary statistics:")
        self.logger.info(data.describe())
        self.logger.info("\nfirst five records:")
        self.logger.info(data.head())


