

import os
import yaml
import joblib
import logging

from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.feature_engineer import FeatureEngineer
from src.preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.hyperparameter_tuner import HyperparameterTuner
from src.model_evaluator import ModelEvaluator


class PipelineRunner:

    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()

        self.configure_logging()

        self.logger = logging.getLogger("PipelineRunner")

    def load_config(self):
        # read configuration file.

        with open(self.config_path, "r") as file:
            return yaml.safe_load(file)

    def configure_logging(self):
        # configure console and file logging."""

        logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[
                logging.StreamHandler(),
                logging.FileHandler("pipeline.log", mode="w")])

    def run(self):

        self.logger.info("=" * 60)
        self.logger.info("Kidney Disease Classification Pipeline Started")
        self.logger.info("=" * 60)


        # Load Dataset
        data_config = self.config["data"]
        loader = DataLoader(data_config["raw_data_path"])
        raw_df = loader.load_data()

        # Data Cleaning
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_data(raw_df)

        # Feature Engineering

        engineer = FeatureEngineer()

        processed_df = engineer.engineer_features(cleaned_df)
        os.makedirs( os.path.dirname(data_config["processed_data_path"]), exist_ok=True )

        processed_df.to_csv(data_config["processed_data_path"],index=False)

        # Data Preprocessing
        feature_config = self.config["features"]
        preprocessing = DataPreprocessor(
            target_column=data_config["target_column"],
            numerical_cols=feature_config["numerical_cols"],
            categorical_cols=feature_config["categorical_cols"],
            test_size=data_config["test_size"],
            random_state=data_config["random_state"]
        )
        transformer = preprocessing.build_preprocessor()
        X_train, X_test, y_train, y_test = preprocessing.split_data( processed_df )

        # Train Models
        model_config = self.config["model"]
        trainer = ModelTrainer(preprocessor=transformer,random_state=model_config["random_state"])

        trained_models = trainer.train_models( X_train, y_train)


        # Evaluate Baseline Models
        evaluator = ModelEvaluator()
        baseline_results = {}
        for model_name, pipeline in trained_models.items():
            metrics = evaluator.evaluate_model(model_name, pipeline,X_test,y_test)

            baseline_results[model_name] = metrics
        best_model = max(baseline_results,key=lambda name: baseline_results[name]["accuracy"])
        self.logger.info(f"Best model: {best_model}")

        # Hyperparameter Tuning
        parameter_grid = self.config["hyperparameters"].get(best_model,  {} )

        tuner = HyperparameterTuner(cv_folds=model_config["cv_folds"], random_state=model_config["random_state"])

        best_pipeline, best_parameters, best_cv_score = tuner.tune_pipeline(
            pipeline=trained_models[best_model],
            param_grid=parameter_grid,
            X_train=X_train,
            y_train=y_train
        )

        # Final Evaluation
        final_metrics = evaluator.evaluate_model( "tuned_" + best_model,
            best_pipeline,
            X_test,
            y_test)
        results = {
            "baseline_models": baseline_results,
            "best_model": {
                "name": best_model,
                "cv_accuracy": float(best_cv_score),
                "test_metrics": final_metrics}}

        evaluator.save_metrics(results)

        # Save Final Model
        model_path = model_config["model_save_path"]
        os.makedirs(os.path.dirname(model_path),exist_ok=True)

        joblib.dump( best_pipeline,  model_path)

        self.logger.info(f"Model saved at {model_path}")


        self.logger.info("Pipeline Completed Successfully")
