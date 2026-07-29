
import logging

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

class ModelTrainer:
    # training multiple classification models using the same preprocessing pipeline.

    def __init__(self, preprocessor, random_state=42):
        self.preprocessor = preprocessor
        self.random_state = random_state
        self.logger = logging.getLogger("ModelTrainer")
        self.models = {}
        self.pipelines = {}
    def get_base_models(self):
        models = {
            "logistic Regression": LogisticRegression(
                random_state=self.random_state,
                max_iter=1000),
            "random forest": RandomForestClassifier(random_state=self.random_state),
            "decision tree": DecisionTreeClassifier(random_state=self.random_state),
            "Gradient Boosting": GradientBoostingClassifier(random_state=self.random_state)}

        return models

    def train_models(self, X_train, y_train):
        self.logger.info("Training models...")
        self.models = self.get_base_models()
        for model_name, model in self.models.items():
            self.logger.info(f"Training {model_name}")
            pipeline = Pipeline(steps=[("preprocessor", self.preprocessor),("classifier", model)] )
            pipeline.fit(X_train, y_train)
            self.pipelines[model_name] = pipeline
        self.logger.info("All models trained successfully.")
        return self.pipelines