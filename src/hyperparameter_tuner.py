
from sklearn.model_selection import GridSearchCV
import logging


class HyperparameterTuner:

    def __init__(self, cv_folds=5, random_state=42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.logger = logging.getLogger("HyperparameterTuner")

    def tune_pipeline(self, pipeline, param_grid, X_train, y_train):
        # performing gridSearchCV on the selected pipeline and return the beest trained model
        self.logger.info("Starting hyperparameter tuning...")
        # GridSearchCV expects pipeline parameters
        search_grid = {}
        for parameter, values in param_grid.items():
            search_grid[f"classifier__{parameter}"] = values
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=search_grid,
            cv=self.cv_folds,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1)
        grid_search.fit(X_train, y_train)
        self.logger.info("Hyperparameter tuning completed.")
        self.logger.info( f"Best cross-validation accuracy: {grid_search.best_score_:.4f}" )
        return (grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_)