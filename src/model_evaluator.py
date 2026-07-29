



import os
import json
import logging

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (accuracy_score,precision_score,recall_score, f1_score,roc_auc_score,confusion_matrix,roc_curve,auc)


class ModelEvaluator:

    def __init__(self, reports_dir="reports"):
        self.reports_dir = reports_dir
        self.figures_dir = os.path.join(reports_dir, "figures")

        os.makedirs(self.figures_dir, exist_ok=True)
        self.logger = logging.getLogger("ModelEvaluator")
    def evaluate_model(self, model_name, pipeline, X_test, y_test):
        # Evaluate a trained model on the test set.
        self.logger.info(f"Evaluating model: {model_name}")
        predictions = pipeline.predict(X_test)
        probabilities = None
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(X_test)[:, 1]
        metrics = { "accuracy": float( accuracy_score(y_test, predictions) ),
            "precision": float( precision_score(y_test, predictions,zero_division=0)),
            "recall": float(recall_score(y_test,predictions,zero_division=0)),
            "f1_score": float( f1_score(  y_test, predictions, zero_division=0)) }

        if probabilities is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_test,probabilities))

        self.plot_confusion_matrix( model_name, y_test, predictions)

        if probabilities is not None:
            self.plot_roc_curve(model_name,  y_test, probabilities )

        return metrics

    def plot_confusion_matrix(self, model_name, y_true, y_pred):
        # create and save confusion matrix.

        matrix = confusion_matrix(y_true,y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap( matrix, annot=True,fmt="d", cmap="Blues", cbar=False,xticklabels=["Not CKD", "CKD"],yticklabels=["Not CKD", "CKD"])

        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        output_path = os.path.join(self.figures_dir,f"confusion_matrix_{model_name}.png" )
        plt.savefig( output_path, dpi=300)
        plt.close()

    def plot_roc_curve(self, model_name, y_true, y_prob):
        # Create and save ROC curve.
        fpr, tpr, _ = roc_curve(y_true,y_prob)
        roc_value = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot( fpr,    tpr, color="darkorange",    linewidth=2,    label=f"AUC = {roc_value:.2f}")

        plt.plot([0, 1],[0, 1], linestyle="--", linewidth=2 )

        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.tight_layout()
        output_path = os.path.join( self.figures_dir, f"roc_curve_{model_name}.png")

        plt.savefig( output_path, dpi=300)

        plt.close()

    def save_metrics(self, metrics, filename="evaluation.json"):
        # save evaluation metrics as json.

        output_file = os.path.join(self.reports_dir,filename)

        with open(output_file, "w") as file: json.dump(metrics,file,indent=4)
        self.logger.info("Evaluation results saved successfully.")
