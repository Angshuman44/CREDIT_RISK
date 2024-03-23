import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        return accuracy, precision, recall, f1

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Support Vector Machine": SVC(),
                "Logistic Regression": LogisticRegression(),
                "XGBoost": XGBClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier()
            }
            params = {
                "Support Vector Machine": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto']
                },
                "Logistic Regression": {
                    'penalty': ['l2'],
                    'C': [0.1, 1, 10],
                    'max_iter': [1000, 2000, 3000]

                },
                "XGBoost": {
                    'learning_rate': [0.1, 0.01, 0.05],
                    'n_estimators': [100, 200, 300]
                },
                "Random Forest": {
                    'n_estimators': [100, 200, 300]
                },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random']
                }
            }
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            print(model_report)
            best_model_name = max(model_report, key=lambda x: model_report[x]['accuracy'])
            best_model_score = model_report[best_model_name]['accuracy']

            print("This is the best model based on accuracy:")
            print(best_model_name)

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info("Best found model based on accuracy on both training and testing dataset")

            best_model = models[best_model_name] 
            # if not hasattr(best_model, 'fit'):
            #     raise ValueError("The selected model has no 'fit' method, so it cannot be fitted with training data.")
            # best_model.fit(X_train, y_train)
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )            
            predicted = best_model.predict(X_test)

            accuracy, precision, recall, f1 = self.eval_metrics(y_test, predicted)  # Replace with your evaluation method

            return accuracy, precision, recall, f1

        except Exception as e:
            raise CustomException(e, sys)
        


# if __name__=='__main__':
#     obj=ModelTrainer()
#     obj.initiate_model_trainer('artifacts/train.csv','artifacts/test.csv')
        
