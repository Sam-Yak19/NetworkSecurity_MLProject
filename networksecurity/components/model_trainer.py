import os
import sys
import numpy as np
import pandas as pd
import tempfile

from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException

from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array

from networksecurity.utils.ml_utils.metrics.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.models.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import evaluate_model

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

import mlflow
import dagshub
dagshub.init(repo_owner='Sam-Yak19', repo_name='NetworkSecurity_MLProject', mlflow=True)


class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def track_ml_flow(self,best_model,classificationmetric):
        with mlflow.start_run():
            f1_score=classificationmetric.f1_score
            precision_score=classificationmetric.precison_score
            recall_score=classificationmetric.recall_score

            mlflow.log_metric("F1_Score",f1_score)
            mlflow.log_metric("Precision Score",precision_score)
            mlflow.log_metric("Recall Score",recall_score)
            try:
                # Create a temporary directory to save the model.
                with tempfile.TemporaryDirectory() as tmpdir:
                    model_path = os.path.join(tmpdir, "model")
                    # Save the scikit-learn model to the temporary path.
                    mlflow.sklearn.save_model(best_model, model_path)
                    # Log the entire directory as an artifact.
                    mlflow.log_artifacts(model_path, artifact_path="model")
            except Exception as e:
                    # Log the error but don't stop the run, just in case
                    # it's a transient issue.
                    logging.error(f"Failed to log model as artifact: {e}")


    def train_model(self,X_train,Y_train,X_test,Y_test):
        models={
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
        params={
            "Decision Tree": {
                'criterion':['gini', 'entropy'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
            
        }

        model_report:dict=evaluate_model(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models,params=params)

        best_model_score=max(model_report.values())

        best_model_name = None
        for model_name, score in model_report.items():
            if score == best_model_score:
                best_model_name = model_name
                break

        logging.info(f"Best model: {best_model_name} with score: {best_model_score}")
        
        if best_model_name is None:
            raise Exception("Could not determine best model")

        best_model = models[best_model_name]
        
        best_model.fit(X_train,Y_train)
        y_train_pred=best_model.predict(X_train)
        classification_train_metric=get_classification_score(y_true=Y_train,y_pred=y_train_pred)

        ##Write function for MLflow
        self.track_ml_flow(best_model,classification_train_metric)

        y_test_pred=best_model.predict(X_test)
        classification_test_metric=get_classification_score(y_true=Y_test,y_pred=y_test_pred)

        self.track_ml_flow(best_model,classification_test_metric)

        preprocessor=load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        Network_Model=NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,object=Network_Model)

        save_object("final_model/model.pkl",best_model)

        ##Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric)
        
        logging.info(f"Model Trainer artifact{model_trainer_artifact}")

        return model_trainer_artifact


    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path=self.data_transformation_artifact.transformed_train_file_path
            test_file_path=self.data_transformation_artifact.transformed_test_file_path

            ##loading training array and testing array
            train_arr=load_numpy_array(train_file_path)
            test_arr=load_numpy_array(test_file_path)

            ##split the data
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)

            return model_trainer_artifact


        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
