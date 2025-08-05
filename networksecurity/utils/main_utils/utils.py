import yaml
import dill
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def write_yaml_file(file_path:str,content:object,replace :bool=False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as file:
            yaml.dump(content,file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def save_numpy_array_data(file_path:str,array:np.array):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise NetworkSecurityException(e,sys)

def save_object(file_path:str,object:object)->None:
    try:
        logging.info("Entered the save_object function of mainUtils class")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(object,file_obj)
        logging.info("Exiting the save_object fucntion of mainUtils class")
    except Exception as e:
        raise NetworkSecurityException(e,sys)

def load_object(file_path:str)->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} does not exists")
        with open(file_path,'rb') as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def load_numpy_array(file_path:str)->np.array:
    try:
        with open(file_path,'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def evaluate_model(X_train, Y_train, X_test, Y_test, models, params):
    try:
        report = {}
        
        # Iterate through models dictionary properly
        for model_name, model in models.items():
            try:
                # Get parameters for this specific model
                para = params.get(model_name, {})
                
                if para:  # Only perform GridSearchCV if parameters are provided
                    gs = GridSearchCV(model, para, cv=3, scoring='accuracy', n_jobs=-1)
                    gs.fit(X_train, Y_train)
                    
                    # Set the best parameters to the model
                    model.set_params(**gs.best_params_)
                    logging.info(f"Best parameters for {model_name}: {gs.best_params_}")
                
                # Fit the model
                model.fit(X_train, Y_train)
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # For classification, use accuracy score instead of r2_score
                # r2_score is for regression, accuracy_score is for classification
                train_model_score = accuracy_score(Y_train, y_train_pred)
                test_model_score = accuracy_score(Y_test, y_test_pred)
                
                # Store the test score in report
                report[model_name] = test_model_score
                
                logging.info(f"{model_name}: Train Accuracy = {train_model_score:.4f}, Test Accuracy = {test_model_score:.4f}")
                
            except Exception as model_error:
                logging.warning(f"Error training {model_name}: {str(model_error)}")
                report[model_name] = 0.0  # Set a low score for failed models
                continue
        
        return report


    except Exception as e:
        raise NetworkSecurityException(e,sys)