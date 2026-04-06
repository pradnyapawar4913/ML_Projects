
import sys
import os
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import(
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("split training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            
            models = {
                "Random Forest " : RandomForestRegressor(),
                "Decision Tree ":DecisionTreeRegressor(),
                "Gradient Boosting ":GradientBoostingRegressor(),
                "Linear Regression ":LinearRegression(),
                "K - Neighbors Classifier ":KNeighborsRegressor(),
                "XGBClassifier ":XGBRegressor(),
                "catboosting classifier ":CatBoostRegressor(verbose=False),
                "Adaboost Classifier ":AdaBoostClassifier(),
            }
            
            model_report : dict = evaluate_models(X_train,y_train,X_test,y_test,
                                                         models= models)
            
            # to get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            # to get the best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score <0.6:
                raise CustomException("No best model found")
            
            logging.info("Best found model on both training and testing dataset")
            
            
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
            predicted = best_model.predict(X_test)
            
            score = r2_score(y_test,predicted)
            return score
        except Exception as e:
                raise CustomException(e, sys)
               
