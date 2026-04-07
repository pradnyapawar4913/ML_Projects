import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok = True)
        
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)        

# When you start learning Machine Learning 😎

def evaluate_models(x_train, y_train, x_test, y_test, models,params):
    try:
        report = {}

        for model_name, model in models.items():
            model.fit(x_train, y_train)
            para = params[model_name]
            
            gs = GridSearchCV(model,para,cv = 3,n_jobs = 1,verbose = 0,refit=True)
            gs.fit(x_train,y_train)

            # model.fit(x_train,y_train) # train model
            best_model = gs.best_estimator_
            # model.set_params(**gs.best_params_)
            # model.fit(x_train,y_train)

            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)
            
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            
            report[model_name] = test_model_score
            
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
            



