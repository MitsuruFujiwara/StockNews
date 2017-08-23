import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

def main():
    # load Data
    df = pd.read_hdf('data.h5', key='train_selected', mode='r')

    # set data
    trX, trY = df.drop('Class', axis=1).values, df['Class'].values

    # define Xgboost Classifier
    XGB = XGBClassifier()

    prm_learning_rate = [0.01, 0.10, 0.20]
    prm_max_depth = [10, 50, 100, 200, 300, 400, 500, 1000, 1500]
    prm_n_estimators = [10, 100, 1000]
    prm_min_child_weight = [0.5, 0.75, 1.0]

    param_grid = [{'learning_rate':prm_learning_rate,
                   'max_depth':prm_max_depth,
                   'n_estimators': prm_n_estimators,
                   'min_child_weight': prm_min_child_weight,
                   'subsample': [0.6],
                   'colsample_bytree' :[0.6],
                   'colsample_bylevel':[0.6],
#                   'reg_alpha': [1,10,20],
#                   'reg_lambda': [1,10,20]
                   }]

    gs = GridSearchCV(estimator=XGB, param_grid=param_grid, scoring='f1_weighted', cv=3, n_jobs=-1)

    gs.fit(trX, trY)

    result = pd.DataFrame(gs.cv_results_)

    result.to_csv('result_xgb.csv')

    joblib.dump(gs.best_estimator_, 'xgb.pkl')

    print(gs.best_score_)
    print(gs.best_params_)
    print(result)

if __name__ == '__main__':
    main()
