import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.preprocessing import label_binarize

def main():
    # load Data
    df = pd.read_hdf('data.h5', key='train_selected', mode='r')

    # set data
    trX, trY = df.drop('Class', axis=1).values, df['Class'].values

    # define Multi Layer Perceptron
    mlp = MLPClassifier()

    prm_hidden_layer_sizes = [1000, 1500, 1943]
    prm_activation = ['relu', 'identity', 'logistic', 'tanh']
    prm_solver=['adam']
    prm_max_iter = [100, 1000, 5000]

    param_grid = [{'hidden_layer_sizes':prm_hidden_layer_sizes, 'activation':prm_activation,\
    'solver':prm_solver, 'max_iter':prm_max_iter}]

    gs = GridSearchCV(estimator=mlp, param_grid=param_grid, scoring='f1_weighted', cv=3, n_jobs=-1)

    gs.fit(trX, trY)

    result = pd.DataFrame(gs.cv_results_)

    result.to_csv('result_mlp.csv')

    joblib.dump(gs.best_estimator_, 'mlp.pkl')

    print(gs.best_score_)
    print(gs.best_params_)
    print(result)


if __name__ == '__main__':
    main()
