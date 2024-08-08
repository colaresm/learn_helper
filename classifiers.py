from basic_libs import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from evaluation_metrics import *


def mlp_classifier(X_train_features, X_test_features, y_train, y_test):

    X_train, X_test, y_train, y_test = buildData(X_train_features, X_test_features, y_train, y_test)

    param_grid = {'hidden_layer_sizes': [(2**i,) for i in range(3, 8)]}
   
    mlp = MLPClassifier(max_iter=500, random_state=42)
    
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
    
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    best_model = grid_search.best_estimator_

    return buildResults(best_model,X_train, X_test, y_train, y_test,best_params,"MLP")