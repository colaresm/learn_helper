from basic_libs import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score



def mlp_classifer(X_train_features, X_test_features, y_train, y_test):

    param_grid = {
        'hidden_layer_sizes': [(2**i,) for i in range(3, 8)]   
    }
   
    mlp = MLPClassifier(max_iter=500, random_state=42)

    
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)

    
    grid_search.fit(X_train_features, y_train)

    best_params = grid_search.best_params_

    best_model = grid_search.best_estimator_
   
    y_pred = best_model.predict(X_test_features)

    accuracy = accuracy_score(y_test, y_pred)
    
    print("Acurácia no conjunto de teste:", accuracy)

    print("Melhores parâmetros encontrados:", best_params)