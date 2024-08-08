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


def linear_svc_classifier(X_train_features, X_test_features, y_train, y_test):

    X_train, X_test, y_train, y_test = buildData(X_train_features, X_test_features, y_train, y_test)

    param_grid = {'C': np.logspace(-5, 15, 4)}

    svc = LinearSVC(max_iter=1000, random_state=42)
    
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
    
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    best_model = grid_search.best_estimator_

    return buildResults(best_model, X_train, X_test, y_train, y_test, best_params, "SVM (Linear)")

def rbf_svc_classifier(X_train_features, X_test_features, y_train, y_test):

    X_train, X_test, y_train, y_test = buildData(X_train_features, X_test_features, y_train, y_test)

    param_grid = {
        'C': np.logspace(-5, 15, 4),
        'gamma': np.logspace(-15, 3, 4)
    }

    svc = SVC(kernel='rbf', random_state=42)
    
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
    
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    best_model = grid_search.best_estimator_

    return buildResults(best_model, X_train, X_test, y_train, y_test, best_params, "SVM (RBF)")



def knn_classifier(X_train_features, X_test_features, y_train, y_test):

    X_train, X_test, y_train, y_test = buildData(X_train_features, X_test_features, y_train, y_test)

    param_grid = {
        'n_neighbors': [3, 5, 7, 13] 
    }

    knn = KNeighborsClassifier()
    
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
    
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    best_model = grid_search.best_estimator_

    return buildResults(best_model, X_train, X_test, y_train, y_test, best_params, "KNN")


def naive_bayes_classifier(X_train_features, X_test_features, y_train, y_test):

    X_train, X_test, y_train, y_test = buildData(X_train_features, X_test_features, y_train, y_test)

    param_grid = {
        'var_smoothing': np.logspace(-9, 0, 4)  
    }

    gnb = GaussianNB()
    
    grid_search = GridSearchCV(estimator=gnb, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
    
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    best_model = grid_search.best_estimator_

    return buildResults(best_model, X_train, X_test, y_train, y_test, best_params, "Bayes")


def random_forest_classifier(X_train_features, X_test_features, y_train, y_test):

    X_train, X_test, y_train, y_test = buildData(X_train_features, X_test_features, y_train, y_test)

    param_grid = {'n_estimators': np.linspace(50, 3000, 4, dtype=int) }

    rf = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
    
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    best_model = grid_search.best_estimator_

    return buildResults(best_model, X_train, X_test, y_train, y_test, best_params, "Random Forest")

