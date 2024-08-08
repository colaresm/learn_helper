from basic_libs import *
from classifiers import *



def generate_result(X_train_features, X_test_features, y_train, y_test):
    results = []

    classifiers = [
        (mlp_classifier),
        (linear_svc_classifier),
        (rbf_svc_classifier),
        (knn_classifier),
        (naive_bayes_classifier),
        (random_forest_classifier)
    ]

    for clf in classifiers:
        result = clf(X_train_features, X_test_features, y_train, y_test)
        results.append(result)
    
    df = pd.DataFrame(results)

    return df

