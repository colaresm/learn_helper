from basic_libs import *
from classifiers import *


def generate_result(X_train_features, X_test_features, y_train, y_test):

    results=[]

    classifiers = [mlp_classifier(X_train_features, X_test_features, y_train, y_test)]

    for clf in classifiers:
        results.append(clf)
    
    df = pd.DataFrame(results)

    return df

