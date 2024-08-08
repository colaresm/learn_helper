from basic_libs import *

def formatTime(time,isTestingTime=False):
  if isTestingTime:
    return round(1000*np.mean(time),3)
  return round(np.mean(time),2)

def formatStdTime(time,isTestingTime=False):
  if isTestingTime:
    return round(1000*np.std(time),3)
  return round(np.std(time),4)

def formatMetric(metric):
  return round(100*np.mean(metric),3)

def formatStd(metric):
  print(metric)
  return round(100*np.std(metric),3)

def calculateTrainingTime(clf,X_train, y_train):
    start_time = time.time()
    clf.fit(X_train,y_train)
    end_time = time.time()
    training_time =end_time - start_time
    return training_time

def calculateTestingTime(clf,X_test, y_test):
    start_time = time.time()
    y_pred = clf.predict(X_test)
    testing_time = time.time() - start_time
    return testing_time

def export_metric_to_statistical_analysis(accuracies,cnn,classifier):
    accuracies_formated=[]

    for acc in accuracies:
       accuracies_formated.append(formatMetric(acc))

    result={"extractor":cnn,"classifier":classifier,"accuracies":accuracies}
    
    df=pd.DataFrame(result)

    df.to_csv(cnn+"_"+classifier+".csv")


def buildResults(clf,X_train, X_test, y_train, y_test,best_params,classifier):
    training_times = []
    testing_times = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(0,10):

        X_train, X_test, y_train, y_test = buildData(X_train, X_test, y_train, y_test)

        training_times.append(calculateTrainingTime(clf,X_train, y_train))

        testing_times.append(calculateTestingTime(clf,X_test, y_test))

        accuracies.append(accuracy_score(y_test, clf.predict(X_test)))

        precisions.append(precision_score(y_test, clf.predict(X_test), average='macro'))

        recalls.append(recall_score(y_test, clf.predict(X_test), average='macro'))

        f1_scores.append(f1_score(y_test, clf.predict(X_test), average='macro'))

    export_metric_to_statistical_analysis(accuracies,"cnn",classifier)

    results={
    "Classifier":classifier,
    "training time (s)":formatTime(training_times),  "training time std":formatStdTime(training_times),
    "testing time (ms)":formatTime(testing_times,True),  "testing time std":formatStdTime(testing_times,True),
    "accuracy":formatMetric(accuracies),"accuracy std":formatStd(accuracies),
    "precision":formatMetric(precisions),"precision std":formatStd(precisions),
    "recall":formatMetric(recalls),"recall std":formatStd(recalls),
    "f1-score":formatMetric(f1_scores),"f1-score std":formatStd(f1_scores),
    "best params":best_params
    }

    return results

def buildData(X_train_features, X_test_features, y_train, y_test):
    X= np.concatenate((X_train_features, X_test_features), axis=0)
    y=np.concatenate((y_train, y_test), axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    return   X_train, X_test, y_train, y_test



