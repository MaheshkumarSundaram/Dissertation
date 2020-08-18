import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV


def plot_confusion_matrix(y_test, model_test):
    cm = metrics.confusion_matrix(y_test, model_test)
    plt.figure(1)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Low', 'High']
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    s = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))
    plt.savefig('ConfusionMatrix.png', bbox_inches='tight')
    print("\nConfusion Matrix: ", cm)
    total = sum(sum(cm))
    specificity = cm[0, 0]/(cm[0, 0]+cm[0, 1])
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print("\nSensitivity: ", sensitivity)
    print("\nSpecificity: ", specificity)
    plt.show()


def report_performance(model):
    model_test = model.predict(X_test)
    print("\n\nClassification Report: ")
    print(metrics.classification_report(y_test, model_test))
    plot_confusion_matrix(y_test, model_test)


def roc_curves(model):
    predictions_test = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(predictions_test, y_test)
    roc_auc = auc(fpr, tpr)
    print('AUROC = %.6f' % metrics.auc(fpr, tpr))
    plt.figure(2)
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('ROC.png', bbox_inches='tight')
    plt.show()
    y_probabilities = model.predict_proba(X_test)[:, 1]
    pr, rc, thresholds = metrics.precision_recall_curve(y_test, y_probabilities)
    plt.plot(pr, rc, color='darkorange')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.savefig('PRcurve.png', bbox_inches='tight')
    plt.show()


def accuracy(model):
    pred = model.predict(X_test)
    accu = metrics.accuracy_score(y_test, pred)
    print("\nAcurracy Of the Model: ", accu, "\n\n")


data_path = "./Data/preprocessed_trail_data_categorized.csv"
data_set = pd.read_csv(data_path)
features = list(data_set.columns)
predicted_class = ['dementia_risk']
feature_classes = list(set(features) - set(predicted_class))

X = data_set[feature_classes].values
y = data_set[predicted_class].values
# y = preprocessing.label_binarize(y, classes=[0, 1, 2, 3, 4])
print(X.shape, y.shape)


def count_freq(x):
    (unique, counts) = np.unique(np.array(x), return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    return frequencies


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print("\n\nTrain Class:")
print(count_freq(y_train))
print("\n\nTest Class:")
print(count_freq(y_test))

smote = SMOTE(sampling_strategy=0.35)
under_sampling = RandomUnderSampler(sampling_strategy=0.35)
pipeline = Pipeline([('smote', smote), ('under_sampling', under_sampling)])
# pipeline = Pipeline([('smote', smote)])
X_train, y_train = pipeline.fit_resample(X_train, y_train)
print(X_train.shape, y_train.shape)

print("\n\nTrain Class:")
print(count_freq(y_train))
print("\n\nTest Class:")
print(count_freq(y_test))

fs = SelectKBest(score_func=f_classif, k='all')
classifier = RandomForestClassifier(min_samples_leaf=2, min_samples_split=100, max_depth=100, bootstrap=True,
                                    n_jobs=-1, max_features='sqrt', class_weight={0: 1, 1: 1.25},
                                    n_estimators=1400, criterion="gini", random_state=42)
pipeline = Pipeline([
    ('fs', fs),
    ('classifier', classifier)
])
pipeline.fit(X_train, y_train.ravel())
y_pred = pipeline.predict(X_test)

# Cross Validation Pipeline for hyper-parameter tuning.
# sm = SMOTE()
# us = RandomUnderSampler()
# fs = SelectKBest(score_func=f_classif)
# rf = RandomForestClassifier(random_state=42)
# pipeline = Pipeline([('sm', sm), ('us', us), ('fs', fs), ('rf', rf)])
# kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
# n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# max_features = ['auto', 'sqrt']
# criterion = ['gini', 'entropy']
# max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
# max_depth.append(None)
# min_samples_split = [2, 5, 10, 15, 100]
# min_samples_leaf = [1, 2, 5, 10]
# bootstrap = [True, False]
# k = [20,25,30,35,40,45,'all']
# smote_sampling_strategy = [0.3, 0.35, 0.4, 0.45, 0.5]
# under_sampling_strategy = [0.3, 0.35, 0.4, 0.45, 0.5]
# class_weights = [{0: 1, 1: 1.5}, {0: 1, 1: 1.25}, {0: 1, 1: 1.75}, 'balanced_subsample']
# hyperF = {'sm__sampling_strategy': smote_sampling_strategy,
#           'us__sampling_strategy': under_sampling_strategy,
#           'fs__k': k,
#           'rf__n_estimators': n_estimators,
#           'rf__max_features': max_features,
#           'rf__max_depth': max_depth,
#           'rf__min_samples_split': min_samples_split,
#           'rf__criterion': criterion,
#           'rf__min_samples_leaf': min_samples_leaf,
#           'rf__bootstrap': bootstrap,
#           'rf__class_weight': class_weights}
#
# rf_random = RandomizedSearchCV(pipeline, param_distributions=hyperF, scoring='roc_auc',
#                                n_iter=1000, cv=kf, verbose=2, random_state=42, n_jobs=-1)
# bestF = rf_random.fit(X_train, y_train.ravel())
#
# print("Best parameters set found on development set:")
# print(bestF.best_params_)
# report_performance(bestF)
# accuracy(bestF)
# roc_curves(bestF)

labels = {0: "Low", 1: "High"}
predictions = [labels[k] for k in y_pred]
actual = [labels[k] for k in y_test.flatten()]

feature_importances = pd.DataFrame(classifier.feature_importances_, index=feature_classes,
                                   columns=['Importance']).sort_values('Importance', ascending=False)
print(feature_importances)

feature_importances.nlargest(13,'Importance').plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Feature Importances')
plt.xlabel('Relative Importance')
plt.ylabel("Features")
plt.title("Feature Importance of Random Forest Model")
plt.savefig('./Plots/wrt_MMSE/Feature_Import.png', bbox_inches='tight')
plt.show()

print("\n\nPredicted Class:")
print(count_freq(y_pred))
print("\n\nActual Class:")
print(count_freq(y_test.flatten()))
print("\n\nConfusion Matrix:")
print(pd.crosstab(np.array(predictions), np.array(actual), rownames=['Predicted Risk'], colnames=['Actual Risk']))
report_performance(pipeline)
accuracy(pipeline)
roc_curves(pipeline)
# joblib.dump(classifier, 'random_forest_model.pkl')
