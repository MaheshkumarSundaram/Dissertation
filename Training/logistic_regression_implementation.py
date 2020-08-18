from collections import Counter

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV


def plot_confusion_matrix(y_test, model_test):
    cm = metrics.confusion_matrix(y_test, model_test)
    plt.figure(1)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['N', 'Y']
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
    plt.show()
    specificity = cm[0, 0]/(cm[0, 0]+cm[0, 1])
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print("\nSensitivity: ", sensitivity)
    print("\nSpecificity: ", specificity)

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
print(X_train.shape, y_train.shape)

smote = SMOTE(sampling_strategy=0.35)
under_sampling = RandomUnderSampler(sampling_strategy=0.4)
pipeline = Pipeline([('smote', smote), ('under_sampling', under_sampling)])
pipeline = Pipeline([('smote', smote)])
X_train, y_train = pipeline.fit_resample(X_train, y_train)
print(X_train.shape, y_train.shape)

fs = SelectKBest(f_classif, k='all')
classifier = LogisticRegression()
pipeline = Pipeline([
    ('fs', fs),
    ('classifier', classifier)
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

labels = {0: "Low", 1: "High"}
predictions = [labels[k] for k in y_pred]
actual = [labels[k] for k in y_test.flatten()]


def count_freq(x):
    (unique, counts) = np.unique(np.array(x), return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    return frequencies


print("\n\nPredicted Class:")
print(count_freq(y_pred))
print("\n\nActual Class:")
print(count_freq(y_test.flatten()))
print("\n\nConfusion Matrix:")
print(pd.crosstab(np.array(predictions), np.array(actual), rownames=['Predicted Risk'], colnames=['Actual Risk']))
report_performance(pipeline)
accuracy(pipeline)
roc_curves(pipeline)