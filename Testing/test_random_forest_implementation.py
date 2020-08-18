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
    model_test = model.predict(X)
    print("\n\nClassification Report: ")
    print(metrics.classification_report(y, model_test))
    plot_confusion_matrix(y, model_test)


def roc_curves(model):
    predictions_test = model.predict(X)
    fpr, tpr, thresholds = roc_curve(predictions_test, y)
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
    y_probabilities = model.predict_proba(X)[:, 1]
    pr, rc, thresholds = metrics.precision_recall_curve(y, y_probabilities)
    plt.plot(pr, rc, color='darkorange')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.savefig('PRcurve.png', bbox_inches='tight')
    plt.show()


def accuracy(model):
    pred = model.predict(X)
    accu = metrics.accuracy_score(y, pred)
    print("\nAcurracy Of the Model: ", accu, "\n\n")


data_path = "./Data/test_preprocessed_trail_data_categorized.csv"
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


loaded_model = joblib.load("random_forest_model.pkl")
y_pred = loaded_model.predict(X)


report_performance(loaded_model)
accuracy(loaded_model)
roc_curves(loaded_model)

