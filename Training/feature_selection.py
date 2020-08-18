import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

data_set = pd.read_csv("./Data/preprocessed_trail_data_categorized.csv")
cols = list(range(10, 26))
data_set.drop(data_set.columns[cols],axis=1,inplace=True)
data_set.rename(columns={'rbans_story_recall': 'rbans'}, inplace=True)
print(data_set.describe())
features = list(data_set.columns)
predicted_class = ['dementia_risk']
feature_classes = list(set(features) - set(predicted_class))
X = data_set[feature_classes].values  # independent columns
y = data_set[predicted_class].values  # target column i.e risk

# apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=f_classif, k='all')
fit = bestfeatures.fit(X, y.ravel())
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(feature_classes)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']
print(featureScores.nlargest(28, 'Score'))

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

X = data_set.iloc[:, 0:41]  # independent columns
y = data_set.iloc[:, -1]  # target column i.e risk

model = ExtraTreesClassifier()
model.fit(X, y.ravel())
print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
# plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(13).plot(kind='barh')
plt.xlabel("Proportion of Importance")
plt.ylabel("Features")
plt.title("Feature Selection based on importance")
plt.savefig('./Plots/wrt_MMSE/Feature Selection.png', bbox_inches='tight')
plt.show()

