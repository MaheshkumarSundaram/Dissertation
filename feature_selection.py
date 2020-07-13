import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

data_set = pd.read_csv("./Data/preprocessed_trail_data_categorized.csv")
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
