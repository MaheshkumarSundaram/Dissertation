# Dissertation

## Single Point of Access

Since the EPAD data set has various amounts of data, a single point of access with necessary information would be more appropriate for the model. 

Python script ```data_concat.py``` is used to create a single file with all the necessary data for modelling.


## Data Preprocessing

Data preprocessing has been carried out in ```data_preprocessing.py```.

Feature transformations like hanlding missing values and one-hot encoding have been carried out.

## Feature Selection

For determing the importance of each feature, SelectKBest is used in ```feature_selection.py```.

Best features can be analysed with the script.

## Random Forest Model

RF model has been devised in ```random_forest_implementation.py```.

### Handling Data Imbalance

Synthetic Minority Oversampling Technique (SMOTE) - a very popular oversampling method that was proposed to improve random oversampling has been used on the minority class.

Same way, random undersampling has been performed on the majority class.

A pipeline has been created to ensure the resampling of dataset such that the data becomes balanced. 

### Cross Validation

Cross validation has been used to ensure that the hyper-parameters selected perform the best.

A pipeline was created for a 10-fold RepeatedStratifiedKFold and RandomizedSearchCV to choose the best parameters for the model.

After the data has been analysed by this pipeline, the chosen parameters have been used for the Random Forest Model.




