import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


def mmse_output_categorization(data):
    if (data['mmse_total'] < 27) and (data['years_education'] >= 16):
        return 1
    elif (data['mmse_total'] <= 24) and (data['years_education'] < 16):
        return 1
    else:
        return 0


def mmse_categorization(x):
    if 0 <= x <= 24:
        return "MMSE_High"
    else:
        return "MMSE_Low"


def cdr_categorization(x):
    if 0.5 <= x <= 3:
        return "CDR_High"
    else:
        return "CDR_Low"


def gds_categorization(x):
    if x > 11:
        return "GDS_High"
    else:
        return "GDS_Low"


def psqi_categorization(x):
    if x > 5:
        return "PSQI_High"
    else:
        return "PSQI_Low"


def rbans_categorization(x):
    if x < 70:
        return "RBANS_High"
    else:
        return "RBANS_Low"


def stai_categorization(x):
    if x >= 54:
        return "STAI_High"
    else:
        return "STAI_Low"


def abeta_categorization(x):
    if x < 1000:
        return "Abeta_High"
    else:
        return "Abeta_Low"


def family_dementia_history_categorization(x):
    if x == 'Yes':
        return 'Family_risk_High'
    else:
        return 'Family_risk_Low'


def systolic_bp_categorization(x):
    if 90 <= x < 120:
        return 'Systolic_BP_Low'
    else:
        return 'Systolic_BP_High'


def diastolic_bp_categorization(x):
    if 60 <= x < 80:
        return 'Diastolic_BP_Low'
    else:
        return 'Diastolic_BP_High'


def medavante_categorization(x):
    if x <= 8:
        return '4mt_medavante_High'
    else:
        return '4mt_medavante_Low'


disease_list = ['hypertension', 'stroke', 'depression', 'hypercholesterolemia', 'pneumonia', 'diabetes',
                'head injury', 'brain injury', 'fall on head', 'brain contusion', 'chronic kidney', 'mci',
                'myocardial infarction', 'heart arrhythmia', 'atrial fibrillation', 'hepatitis c']
disease_pattern = '|'.join(disease_list)


def medical_history_categorization(data):
    if disease_pattern in data['medical_history_term']:
        return 'Medical_history_high'
    else:
        return 'Medical_history_low'


data_file = "./Data/trial_data.csv"
output_path1 = "./Data/preprocessed_trail_data.csv"
output_path2 = "./Data/preprocessed_trail_data_categorized.csv"
data_set = pd.read_csv(data_file)
data_set = data_set.replace([995], np.nan)
data_set.dropna(axis=0, how='any', inplace=True)
print(data_set.shape)
data_set.drop(list(data_set.columns)[0:2], axis=1, inplace=True)
data_set['abeta_1_42_result'] = pd.to_numeric(data_set.abeta_1_42_result.astype(str).str.replace(',', ''),
                                              errors='coerce').fillna(1700).astype(int)
# data_set['medical_history_term'] = data_set['medical_history_term'].str.lower()
# data_set['medical_history_risk'] = data_set.apply(medical_history_categorization, axis=1)
# print(data_set['medical_history_risk'].value_counts())
data_set['bmi'] = data_set['weight'] / ((data_set['height'] / 100) ** 2)
data_set.drop(['weight', 'height'], axis=1, inplace=True)
data_set['dementia_risk'] = data_set.apply(mmse_output_categorization, axis=1)
data_set.to_csv(output_path1, encoding='utf-8', index=False)
data_set['family_dementia_history'] = data_set['family_dementia_history'].apply(family_dementia_history_categorization)
print(data_set['dementia_risk'].value_counts())
data_set.drop('dementia_risk', axis=1, inplace=True)
features = list(data_set.columns)
print(features)
numerical_vars = data_set.select_dtypes([np.number]).columns
print(set(numerical_vars))
categorical_vars = list(set(features) - set(numerical_vars))
print(categorical_vars)
for var in categorical_vars:
    one_hot_df = pd.get_dummies(data_set[var])
    data_set = pd.concat([data_set, one_hot_df], axis=1)
    data_set.drop(var, axis=1, inplace=True)
data_set['dementia_risk'] = data_set.apply(mmse_output_categorization, axis=1)
data_set.drop(['mmse_total'], axis=1, inplace=True)
print(data_set.columns)
print(data_set.describe())
data_set.to_csv(output_path2, encoding='utf-8', index=False)
