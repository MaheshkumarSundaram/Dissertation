import os, glob
import pandas as pd
from functools import reduce

data_path = "./Replica_Data/epadlcs_four_mountains_medavante.csv"
output_path = "./Replica_Data/epadlcs_4MT_medavante_preprocessed.csv"
df = pd.read_csv(data_path)
df['4mt_medavante_score'] = df.apply(lambda x: x.value_counts(), axis=1)[['Correct']]
df.drop(df.columns.difference(['patient_id', '4mt_medavante_score']), 1, inplace=True)
df.to_csv(output_path, encoding='utf-8', index=False)
data_path = "./Replica_Data/epadlcs_four_mountains_uedin.csv"
output_path = "./Replica_Data/epadlcs_4MT_uedin_preprocessed.csv"
df = pd.read_csv(data_path)
df['4mt_uedin_score'] = df.apply(lambda x: x.value_counts(), axis=1)[['CORRECT']]
df.drop(df.columns.difference(['patient_id', '4mt_uedin_score']), 1, inplace=True)
df.to_csv(output_path, encoding='utf-8', index=False)
data_path = "./Replica_Data/epadlcs_vr_supermarket_trolley.csv"
output_path = "./Replica_Data/epadlcs_vr_supermarket_trolley_preprocessed.csv"
df = pd.read_csv(data_path)
df['supermarket_trolley_score'] = df.apply(lambda x: x.value_counts(), axis=1)[['Correct']]
df.drop(df.columns.difference(['patient_id', 'supermarket_trolley_score']), 1, inplace=True)
df.to_csv(output_path, encoding='utf-8', index=False)