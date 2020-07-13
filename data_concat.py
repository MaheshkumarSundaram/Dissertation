import os, glob
import pandas as pd
from functools import reduce

path = "./Replica_Data"
all_files = glob.glob(os.path.join(path, "epadlcs_*.csv"))
output_path = "./Data/trial_data.csv"
fieldnames = ["patient_id", "apoe_result", "abeta_1_42_result", "cdr_global_score", "gds_total", "smoking",
              "physical_activity", "family_dementia_history", "physical_fitness", "mmse_total", "sex",
              "age_years", "years_education", "marital_status", "psqi_total", "stai_40_total_score",
              "height", "weight", "systolic_bp", "diastolic_bp", "4mt_medavante_score",
              "rbans_attention_index","rbans_delayed_memory_index","rbans_immediate_memory_index","rbans_language_index",
              "rbans_visuo_constructional_index","rbans_coding","rbans_digit_span","rbans_figure_copy",
              "rbans_figure_recall","rbans_list_learning","rbans_line_orientation","rbans_list_recall",
              "rbans_list_recognition","rbans_picture_naming","rbans_semantic_fluency","rbans_story_memory",
              "rbans_story_recall", "supermarket_trolley_score"]

df_list = []
for file in all_files:
    df = pd.read_csv(file)
    temp = df[df.columns & fieldnames]
    df_list.append(temp)

df_merged = reduce(lambda left, right: pd.merge(left, right, on='patient_id', how='left'), df_list)
df_merged = df_merged.drop_duplicates(subset='patient_id', keep='first')
df_merged.to_csv(output_path, encoding='utf-8')




