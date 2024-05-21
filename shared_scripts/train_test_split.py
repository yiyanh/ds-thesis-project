import pandas as pd
import numpy as np
from tqdm import tqdm
import datasets
from datasets import Dataset


data_path = "/wynton/protected/home/yala/yiyan/breast_path_temp/preprocessed_EK_updated_040724.csv"
df = pd.read_csv(data_path)

original_dataset = Dataset.from_pandas(df)
#for s in range(1000, 5000): 
#    train_test_dataset = original_dataset.train_test_split(test_size=0.5, shuffle=True, seed=s)
#    train_dataset, temp_dataset = train_test_dataset['train'], train_test_dataset['test']
#    val_test_dataset = temp_dataset.train_test_split(test_size=0.5, shuffle=True, seed=s)
#    val_dataset, test_dataset = val_test_dataset['train'], val_test_dataset['test']
#
#    train_df = pd.DataFrame(train_dataset)
#    val_df = pd.DataFrame(val_dataset)
#    test_df = pd.DataFrame(test_dataset)
#
#    val_lb = val_df[val_df["Source_LB"]==1]
#    val_rb = val_df[val_df["Source_RB"]==1]
#    val_ll = val_df[val_df["Source_LL"]==1]
#    val_rl = val_df[val_df["Source_RL"]==1]
#
#    te_lb = test_df[test_df["Source_LB"]==1]
#    te_rb = test_df[test_df["Source_RB"]==1]
#    te_ll = test_df[test_df["Source_LL"]==1]
#    te_rl = test_df[test_df["Source_RL"]==1]
#
#
#    types = np.unique(te_lb["Type_LB"])
#    if len(np.unique(val_lb["Diagnosis_LB"]))==3 and len(np.unique(te_lb["Diagnosis_LB"]))==3 and len(np.unique(val_rb["Diagnosis_RB"]))==3 and len(np.unique(te_rb["Diagnosis_RB"]))==3 and len(np.unique(val_lb["Type_LB"]))==6 and len(np.unique(val_rb["Type_RB"]))==6 and len(np.unique(te_lb["Type_LB"])) == 6 and len(np.unique(te_rb["Type_RB"])) == 6:
#        print(f'found seed: {s}')
#        break

train_test_dataset = original_dataset.train_test_split(test_size=0.5, shuffle=True, seed=126)
train_dataset, temp_dataset = train_test_dataset['train'], train_test_dataset['test']
val_test_dataset = temp_dataset.train_test_split(test_size=0.5, shuffle=True, seed=126)
val_dataset, test_dataset = val_test_dataset['train'], val_test_dataset['test']
train_df = pd.DataFrame(train_dataset)
val_df = pd.DataFrame(val_dataset)
test_df = pd.DataFrame(test_dataset)

train_path = "/wynton/protected/home/yala/yiyan/breast_path_temp/data/split_50_25_25/breast_train_50_25_25.csv"
val_path = "/wynton/protected/home/yala/yiyan/breast_path_temp/data/split_50_25_25/breast_val_50_25_25.csv"
test_path = "/wynton/protected/home/yala/yiyan/breast_path_temp/data/split_50_25_25/breast_test_50_25_25.csv"
train_df.to_csv(train_path)
val_df.to_csv(val_path)
test_df.to_csv(test_path)
print(f"Train test split completed.\nTraining data saved to: {train_path}\nValiation data saved to: {val_path}\nTest data saved to: {test_path}")