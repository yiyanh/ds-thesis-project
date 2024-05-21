import pandas as pd
import numpy as np
import re
import math
from tqdm import tqdm


def extract_text_fn(text):
    result = text
    pattern = r'(?:(?<=PATHOLOGIC DIAGNOSIS)|(?<=FINAL DIAGNOSIS)|(?<=CYTOLOGIC DIAGNOSIS)|(?<=FINAL PATHOLOGIC DIAGNOSIS))([\s\S]*)'
    match = re.search(pattern, result, re.DOTALL)
    if match:
        result = match.group(0)

    results = [result]
    matches = []

    if 'Clinical History' in result:
        pattern = r'[\s\S]*(?=\s*Clinical History)'
        match1 = re.search(pattern, result, re.DOTALL)
        matches.append(match1)
    elif 'Past Medical History' in result:
        pattern = r'[\s\S]*(?=\s*Past Medical History)'
        match2 = re.search(pattern, result, re.DOTALL)
        matches.append(match2)
    elif 'Gross Description' in result:
        pattern = r'[\s\S]*(?=\s*Gross Description)'
        match3 = re.search(pattern, result, re.DOTALL)
        matches.append(match3)
    elif 'REFERENCE RANGE' in result:
        pattern = r'[\s\S]*(?=\s*REFERENCE RANGE)'
        match4 = re.search(pattern, result, re.DOTALL)
        matches.append(match4)

    for m in matches:
        if m:
            results.append(m.group(0))
    
    return min(results, key=len)

def extract_text_sp(text):
    result = text
    pattern = r'(?:(?<=PATHOLOGIC DIAGNOSIS)|(?<=FINAL DIAGNOSIS)|(?<=CYTOLOGIC DIAGNOSIS)|(?<=FINAL PATHOLOGIC DIAGNOSIS))([\s\S]*)'
    match = re.search(pattern, result, re.DOTALL)
    if match:
        result = match.group(0)

    results = [result]
    matches = []

    if 'Methodology' in result:
        pattern = r'[\s\S]*(?=\s*Methodology)'
        match1 = re.search(pattern, result, re.DOTALL)
        matches.append(match1)

    if 'Specimen\(s\) Received' in result:
        pattern = r'[\s\S]*(?=\s*Specimen\(s\) Received)'
        match2 = re.search(pattern, result, re.DOTALL)
        matches.append(match2)

    if 'SPECIMEN\(S\) RECEIVED' in result:
        pattern = r'[\s\S]*(?=\s*SPECIMEN\(S\) RECEIVED)'
        match3 = re.search(pattern, result, re.DOTALL)
        matches.append(match3)

    if 'Clinical History' in result:
        pattern = r'[\s\S]*(?=\s*Clinical History)'
        match4 = re.search(pattern, result, re.DOTALL)
        matches.append(match4)

    if 'Gross Description' in result:
        pattern = r'[\s\S]*(?=\s*Gross Description)'
        match5 = re.search(pattern, result, re.DOTALL)
        matches.append(match5)
    
    for m in matches:
        if m:
            results.append(m.group(0))
    
    return min(results, key=len)

def preprocess(df, filter_columns=True):
    # If this case should be included
    df = df[df["Include case"] == 1]
    df = df[df["Accession Number"].isna()==False]
    
    # Exclude other exam types
    #df = df[df["Exam Description"] != "HEMATOPATHOLOGY"]
    df = df[df["Exam Description"] != "HER2 GENE AMPLIFICATION TESTING BY FISH"]
    df = df[df["Exam Description"] != "MOLECULAR PATHOLOGY"]

    # Exclude rows with comments from the annotator
    #df = df[df["Comment"].isna()] # 24674 samples --> 24532 samples

    # Only keep specimen types of interest: FNA / core / surgery / punch biology
    #df = df[df['Specimen_Type'].notna()]

    #df = df[df["Source_NSB"] != 1] # this has been accounted for by include case
    
    # Exclude the reports without a breast / lymph node source (i.e. with only other source)
    df = df[(df.Source_LB != 0) | (df.Source_RB != 0) | (df.Source_LL != 0) | (df.Source_RL != 0)]
    
    df = df.reset_index()
    
    for index, row in df.iterrows():
        # Extract the text 
        text = row["Report Text"]  

        if row["Exam Description"] == "FINE NEEDLE ASPIRATION (DELIVER SPECIMEN TO PATHOLOGY)" or row["Exam Description"] == "HEMATOPATHOLOGY":
            text = extract_text_fn(text)
        
        elif row["Exam Description"] == "SURGICAL PATHOLOGY":
            text = extract_text_sp(text)

        df.at[index, "Preprocessed Report Text"] = text.strip(":\n\t ")

    # Diagnosis_LB	Diagnosis_RB	Diagnosis_LL	Diagnosis_RL can be NaN if they are not present in the annotation.
    df["Type_LB"] = df["Type_LB"].fillna(-1)
    df["Type_RB"] = df["Type_RB"].fillna(-1)

    if filter_columns:
        columns_to_keep = ["Accession Number", "Exam Description", "Preprocessed Report Text", "Source_LB", "Source_RB", "Source_LL", "Source_RL", "Source_O", "Diagnosis_LB", "Diagnosis_RB", "Diagnosis_LL", "Diagnosis_RL", "Type_LB", "Type_RB"]
    
        df = df[columns_to_keep]
    
    return df

#def get_answer_q1(row): # returns 0 if no source, an integer if one source, a list if multiple sources
#    answer = []
#
#    if row.loc["Source_LB"] == 1:
#        answer.append(1)
#    if row.loc["Source_RB"] == 1:
#        answer.append(2)
#    if row.loc["Source_LL"] == 1:
#        answer.append(3)
#    if row.loc["Source_RL"] == 1:
#        answer.append(4)
#    if row.loc["Source_O"] == 1:
#        answer.append(5)
#    
#    if len(answer) == 0:
#        return 0
#    elif len(answer) == 1:
#        return answer[0]
#    else:
#        return answer
#
#def get_answer_q2(row): # returns 1 if there is cancer, 0 if not
#    return int(np.nansum(row[["Diagnosis_LB", "Diagnosis_RB", "Diagnosis_LL", "Diagnosis_RL"]]) > 0)
#
#def get_answer_q3(row): # for each subset, returns -1 if no cancer, returns the index if there is cancer
#    tissues = ["Type_LB", "Type_RB"]
#    answer = []
#    # types = ["ductal carcinoma in situ (DCIS)", "invasive ducal carcinoma (IDC)", "invasive lobular carcinoma (ILC)", "other invasive", "adenocarcinoma", "other"]
#    for i in range(2):
#        if not math.isnan(row.loc[tissues[i]]):
#            answer.append(int(row.loc[tissues[i]]))
#        else:
#            # no cancer
#            answer.append(-1)
#    
#    return answer

#def save_preprocessed_csv(preprocessed_df, destination_path):
#    data = []
#    for idx, row in tqdm(preprocessed_df.iterrows(), total=preprocessed_df.shape[0]):
#        q1_answer = get_answer_q1(row)
#        q2_answer = get_answer_q1(row)
#        q3_answer = get_answer_q1(row)
#        
#        data.append({
#            "Q1": q1_answer,
#            "Q2": q2_answer,
#            "Q3": q3_answer
#        })
#
#    final_df = pd.concat([preprocessed_df, pd.DataFrame(data)], axis=1)
#    final_df.to_csv(destination_path)


#data_path = "/home/yiyan_hao/data/annotation/EK_updated_040724.csv"
#destination_path = "/home/yiyan_hao/data#/annotation/preprocessed_EK_updated_040724.csv"
#data_path = "/home/yiyan_hao/data/annotation/breast_with_comments.csv"
#destination_path = "/home/yiyan_hao/data/annotation/preprocessed_breast_with_comments.csv"
data_path = "/home/yiyan_hao/data/annotation/updated-separate-label.csv"
destination_path = "/home/yiyan_hao/data/annotation/preprocessed_updated-separate-label.csv"
#data_path = "/mnt/shareddata/yiyan/breast_annotation_0427_v0.csv"
#destination_path = "/home/yiyan_hao/data/annotation/preprocessed_0427_v0.csv"


df = pd.read_csv(data_path)
preprocessed_df = preprocess(df)

preprocessed_df.to_csv(destination_path)
