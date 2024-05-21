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
    df = df[df["Report Text"].isna()==False]
    df = df.reset_index()
    
    for index, row in df.iterrows():
        # Extract the text 
        text = row["Report Text"]
        if row["Exam Description"] == "FINE NEEDLE ASPIRATION (DELIVER SPECIMEN TO PATHOLOGY)" or row["Exam Description"] == "HEMATOPATHOLOGY":
            text = extract_text_fn(text)
        
        elif row["Exam Description"] == "SURGICAL PATHOLOGY":
            text = extract_text_sp(text)

        df.at[index, "Preprocessed Report Text"] = text.strip(":\n\t ")

    if filter_columns:
        columns_to_keep = ["Accession Number", "Exam Description", "Preprocessed Report Text"]
    
        df = df[columns_to_keep]
    
    return df



data_path = "/mnt/shareddata/datasets/consolidated_all_051324.csv"
destination_path = "/home/yiyan_hao/data/preprocessed_consolidated_all_051324.csv"

df = pd.read_csv(data_path)
preprocessed_df = preprocess(df)

preprocessed_df.to_csv(destination_path)
