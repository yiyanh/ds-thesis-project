import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

def main(args):
    inf_dir = str(args.output_directory)
    print(f"args.output_directory: {args.output_directory}")
    
    log_file = os.path.join(inf_dir, "inference_log.txt")
    
    df_q1 = pd.read_csv(os.path.join(inf_dir, "output_q1.csv"))
    df_q2 = pd.read_csv(os.path.join(inf_dir, "output_q2.csv"))
    df_q3 = pd.read_csv(os.path.join(inf_dir, "output_q3.csv"))
    df_all = df_q1.merge(df_q2, on="Accession Number")
    df_all = df_all.merge(df_q3, on="Accession Number")

    correct_1 = df_all["Q1 All correct excluding others"]
    correct_2 = df_all["Q2 Correct Both"]
    correct_3 = df_all["Q3 Correct"]
    n = df_all.shape[0]

    correct_all = np.array(list(correct_1)) & np.array(list(correct_2)) & np.array(list(correct_3))
    accuracy_all = sum(correct_all) / n
    print(f"All 3 question accuracy (corrected): {accuracy_all}")
    
    with open(log_file, "a") as f:
        f.write(f"Overall accuracy (3 questions all correct, excluding other category): {accuracy_all}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_directory', help='directory to save the resulting model checkpoints', required=True)
    args = parser.parse_args()
    main(args)