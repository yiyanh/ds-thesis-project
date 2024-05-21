import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

def main(args):
    #inf_dir = "/mnt/shareddata/yiyan/grid_search/gs_042824_new/epoch_3_lr_0.0001_r_32_alpha_16_dropout_0.05/inference"
    inf_dir = str(args.output_directory)
    print(f"args.output_directory: {args.output_directory}")
    
    log_file = os.path.join(inf_dir, "inference_log.txt")

    df_q1 = pd.read_csv(os.path.join(inf_dir, "output_q1.csv"))
    df_q2 = pd.read_csv(os.path.join(inf_dir, "output_q2.csv"))
    df_q3 = pd.read_csv(os.path.join(inf_dir, "output_q3_temp.csv"))
    #df_q3.to_csv(os.path.join(inf_dir, "output_q3_wrong.csv"))
    #df_all = pd.read_csv(os.path.join(inf_dir, "output_allq.csv"))
    #df_all.to_csv(os.path.join(inf_dir, "output_allq_wrong.csv"))

    print(f'Question 1 df columns: {df_q1.columns}\nQuestion 2 df columns: {df_q2.columns}\nQuestion 3 df columns: {df_q3.columns}\n')
    print(f'Output data dimensions: Q1: {df_q1.shape}, Q2: {df_q2.shape}, Q3: {df_q3.shape},')

    updated_annotation_df = pd.read_csv("/home/yiyan_hao/data/annotation/preprocessed_updated-separate-label.csv")

    correct_left_by_cat = np.zeros((df_q3.shape[0], 6))
    correct_right_by_cat = np.zeros((df_q3.shape[0], 6))
    correct_left_all = []
    correct_right_all = []
    correct_overall = []
    diagnoses_list = [1, 2, 3, 4, 5, 6]
    gt_left = np.zeros((df_q3.shape[0], 6))
    gt_right = np.zeros((df_q3.shape[0], 6))

    for row_index, row in tqdm(df_q3.iterrows(), total=df_q3.shape[0]):
        id = row["Accession Number"]
        try:
            left_annotation = updated_annotation_df[updated_annotation_df["Accession Number"]==id]["Type_LB"].iloc[0]
            right_annotation = updated_annotation_df[updated_annotation_df["Accession Number"]==id]["Type_RB"].iloc[0]
        except:
            print(f"\nCase {id} is not included in updated annotatons.\n")
            left_annotation = -1
            right_annotation = -1

        gt_l = np.zeros(len(diagnoses_list))
        gt_r = np.zeros(len(diagnoses_list))

        if isinstance(left_annotation, int) or isinstance(left_annotation, float):
            if int(left_annotation) in diagnoses_list:
                gt_l[int(left_annotation)-1] = 1
            elif left_annotation==-1:
                pass
        elif isinstance(left_annotation, str):
            if isinstance(eval(left_annotation), int) and eval(left_annotation)==-1:
                pass
            elif isinstance(eval(left_annotation), int) or isinstance(eval(left_annotation), float) and eval(left_annotation) in diagnoses_list:
                gt_l[int(eval(left_annotation))-1] = 1
            elif isinstance(eval(left_annotation), tuple):
                answers = list(eval(left_annotation))
                for answer in answers:
                    if int(answer) in diagnoses_list:
                        gt_l[int(answer)-1] = 1
            else:
                print(f"\n{id}: Label {left_annotation} for LB cannot be converted to int.\n")
            
        if isinstance(right_annotation, int) or isinstance(right_annotation, float):
            if int(right_annotation) in diagnoses_list:
                gt_r[int(right_annotation)-1] = 1
            elif right_annotation==-1:
                pass
        elif isinstance(right_annotation, str):
            if isinstance(eval(right_annotation), int) and eval(right_annotation)==-1:
                pass
            elif isinstance(eval(right_annotation), int) or isinstance(eval(right_annotation), float) and int(eval(right_annotation)) in diagnoses_list:
                gt_r[int(eval(right_annotation))-1] = 1
            elif isinstance(eval(right_annotation), tuple):
                answers = list(eval(right_annotation))
                for answer in answers:
                    if int(answer) in diagnoses_list:
                        gt_r[int(answer)-1] = 1
            else:
                print(f"\n{id}: Label {right_annotation} for RB cannot be converted to int.\n")

        gt_left[row_index] = gt_l
        gt_right[row_index] = gt_r

        parsed_answer_l = np.array(eval(row["Q3 Answers Left"]))
        parsed_answer_r = np.array(eval(row["Q3 Answers Right"]))

        correct_l = gt_l == parsed_answer_l
        correct_r = gt_r == parsed_answer_r
        correct_all_l = np.all(correct_l[:-1])
        correct_all_r = np.all(correct_r[:-1])
        correct_all = np.all([correct_l, correct_r])

        correct_left_by_cat[row_index] = correct_l
        correct_right_by_cat[row_index] = correct_r
        correct_left_all.append(correct_all_l)
        correct_right_all.append(correct_all_r)
        correct_overall.append(correct_all)

    df_q3["Q3 GT Left new"] = gt_left.tolist()
    df_q3["Q3 GT Right new"] = gt_right.tolist()
    df_q3["Q3 Correct Left By Type"] = correct_left_by_cat.tolist()
    df_q3["Q3 Correct Right By Type"] = correct_right_by_cat.tolist()
    df_q3["Q3 Correct Left"] = correct_left_all
    df_q3["Q3 Correct Right"] = correct_right_all
    df_q3["Q3 Correct"] = correct_overall

    df_q3.to_csv(os.path.join(inf_dir, "output_q3.csv"))
    df_all = df_q1.merge(df_q2, on="Accession Number")
    df_all = df_all.merge(df_q3, on="Accession Number")
    df_all.to_csv(os.path.join(inf_dir, "output_allq.csv"))

    count = df_q3.shape[0]
    n_correct_l, n_correct_r = sum(correct_left_all), sum(correct_right_all)
    q3_acc_per_laterality = np.array([n_correct_l, n_correct_r]) / count
    q3_acc_left = sum(np.array([np.array(lst) for lst in df_q3["Q3 Correct Left By Type"]])) / count
    q3_acc_right = sum(np.array([np.array(lst) for lst in df_q3["Q3 Correct Right By Type"]])) / count
    q3_acc_overall = sum(df_q3["Q3 Correct"]) / count
    print(f"\nCorrected results for: {inf_dir}\n")
    print(f"\nTotal number of samples: {count}\n")
    print(f"Q3 accuracy per category: \nLeft ({q3_acc_per_laterality[0]}): {q3_acc_left}\nRight ({q3_acc_per_laterality[1]}): {q3_acc_right}\n")
    print(f"Q3 overall accuracy: {q3_acc_overall}\n")

    with open(log_file, "a") as f:
        f.write(f"\nCorrected results for: {inf_dir}\n")
        f.write(f"\nTotal number of samples: {count}\n")
        f.write(f"\nQ3 accuracy per category: \nLeft ({q3_acc_per_laterality[0]}): {q3_acc_left}\nRight ({q3_acc_per_laterality[1]}): {q3_acc_right}\n")
        f.write(f"Q3 overall accuracy: {q3_acc_overall}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_directory', help='directory to save the resulting model checkpoints', required=True)
    args = parser.parse_args()
    main(args)
