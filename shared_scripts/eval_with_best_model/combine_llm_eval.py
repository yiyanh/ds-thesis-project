import pandas as pd

q1_df = pd.read_csv("/mnt/shareddata/yiyan/eval_best_llm/eval_051824/output_q1.csv")
q2_df = pd.read_csv("/mnt/shareddata/yiyan/eval_best_llm/eval_051824/output_q2.csv")
q3_df = pd.read_csv("/mnt/shareddata/yiyan/eval_best_llm/eval_051824/output_q3.csv")
original_df = pd.read_csv("/mnt/shareddata/datasets/consolidated_all_051324.csv")

#def convert_string_to_list(input_string):
#    list_str = input_string.strip('[]')
#    list_str = list_str.replace(',', ' ').split()
#    result_list = [int(num) for num in list_str if num.isdigit()]
#    return result_list
#
#def convert_string_to_list_q3(input_string):
#    list_str = input_string.strip('[]')
#    list_str = list_str.replace(',', ' ').split()
#    result_list = [int(float(num)) for num in list_str]
#    return result_list
#
#source_lb = []
#source_rb = []
#source_ll = []
#source_rl = []
#source_o = []
#q1_lists = [source_lb, source_rb, source_ll, source_rl, source_o]
#cancer_rb = []
#cancer_lb = []
#type_lb_dcis = []
#type_lb_idc = []
#type_lb_ilc = []
#type_lb_oi = []
#type_lb_a = []
#type_rb_dcis = []
#type_rb_idc = []
#type_rb_ilc = []
#type_rb_oi = []
#type_rb_a = []
#q3_lists_l = [type_lb_dcis, type_lb_idc, type_lb_ilc, type_lb_oi, type_lb_a]
#q3_lists_r = [type_rb_dcis, type_rb_idc, type_rb_ilc, type_rb_oi, type_rb_a]
#
#ids = q1_df["Accession Number"]
#for i in range(len(ids)):
#    id = ids.iloc[i]
#    row_q1 = q1_df[q1_df["Accession Number"]==id].iloc[0]
#    row_q2 = q2_df[q2_df["Accession Number"]==id].iloc[0]
#    row_q3 = q3_df[q3_df["Accession Number"]==id].iloc[0]
#
#    q1_ans = row_q1["Q1 Parsed Response"]
#    try:
#        q1_ans = convert_string_to_list(q1_ans)
#    except:
#        print(q1_ans)
#    for i, ls in enumerate(q1_lists):
#        ls.append(q1_ans[i])
#    
#    try:
#        q2_ans = eval(row_q2["Q2 Parsed Response"])
#    except:
#        print(f'Cannot eval: \n{row_q2["Q2 Parsed Response"]}')
#    if q2_ans.get("left") is not None:
#        if isinstance(q2_ans.get("left"), int) or isinstance(q2_ans.get("left"), str) and q2_ans.get("left").isnumeric():
#            cancer_lb.append(int(q2_ans["left"]))
#    else: 
#        cancer_lb.append(0)
#    if q2_ans.get("right") is not None:
#        if isinstance(q2_ans.get("right"), int) or isinstance(q2_ans.get("right"), str) and q2_ans.get("right").isnumeric():
#            cancer_rb.append(int(q2_ans["right"]))
#    else: 
#        cancer_rb.append(0)
#    
#
#    q3_ans_l = row_q3["Q3 Answers Left"]
#    q3_ans_l = convert_string_to_list_q3(q3_ans_l)
#    for i, ls in enumerate(q3_lists_l):
#        ls.append(q3_ans_l[i])
#
#    q3_ans_r = row_q3["Q3 Answers Right"]
#    q3_ans_r = convert_string_to_list_q3(q3_ans_r)
#    for i, ls in enumerate(q3_lists_r):
#        ls.append(q3_ans_r[i])
#    

#df_new = pd.DataFrame({"Accession Number": ids,
#                       "Source_LB": source_lb,
#                       "Source_RB": source_rb,
#                       "Source_LL": source_ll,
#                       "Source_RL": source_rl,
#                       "Source_O": source_o,
#                       "Cancer_LB": cancer_lb,
#                       "Cancer_RB": cancer_rb,
#                       "Type_L_DCIS": type_lb_dcis,
#                       "Type_L_IDC": type_lb_idc,
#                       "Type_L_ILC": type_lb_ilc,
#                       "Type_L_OI": type_lb_oi,
#                       "Type_L_A": type_lb_a,
#                       "Type_R_DCIS": type_rb_dcis,
#                       "Type_R_IDC": type_rb_idc,
#                       "Type_R_ILC": type_rb_ilc,
#                       "Type_R_OI": type_rb_oi,
#                       "Type_R_A": type_rb_a})
#df_new.to_csv("/mnt/shareddata/yiyan/eval_best_llm/eval_051824/inference_all_results.csv")
#
#df_all = original_df.merge(df_new, on="Accession Number")
#df_all.to_csv("/mnt/shareddata/yiyan/eval_best_llm/eval_051824/inference.csv")

df_new = pd.read_csv("/mnt/shareddata/yiyan/eval_best_llm/eval_051824/inference_all_results.csv")
for idx, row in df_new.iterrows():
    if row["Source_LB"]==0:
        df_new.at[idx, "Cancer_LB"]=0
    if row["Source_RB"]==0:
        df_new.at[idx, "Cancer_RB"]=0
    
    if row["Cancer_LB"]==0:
        df_new.at[idx, "Type_L_DCIS"]=0
        df_new.at[idx, "Type_L_IDC"]=0
        df_new.at[idx, "Type_L_ILC"]=0
        df_new.at[idx, "Type_L_OI"]=0
        df_new.at[idx, "Type_L_A"]=0
    if row["Cancer_RB"]==0:
        df_new.at[idx, "Type_R_DCIS"]=0
        df_new.at[idx, "Type_R_IDC"]=0
        df_new.at[idx, "Type_R_ILC"]=0
        df_new.at[idx, "Type_R_OI"]=0
        df_new.at[idx, "Type_R_A"]=0

df_new.to_csv("/mnt/shareddata/yiyan/eval_best_llm/eval_051824/inference_all_results_corrected.csv")

df_all = original_df.merge(df_new, on="Accession Number")
df_all.to_csv("/mnt/shareddata/yiyan/eval_best_llm/eval_051824/inference_corrected.csv")
