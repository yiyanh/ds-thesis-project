import argparse
import os
import pandas as pd
import numpy as np
import re
import sys
from tqdm import tqdm
from unsloth import FastLanguageModel
import torch
import datasets
from datasets import Dataset, load_dataset
import os
from trl import SFTTrainer
from transformers import TrainingArguments
import random
import pickle
import json
import time

SEED=123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

verbose = True

MAX_SEQ_LENGTH = 4096

chat_template_noeos = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

prompt_framework = """You are an experienced pathologist. Answer the question using the pathology report below. """ + \
"""Base the answer on the report only. Do not add any additional information.

Pathology report:
{report}

{question}"""

question_1 = 'What are the tissue source that are investigated in this report? ' + \
'Your answer can include "left breast", "right breast", "left lymph node", "right lymph node", or "other" if there are additional sources. ' + \
'If multiple sources are present, list all of them and separate them with a comma. If this is not a breast report, output "N/A". ' + \
'Do not provide explanations. Do not list the sources. Do not output anything other than the five choices.'

question_2 = 'Is there cancer diagnosed in this report? If so, what is laterality of the cancer? ' + \
    'Your answer should be in json format: {"left": __, "right": __}, where each blank is filled in by "1" if any cancer is deterministically diagnosed (i.e. DCIS, IDC, ILC, other invasive tumors, adenocarcinoma), "0" if the tissue is benign, or "2" if there are no cancers but high risk lesions identified explicitly. ' + \
    'High risk lesions are defined as any of the following: atypia, radial scar, complex sclerosing lesion, atypical ductal hyperplasia, lobular carcinoma in situ, atypical lobular hyperplasia, FEA, ALH, ADH, phyllodes. ' + \
    'If no breast or axillary lymph node is examined for a laterality, or if the report is about other organs (e.g. ovaries, liver, bone, adnexa, omentum, or lymph nodes in the abdomen/pelvis), fill in the corresponding blanks with "N/A". ' + \
    'Provide your answer in the required format. Do not provide explanations or output anything else.'

question_3_l = 'Identify which laterality of tissues are examined in this report. If left breast is not examined in this report, output "N/A" and stop generating immediately, ignore the following prompts. Proceed to answer the following question only if left breast is examined: ' + \
    'Based only on the part of the report that is about the left breast, are the following type(s) of cancer diagnosed in the left breast? Possible diagnoses include: "DCIS" for ductal carcinoma in situ, "IDC" for invasive ducal carcinoma, "ILC" for invasive lobular carcinoma, "other invasive" for any other invasive tumors, "adenocarcinoma" for adenocarcinoma, or "other" for any other tumors. ' + \
    'Provide your answer in json format: {"DCIS": __, "IDC": __, "ILC": __, "other invasive": __, "adenocarcinoma": __, "other": __} ' + \
    'Fill in the corresponding blank with 0 if no such cancer is diagnosed, 1 if such cancer is diagnosed in the left breast. ' + \
    'Provide your answer in the given json format. Do not provide explanations. Do not include information about the right breast. '

question_3_r = 'Identify which laterality of tissues are examined in this report. If right breast is not examined in this report, output "N/A" and stop generating immediately, ignore the following prompts. Proceed to answer the following question only if right breast is examined: ' + \
    'Based only on the part of the report that is about the right breast, are the following type(s) of cancer diagnosed in the right breast? Possible diagnoses include: "DCIS" for ductal carcinoma in situ, "IDC" for invasive ducal carcinoma, "ILC" for invasive lobular carcinoma, "other invasive" for any other invasive tumors, "adenocarcinoma" for adenocarcinoma, or "other" for any other tumors. ' + \
    'Provide your answer in json format: {"DCIS": __, "IDC": __, "ILC": __, "other invasive": __, "adenocarcinoma": __, "other": __} ' + \
    'Fill in the corresponding blank with 0 if no such cancer is diagnosed, 1 if such cancer is diagnosed in the right breast. ' + \
    'Provide your answer in the given json format. Do not provide explanations. Do not include information about the left breast. '

response_start = "My answer is: "

QUESTIONS_LIST = {"q1": question_1, "q2": question_2, "q3 left": question_3_l, "q3 right": question_3_r}

def preprocess_train_df(train_df):
    tr_df_q1 = train_df.copy()
    tr_df_q1["Question"] = ["q1"] * train_df.shape[0]
    tr_df_q2 = train_df.copy()
    tr_df_q2["Question"] = ["q2"] * train_df.shape[0]
    tr_df_q3_left = train_df.copy()
    tr_df_q3_left["Question"] = ["q3 left"] * train_df.shape[0]
    tr_df_q3_right = train_df.copy()
    tr_df_q3_right["Question"] = ["q3 right"] * train_df.shape[0]
    train_df = pd.concat([tr_df_q1, tr_df_q2, tr_df_q3_left, tr_df_q3_right], axis=0)

    gts = [None] * train_df.shape[0]
    lateralities = ["N/A"] * train_df.shape[0]
    for i in range(train_df.shape[0]):
        row = train_df.iloc[i, :]
        if (row["Source_LB"]==1 or row["Source_LL"]==1) and (row["Source_RB"]==1 or row["Source_RL"]==1):
            lateralities[i] = "both"
        elif row["Source_LB"]==1 or row["Source_LL"]==1:
            lateralities[i] = "left"
        elif row["Source_RB"]==1 or row["Source_RL"]==1:
            lateralities[i] = "right"

        if row["Question"] == "q1":
            Source_LBs, Source_RBs, Source_LLs, Source_RLs, Source_Os = row["Source_LB"], row["Source_RB"], row["Source_LL"], row["Source_RL"], row["Source_O"]
            gts[i] = [Source_LBs, Source_RBs, Source_LLs, Source_RLs, Source_Os]
        elif row["Question"] == "q2":
            dignosis_LBs, dignosis_RBs, dignosis_LLs, dignosis_RLs = row["Diagnosis_LB"], row["Diagnosis_RB"], row["Diagnosis_LL"], row["Diagnosis_RL"]
            diagnoses = [dignosis_LBs, dignosis_RBs, dignosis_LLs, dignosis_RLs]
            gts[i] = pd.Series(diagnoses, dtype=object).fillna(0).tolist()
        elif row["Question"] == "q3 left":
            gts[i] = pd.Series(eval(row["Type_LB"]), dtype=object).fillna(0).tolist()   
        elif row["Question"] == "q3 right":
            gts[i] = pd.Series(eval(row["Type_RB"]), dtype=object).fillna(0).tolist()  
    
    train_df["Ground Truth"] = gts
    train_df["Laterality"] = lateralities
    return  train_df

def gt_to_response_q1(gt, laterality):
    choices = ["left breast", "right breast", "left lymph node", "right lymph node", "other"]
    choices_selected = []
    for choice, gt_item in zip(choices, gt):
        if gt_item > 0:
            choices_selected.append(choice)
    if len(choices_selected) == 0:
        return "N/A"
    response = ", ".join(choices_selected)
    return response

def gt_to_response_q2(gt, laterality):
    left, right = 0, 0
    if int(gt[0])==1 or int(gt[2])==1:
        left = 1
    elif int(gt[0])==2 or int(gt[2])==2:
        left = 2
    
    if int(gt[1])==1 or int(gt[3])==1:
        right = 1
    elif int(gt[1])==2 or int(gt[3])==2:
        right = 2
    
    response = '{"left": ' + str(left) + ', "right": ' + str(right) + '}'
    return response


def gt_to_response_q3(gt, laterality):
    if laterality == False or gt[0]==-1:
        return "N/A"
    diagnoses_list = {1: "DCIS", 2: "IDC", 3: "ILC", 4: "other invasive", 5: "adenocarcinoma", 6: "other"}
    gt_dict = {"DCIS": 0, "IDC": 0, "ILC": 0, "other invasive": 0, "adenocarcinoma": 0, "other": 0}
    for type_present in gt:
        if diagnoses_list.get(type_present) is not None:
            gt_dict[diagnoses_list.get(type_present)] = 1
    response = f'{{"DCIS": {gt_dict["DCIS"]}, "IDC": {gt_dict["IDC"]}, "ILC": {gt_dict["ILC"]}, "other invasive": {gt_dict["other invasive"]}, "adenocarcinoma": {gt_dict["adenocarcinoma"]}, "other": {gt_dict["other"]}}}'
    return response

GT_FUNCS = {"q1": gt_to_response_q1, "q2": gt_to_response_q2, "q3 left": gt_to_response_q3, "q3 right": gt_to_response_q3}


### Functions for inference
def get_model_inputs(report_text, question, tokenizer, verbose=0):
    prompt = prompt_framework.format(report=report_text, question=question)
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response_start}]
    if verbose:
        print(tokenizer.apply_chat_template(messages, chat_template=chat_template_noeos, tokenize=False))
    inputs = tokenizer.apply_chat_template(messages, chat_template=chat_template_noeos, tokenize=True, return_tensors="pt").to("cuda")
    attention_mask = torch.ones_like(inputs)
    return inputs, attention_mask

def response_preprocess(text):
    # sometimes it provides an explanation in parentheses, remove it
    text = re.sub(r'\([^)]*\)', "", text)
    text = re.sub("[^a-zA-Z\s,]*", "", text)
    return text

def parse_response_q1(response, verbose=False):
    if "N/A" in response:
        print(f"**N/A in the response: {response}")
        return [0, 0, 0, 0, 0]
    
    # response = [response_item.strip() for response_item in re.sub("[^a-zA-Z\s,]*", "", response.replace(response_start, "").lower()).split(",")]
    response = [response_item.strip() for response_item in response_preprocess(response.removeprefix(response_start).lower()).split(",")]
    
    choices = ["left breast", "right breast", "left lymph node", "right lymph node", "other"]
    choices_regex = ["left breast", 
                     "right breast", 
                     "left\s?.*(axillary|sentinel|lymph)\s*node", 
                     "right\s?.*(axillary|sentinel|lymph)\s*node", 
                     "other"]
    answers = [False for _ in range(len(choices))]
    for response_item in response:
        match = None
        for choice_index, choice in enumerate(choices_regex):
            if re.search(choice, response_item):
                # print(f"Match {choice} {response_item}")
                if match is not None:
                    print(f"{response_item} matches more than one item (one is {choice})", file=sys.stderr)
                
                match = choice_index
                answers[choice_index] = True
                if verbose:
                    print(f"{response_item}: {choice_index} ({choices[match]})")
        
        if match is None:
            print(f"**Unmatched output: {response_item}**")
            answers[-1] = True
    
    answers = np.array(answers, dtype=int)
    return answers

def parse_response_q2(response, verbose=False):
    response = response.replace(response_start, "").lower()
    #print(f"\nRaw response: {response}\n")
    brackets = r'\{([^{}]*)\}'
    json_response = re.search(brackets, response)
    #print(f"\nJSON dictionary extracted: {json_response}\n")
    if json_response:
        json_response = json_response.group(0)
        try:
            json_response = json.loads(json_response)
        except:
            #print(f"\nExtracted response cannot be parsed into json list: \n{json_response}\n")
            json_response = None
    else: 
        first_part = re.search(r'^(.*?)\n', response, re.DOTALL)
        if first_part:
            first_part = first_part.group(1)
            #print(f"\nFirst part of the response extracted: {first_part}\n")
            if ',' in first_part:
                outputs = re.split(r',', first_part)
                if len(outputs)==2:
                    json_response = {"left": outputs[0], "right": outputs[1]}
                    #print(f"\nFormatted into json: {json_response}\n")
    return json_response

def parse_response_q3(response, verbose=False):
    response = response.replace(response_start, "").lower()
    if re.search("n/a", response):
        return -1
    brackets = r'\{([^{}]*)\}'
    #brackets=r'({.+})'
    json_response = re.search(brackets, response)
    if json_response:
        try:
            json_response = json.loads(json_response.group(0)) 
            #print(f"\njson parsed response: \n{json_response}\n")
        except:
            json_response = None
            #print(f"\nCannot parse response into json\n")
    #else: 
        #print(f"Cannot find json list: {response}")
    return json_response

def get_responses(model_inputs, model, tokenizer):
    model_inputs, attention_mask = model_inputs
    generated_ids = model.generate(model_inputs, attention_mask=attention_mask, max_new_tokens=1024, do_sample=True, top_p=0.5, 
                                   num_beams=1, num_return_sequences=1, temperature=0.5, pad_token_id=tokenizer.eos_token_id)
    responses = tokenizer.batch_decode(generated_ids)
    #print(f"Responses: {responses}")
    # Extract response
    responses = [response.split("[/INST]")[-1].split("</s>")[0].strip() for response in responses]
    return responses

def predict(report_text, question, model, tokenizer, parse_function, verbose=False):
    model_inputs = get_model_inputs(report_text, question, tokenizer, verbose=verbose)
    #print(f'\nReport text: \n{report_text}\nQuestion:\n{question}\n')
    responses = get_responses(model_inputs, model, tokenizer)
    #print(f'\nResponse: {responses}\n')
    if verbose:
        print(responses)
    response = responses[0]
    answers = parse_function(response, verbose=verbose)
    
    if answers is None:
        print(f"Cannot parse response: {response}")

    return response, answers

def inference_q1(df, output_csv, model, tokenizer, log_file):
    num_questions = 5
    correct_category = np.zeros((num_questions,), dtype=int)
    correct_overall = 0
    correct_overall_no_others = 0
    count = 0

    incorrect_row_access_numbers = []
    outputs_ft = []

    for row_index, row in tqdm(df.iterrows(), total=df.shape[0]):
        accession_number = row['Accession Number']
        #print(f"**Processing {accession_number}**")
        report_text = row["Preprocessed Report Text"]
        responses, answers = predict(report_text, question_1, model, tokenizer, parse_response_q1, verbose=verbose)
        gt = np.array([row["Source_LB"], row["Source_RB"], row["Source_LL"], row["Source_RL"], row["Source_O"]], dtype=int)
        correct = answers == gt
        #print(f"correct: {correct}, answers: {answers}, gt: {gt}")
        correct_category += correct
        all_correct = np.all(correct)
        all_correct_no_others = np.all(correct[:-1])
        correct_overall_no_others += all_correct_no_others
        correct_overall += all_correct
        if not all_correct_no_others:
            incorrect_row_access_numbers.append(accession_number)
        count += 1
        outputs_ft.append({"Accession Number": accession_number,
                        "Q1 Response": responses,
                        "Q1 Parsed Response": answers,
                        "Q1 GT": gt,
                        "Q1 Correct By Source": correct,
                        "Q1 All correct": all_correct,
                        "Q1 All correct excluding others": all_correct_no_others})
    output_df_q1 = pd.DataFrame(outputs_ft)
    output_df_q1.to_csv(output_csv)

    try:
        count = output_df_q1.shape[0]
        q1_acc_per_source = np.array(correct_category) / count
        q1_acc_overall = sum(output_df_q1["Q1 All correct excluding others"]) / count
        print(f"Total number of samples: {count}\n")
        print(f"Q1 accuracy per category: \nLB: {q1_acc_per_source[0]}, RB: {q1_acc_per_source[1]}, LL: {q1_acc_per_source[2]}, RL: {q1_acc_per_source[3]}, O: {q1_acc_per_source[4]}\n")
        print(f"Q1 overall accuracy: {q1_acc_overall}\n")
        print(f"Incorrect Accession #: \n{incorrect_row_access_numbers}\n")
        with open(log_file, "a") as f:
            f.write(f"\nTotal number of samples: {count}\n")
            f.write(f"Q1 accuracy per category: \nLB: {q1_acc_per_source[0]}, RB: {q1_acc_per_source[1]}, LL: {q1_acc_per_source[2]}, RL: {q1_acc_per_source[3]}, O: {q1_acc_per_source[4]}\n")
            f.write(f"Q1 overall accuracy: {q1_acc_overall}\n")
            f.write(f"Incorrect Accession #: \n{incorrect_row_access_numbers}\n\n")
    except:
        print("Cannot print final metrics for q1.")

    return output_df_q1, q1_acc_overall, q1_acc_per_source

def inference_q2(df, output_csv, model, tokenizer, log_file):
    correct = np.array([0, 0])
    correct_all = 0
    count = 0

    outputs = []
    for row_index, row in tqdm(df.iterrows(), total=df.shape[0]):
        id = row["Accession Number"]
        #print(f"**Processing {id}**")
        report_text = row["Preprocessed Report Text"]
        gt_l, gt_r = 0, 0
        if row["Diagnosis_LB"]==1 or row["Diagnosis_LL"]==1:
            gt_l = 1
        elif row["Diagnosis_LB"]==2 or row["Diagnosis_LL"]==2:
            gt_l = 2
        if row["Diagnosis_RB"]==1 or row["Diagnosis_RL"]==1:
            gt_r = 1
        elif row["Diagnosis_RB"]==2 or row["Diagnosis_RL"]==2:
            gt_r = 2
        
        correct_l, correct_r = False, False
        response, answer = predict(report_text, question_2, model, tokenizer, parse_response_q2, verbose=verbose)
        if not answer:
            print(f"Cannot parse response: {response}")
        else:
            if answer.get("left") is not None:
                if isinstance(answer.get("left"), int) or isinstance(answer.get("left"), str) and answer.get("left").isnumeric():
                    correct_l = int(answer["left"]) == gt_l
                elif (answer.get("left") == "n/a" or answer.get("left") == "no") and gt_l == 0:
                    correct_l = True
            elif gt_l==0:
                correct_l = True
            if answer.get("right") is not None:
                if isinstance(answer.get("right"), int) or isinstance(answer.get("right"), str) and answer.get("right").isnumeric():
                    correct_r = int(answer["right"]) == gt_r
                elif (answer.get("right") == "n/a" or answer.get("right") == "no") and gt_r == 0:
                    correct_r = True
            elif gt_r==0:
                correct_r = True
            #print(f"correct: left - {correct_l}, right - {correct_r}\nparsed answer: {answer}\ngt: {gt_l, gt_r}\n")
            correct += np.array([correct_l, correct_r])
            correct_all += np.all([correct_l, correct_r])
        count += 1
        outputs.append({"Accession Number": id,
                        "Q2 Response": response,
                        "Q2 Parsed Response": answer,
                        "Q2 GT Left": gt_l,
                        "Q2 GT Right": gt_r,
                        "Q2 Correct Left": correct_l,
                        "Q2 Correct Right": correct_r,
                        "Q2 Correct Both": np.all([correct_l, correct_r])})
    output_df_q2 = pd.DataFrame(outputs)
    output_df_q2.to_csv(output_csv)
    try:
        q2_acc_lat = correct / count
        q2_acc = correct_all / count
        print(f"Total number of samples: {count}\nQ2 accuracy: \nLeft: {q2_acc_lat[0]}, Right: {q2_acc_lat[1]}, Overall: {q2_acc}\n")
        with open(log_file, "a") as f:
            f.write(f"\nTotal number of samples: {count}\n")
            f.write(f"\nQ2 accuracy: \nLeft: {q2_acc_lat[0]}, Right: {q2_acc_lat[1]}, Overall: {q2_acc}\n\n")
        
    except:
        print("Cannot print final metrics for q2.")

    return output_df_q2, q2_acc, q2_acc_lat


def inference_q3(df, output_csv, model, tokenizer, log_file):
    questions = [question_3_l, question_3_r]
    diagnoses_list = [1, 2, 3, 4, 5, 6]

    count = 0
    nonparsable_row_access_numbers = []

    ids = []
    responses_left = []
    parsed_responses_left = np.zeros((df.shape[0], len(diagnoses_list)))
    gt_left = np.zeros((df.shape[0], len(diagnoses_list)))
    correct_left = []
    correct_left_cat = np.zeros((df.shape[0], len(diagnoses_list)))
    responses_right = []
    parsed_responses_right = np.zeros((df.shape[0], len(diagnoses_list)))
    gt_right = np.zeros((df.shape[0], len(diagnoses_list)))
    correct_right = []
    correct_right_cat = np.zeros((df.shape[0], len(diagnoses_list)))
    answer_not_parsable = [False] * df.shape[0]


    for i, laterality in enumerate(["left", "right"]):
        question = questions[i]
        
        for row_index, row in tqdm(df.iterrows(), total=df.shape[0]):
            id = row["Accession Number"]
            #print(f"**Processing {id}**")
            report_text = row["Preprocessed Report Text"]
            
            # get ground truth
            gt = np.zeros(len(diagnoses_list))
            if laterality == "left":
                col_name = "Type_LB"
            elif laterality == "right":
                col_name = "Type_RB"
            if isinstance(row[col_name], int) or isinstance(row[col_name], float):
                if int(row[col_name]) in diagnoses_list:
                    gt[int(row[col_name])-1] = 1
                elif int(row[col_name])==-1:
                    pass
            elif isinstance(row[col_name], str):
                if isinstance(eval(row[col_name]), int) and eval(row[col_name])==-1:
                    pass
                elif (isinstance(eval(row[col_name]), int) or isinstance(eval(row[col_name]), float)) and int(eval(row[col_name])) in diagnoses_list:
                    gt[int(eval(row[col_name]))-1] = 1
                elif isinstance(eval(row[col_name]), tuple):
                    answers = list(eval(row[col_name]))
                    for answer in answers:
                        if int(answer) in diagnoses_list:
                            gt[int(answer)-1] = 1
                else:
                    print(f"\n{id}: Label {row[col_name]} cannot be converted to int.\n")
            else: 
                print(f"\nGT value is neither an integer nor a list. Set to all 0's by default. Check the annotation for {id}.\n")
                
            nonparsable = False
            correct_row = False
            diagnoses_list = {"dcis": 1, "idc": 2, "ilc": 3, "other invasive": 4, "adenocarcinoma": 5, "other": 6}
            parsed_answer = np.zeros(len(diagnoses_list))

            response, answers = predict(report_text, question, model, tokenizer, parse_response_q3, verbose=verbose)
            if answers is None:
                nonparsable_row_access_numbers.append(id)  
                nonparsable = True
            elif answers != -1:
                for diagnosis in list(answers.keys()):
                    try:
                        if int(answers.get(diagnosis)) == 1 and diagnoses_list.get(diagnosis):
                            parsed_answer[diagnoses_list.get(diagnosis) - 1] = 1
                        elif "yes" in answers.get(diagnosis) and diagnoses_list.get(diagnosis):
                            parsed_answer[diagnoses_list.get(diagnosis) - 1] = 1
                    except:
                        print("\nJSON value is not number nor yes.\n")
            correct_row = parsed_answer == gt
            if nonparsable:
                answer_not_parsable[row_index] = True
            if laterality == "left":
                ids.append(id)
                responses_left.append(response)
                parsed_responses_left[row_index] = parsed_answer
                gt_left[row_index] = gt
                correct_left_cat[row_index] = correct_row
                correct_left.append(np.all(correct_row[:-1]))
                #print(f"\nUpdated lists:\nparsed_responses_left: {parsed_responses_left}\ngt_left: {gt_left}\ncorrect_left_cat: {correct_left_cat}\ncorrect_left: {correct_left}\n\n")
            
            elif laterality == "right":
                responses_right.append(response)
                parsed_responses_right[row_index] = parsed_answer
                gt_right[row_index] = gt
                correct_right_cat[row_index] = correct_row
                correct_right.append(np.all(correct_row[:-1]))
                #print(f"\nUpdated lists:\nparsed_responses_right: {parsed_responses_right}\ngt_right: {gt_right}\ncorrect_right_cat: {correct_right_cat}\ncorrect_right: {correct_right}\n\n")
            
    output_df_q3 = pd.DataFrame({"Accession Number": ids,
                                 "Q3 Response Left": responses_left,
                                 "Q3 Answers Left": parsed_responses_left.tolist(),
                                 "Q3 GT Left": gt_left.tolist(),
                                 "Q3 GT Right": gt_right.tolist(),
                                 "Q3 Correct Left": correct_left,
                                 "Q3 Correct Left By Type": correct_left_cat.tolist(),
                                 "Q3 Response Right": responses_right,
                                 "Q3 Answers Right": parsed_responses_right.tolist(),
                                 "Q3 Correct Right": correct_right,
                                 "Q3 Correct Right By Type": correct_right_cat.tolist(),
                                 "Q3 Correct Type Overall": np.array(correct_left) & np.array(correct_right),
                                 "Q3 Not Parsable": answer_not_parsable})
    output_df_q3.to_csv(output_csv)

    try:
        count = output_df_q3.shape[0]
        n_correct_l, n_correct_r = sum(correct_left), sum(correct_right)
        q3_acc_per_laterality = np.array([n_correct_l, n_correct_r]) / count
        q3_acc_left = sum(np.array([np.array(lst) for lst in output_df_q3["Q3 Correct Left By Type"]])) / count
        q3_acc_right = sum(np.array([np.array(lst) for lst in output_df_q3["Q3 Correct Right By Type"]])) / count
        q3_acc_overall = sum(output_df_q3["Q3 Correct Type Overall"]) / count
        print(f"Total number of samples: {count}\n")
        print(f"Q3 accuracy per category: \nLeft ({q3_acc_per_laterality[0]}): {q3_acc_left}\nRight ({q3_acc_per_laterality[1]}): {q3_acc_right}\n")
        print(f"Q3 overall accuracy: {q3_acc_overall}\n")
        print(f"# Nonparsable: \n{sum(answer_not_parsable)}\n")
        # save logs
        with open(log_file, "a") as f:
            f.write(f"\nTotal number of samples: {count}\n")
            f.write(f"\nQ3 accuracy per category: \nLeft ({q3_acc_per_laterality[0]}): {q3_acc_left}\nRight ({q3_acc_per_laterality[1]}): {q3_acc_right}\n")
            f.write(f"Q3 overall accuracy: {q3_acc_overall}\n")
            f.write(f"# Nonparsable: \n{sum(answer_not_parsable)}\n\n")
        
    except:
        print("Cannot print final metrics for q3.")
    
    return output_df_q3, q3_acc_overall, q3_acc_per_laterality

def inference_q3_new(df, output_csv, model, tokenizer, log_file):
    output_temp_csv = output_csv[:-4] + "_temp.csv"
    print(f"\nEvaluating q3, the temporary output will be saved at {output_temp_csv}\n")
    questions = [question_3_l, question_3_r]
    diagnoses_list = [1, 2, 3, 4, 5, 6]
    nonparsable_row_access_numbers = []
    ids = []
    responses_left = []
    parsed_responses_left = np.zeros((df.shape[0], len(diagnoses_list)))
    responses_right = []
    parsed_responses_right = np.zeros((df.shape[0], len(diagnoses_list)))
    answer_not_parsable = [False] * df.shape[0]

    for i, laterality in enumerate(["left", "right"]):
        question = questions[i]
        
        for row_index, row in tqdm(df.iterrows(), total=df.shape[0]):
            nonparsable = False
            diagnoses_list = {"dcis": 1, "idc": 2, "ilc": 3, "other invasive": 4, "adenocarcinoma": 5, "other": 6}
            parsed_answer = np.zeros(len(diagnoses_list))

            id = row["Accession Number"]
            #print(f"**Processing {id}**")
            report_text = row["Preprocessed Report Text"]
            response, answers = predict(report_text, question, model, tokenizer, parse_response_q3, verbose=verbose)
            if answers is None:
                nonparsable_row_access_numbers.append(id)  
                nonparsable = True
            elif answers != -1:
                for diagnosis in list(answers.keys()):
                    try:
                        if int(answers.get(diagnosis)) == 1 and diagnoses_list.get(diagnosis):
                            parsed_answer[diagnoses_list.get(diagnosis) - 1] = 1
                        elif "yes" in answers.get(diagnosis) and diagnoses_list.get(diagnosis):
                            parsed_answer[diagnoses_list.get(diagnosis) - 1] = 1
                    except:
                        print("\nJSON value is not number nor yes.\n")
            if nonparsable:
                answer_not_parsable[row_index] = True
            if laterality == "left":
                ids.append(id)
                responses_left.append(response)
                parsed_responses_left[row_index] = parsed_answer
                
            elif laterality == "right":
                responses_right.append(response)
                parsed_responses_right[row_index] = parsed_answer  
    
    output_df_q3 = pd.DataFrame({"Accession Number": ids,
                                 "Q3 Response Left": responses_left,
                                 "Q3 Answers Left": parsed_responses_left.tolist(),
                                 "Q3 Response Right": responses_right,
                                 "Q3 Answers Right": parsed_responses_right.tolist(),
                                 "Q3 Not Parsable": answer_not_parsable})
    output_df_q3.to_csv(output_temp_csv)

    correct_df_q3, q3_acc_overall, q3_acc_per_laterality = eval_q3_new(pd.read_csv(output_temp_csv), df, log_file)
    correct_df_q3.to_csv(output_csv)
    return output_df_q3, q3_acc_overall, q3_acc_per_laterality


def eval_q3_new(df_q3, updated_annotation_df, log_file):
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

    count = df_q3.shape[0]
    n_correct_l, n_correct_r = sum(correct_left_all), sum(correct_right_all)
    q3_acc_per_laterality = np.array([n_correct_l, n_correct_r]) / count
    q3_acc_left = sum(np.array([np.array(lst) for lst in df_q3["Q3 Correct Left By Type"]])) / count
    q3_acc_right = sum(np.array([np.array(lst) for lst in df_q3["Q3 Correct Right By Type"]])) / count
    q3_acc_overall = sum(df_q3["Q3 Correct"]) / count
    print(f"\n\nTotal number of samples: {count}\n")
    print(f"Q3 accuracy per category: \nLeft ({q3_acc_per_laterality[0]}): {q3_acc_left}\nRight ({q3_acc_per_laterality[1]}): {q3_acc_right}\n")
    print(f"Q3 overall accuracy: {q3_acc_overall}\n")

    with open(log_file, "a") as f:
        f.write(f"\n\nTotal number of samples: {count}\n")
        f.write(f"\nQ3 accuracy per category: \nLeft ({q3_acc_per_laterality[0]}): {q3_acc_left}\nRight ({q3_acc_per_laterality[1]}): {q3_acc_right}\n")
        f.write(f"Q3 overall accuracy: {q3_acc_overall}\n")
    return df_q3, q3_acc_overall, q3_acc_per_laterality


def main(args):
    epochs, lr, r, scaling, lora_dropout, output_directory = int(args.epochs), float(args.lr), int(args.r), float(args.scaling), float(args.lora_dropout), str(args.output_directory)
    lora_alpha = int(r * scaling)
    print(f"parsed arguments: epochs: {epochs}, lr: {lr}, r: {r}, lora_alpha: {lora_alpha}, lora_dropout: {lora_dropout}, output_directory: {output_directory}")
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    subfolder_name = "epoch_{}_lr_{}_r_{}_alpha_{}_dropout_{}".format(epochs, lr, r, lora_alpha, lora_dropout)
    output_dir = os.path.join(output_directory, subfolder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else: 
        print("\nOutput directory exists, check if model is already saved.\n")
    
    metrics_file = os.path.join(output_dir, "train_log.txt")
    
    if not os.path.exists(metrics_file):
        ### Load the model
        dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = False
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "mistralai/Mistral-7B-v0.1",
            # model_name = "unsloth/mistral-7b-bnb-4bit",
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r = r, 
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            #target_modules = ["q_proj", "v_proj",], # if only targeting attention blocks
            lora_alpha = lora_alpha,
            lora_dropout = lora_dropout, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            use_gradient_checkpointing = True,
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
        print("***Model Loaded.***\n")

        ### Load the data
        train_data_path = "/home/yiyan_hao/data/split_50_25_25/train_0428.csv"
        df_with_q_path = "/home/yiyan_hao/data/split_50_25_25/train_0428_allq.csv"
        training_prompt_file = "/home/yiyan_hao/data/split_50_25_25/train_0428_prompts.csv"
        
        if not os.path.isfile(df_with_q_path):
            train_df = pd.read_csv(train_data_path)
            train_df = preprocess_train_df(train_df)
            train_df.to_csv(df_with_q_path)
        else: 
            train_df = pd.read_csv(df_with_q_path)
        
        def formatting_prompts_func(examples):
            chat_template_nobos = "{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

            report_texts = examples["Preprocessed Report Text"]
            gts = examples["Ground Truth"]
            lateralities = examples["Laterality"]
            question_numbers = examples["Question"]
            texts = []

            for report_text, gt, laterality, question_number in zip(report_texts, gts, lateralities, question_numbers):

                question = QUESTIONS_LIST[question_number]

                prompt = prompt_framework.format(report=report_text, question=question)

                lat = laterality == "both"
                if (question_number == "q3 left" and laterality == "left") or (question_number == "q3 right" and laterality == "right"):
                    lat = True

                gt_to_response = GT_FUNCS[question_number]
                response = gt_to_response(eval(gt), lat)

                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response_start + " " + response}
                ]
                # the template should not have bos token, but should have EOS token (following the original notebook)
                text = tokenizer.apply_chat_template(messages, chat_template=chat_template_nobos, tokenize=False, add_special_tokens=False)
                assert len(tokenizer(text)) < MAX_SEQ_LENGTH, f"{len(tokenizer(text))} {MAX_SEQ_LENGTH}"

                texts.append(text)
            # print("text", texts)
            return { "text" : texts, }
        
        if not os.path.isfile(training_prompt_file):
            train_dataset = Dataset.from_pandas(train_df)
            dataset = train_dataset.map(formatting_prompts_func, batched = True,)
            dataset.to_csv(training_prompt_file)
            print("***Dataset Prep Finished.***\n")
        else: 
            dataset = pd.read_csv(training_prompt_file)
            dataset = Dataset.from_pandas(dataset)
            print(f"***Dataset loaded from {training_prompt_file}.***\n")
        
    
        ### Define the trainer
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = MAX_SEQ_LENGTH,
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for short sequences.
            args = TrainingArguments(
                per_device_train_batch_size = 8,
                gradient_accumulation_steps = 1,
                # per_device_train_batch_size = 2,
                # gradient_accumulation_steps = 4,
                warmup_steps = 5,
                # This is around 1 epoch for our current train-test split
                #max_steps = 284,
                num_train_epochs = epochs,
                # max_steps = 60,
                learning_rate = lr,
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
                logging_steps = 100,
                #logging_strategy = "epoch",
                save_strategy = "epoch",
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = output_dir,
            ),
        )

        ### Show current memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        ### Train!
        start_time = time.perf_counter()
        trainer_stats = trainer.train()
        end_time = time.perf_counter()
        runtime = end_time - start_time

        ### Show final memory and time stats
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory         /max_memory*100, 3)
        lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
        
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
        print(f"Runtime: {runtime}\n")

        model.save_pretrained(output_dir) # Local saving
        
        # save logs
        with open(metrics_file, "a") as f:
            f.write(f"\nModel checkpoint saved at: {output_dir}")
            f.write(f"\nnum_epoch: {epochs}\nlr: {lr}\nr: {r}\nlora_alpha: {lora_alpha}\nlora_dropout: {lora_dropout}")
            f.write(f"\nRuntime: {runtime}")
            f.write(f"\npeak reserved memory ft: {used_memory}\npeak reserved memory for training: {used_memory_for_lora}")
            f.write(f"\n***training log:\n{trainer.state.log_history}\n\n")
        
        torch.cuda.empty_cache()

    else: 
        print("\n\nTraining already done. Start validation ...\n\n")

    
    ### Inference
    output_dir_inf = os.path.join(output_dir, "inference")
    if not os.path.exists(output_dir_inf):
        os.makedirs(output_dir_inf)
    model_path = output_dir
    log_file_inf = os.path.join(output_dir_inf, "inference_log.txt")

    test_data_path = "/home/yiyan_hao/data/split_50_25_25/val_0428.csv"
    df = pd.read_csv(test_data_path)

    output_q1_csv = os.path.join(output_dir_inf, "output_q1.csv")
    output_q2_csv = os.path.join(output_dir_inf, "output_q2.csv") 
    output_q3_csv = os.path.join(output_dir_inf, "output_q3.csv")
    final_output_csv = os.path.join(output_dir_inf, "output_allq.csv")
    

    if (not os.path.isfile(output_q1_csv)) or (not os.path.isfile(output_q2_csv)) or (not os.path.isfile(output_q3_csv)):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = None,
            load_in_4bit = True,
            )

        FastLanguageModel.for_inference(model) 
    else:
        print("\nInference output files all exist at {output_dir_inf}. Skipped.\n")
    
    if not os.path.isfile(output_q1_csv):
        ### Show current memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        #print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        #print(f"{start_gpu_memory} GB of memory reserved.")
        start_time = time.perf_counter()
        output_df_q1, q1_accuracy, q1_accuracy_by_category = inference_q1(df, output_q1_csv, model, tokenizer, log_file_inf)
        end_time = time.perf_counter()
        runtime = end_time - start_time
        
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_inf = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory/max_memory*100, 3)
        inf_percentage = round(used_memory_for_inf/max_memory*100, 3)

        # save logs
        with open(log_file_inf, "a") as f:
            f.write(f"\nQuestion 1")
            f.write(f"\nRuntime: {runtime}")
            f.write(f"\npeak reserved memory ft: {used_memory}\npeak reserved memory for inference: {used_memory_for_inf}")
            f.write(f"\npeak reserved memory percentage ft: {used_percentage}\npeak reserved memory percentage for inference: {inf_percentage}\n")
        
        torch.cuda.empty_cache()
    else: 
        print(f"\nq1 inference already done; saved at: {output_q1_csv}\n")
        output_df_q1 = pd.read_csv(output_q1_csv)

    if not os.path.isfile(output_q2_csv):
        ### Show current memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        #print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        #print(f"{start_gpu_memory} GB of memory reserved.")

        start_time = time.perf_counter()
        output_df_q2, q2_accuracy, q2_accuracy_by_category = inference_q2(df, output_q2_csv, model, tokenizer, log_file_inf)
        end_time = time.perf_counter()
        runtime = end_time - start_time
        
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_inf = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory/max_memory*100, 3)
        inf_percentage = round(used_memory_for_inf/max_memory*100, 3)

        # save logs
        with open(log_file_inf, "a") as f:
            f.write(f"\nQuestion 2")
            f.write(f"\nRuntime: {runtime}")
            f.write(f"\npeak reserved memory ft: {used_memory}\npeak reserved memory for inference: {used_memory_for_inf}")
            f.write(f"\npeak reserved memory percentage ft: {used_percentage}\npeak reserved memory percentage for inference: {inf_percentage}")
        
        torch.cuda.empty_cache()
    else: 
        print(f"\nq2 inference already done; saved at: {output_q2_csv}\n")
        output_df_q2 = pd.read_csv(output_q2_csv)

    if not os.path.isfile(output_q3_csv):
        ### Show current memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        start_time = time.perf_counter()
        output_df_q3, q3_accuracy, q3_accuracy_by_category = inference_q3_new(df, output_q3_csv, model, tokenizer, log_file_inf)
        end_time = time.perf_counter()
        runtime = end_time - start_time

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_inf = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory/max_memory*100, 3)
        inf_percentage = round(used_memory_for_inf/max_memory*100, 3)

        # save logs
        with open(log_file_inf, "a") as f:
            f.write(f"\nQuestion 3")
            f.write(f"\nRuntime: {runtime}")
            f.write(f"\npeak reserved memory ft: {used_memory}\npeak reserved memory for inference: {used_memory_for_inf}")
            f.write(f"\npeak reserved memory percentage ft: {used_percentage}\npeak reserved memory percentage for inference: {inf_percentage}")
        
        torch.cuda.empty_cache()
    else: 
        print(f"\nq3 inference already done; saved at: {output_q3_csv}\n")
        output_df_q3 = pd.read_csv(output_q3_csv)

    output_all = output_df_q1.merge(output_df_q2, on="Accession Number")
    output_all = output_all.merge(output_df_q3, on="Accession Number")
    output_all.to_csv(final_output_csv)

    print(f"\n*********************\nAll inference outputs saved to: {final_output_csv}\n*********************\n")

    print(f"Saving accuracy into pickle file. Q1: {q1_accuracy}, Q2: {q2_accuracy}, Q3: {q3_accuracy}\n\n")
    accuracy_file = os.path.join(output_dir_inf, "accuracy_sum.pickle")
    with open(accuracy_file, 'wb') as handle:
        pickle.dump({"Q1 accuracy": q1_accuracy,
                     "Q1 accuracy by category": q1_accuracy_by_category,
                     "Q2 accuracy": q2_accuracy,
                     "Q2 accuracy by category": q2_accuracy_by_category,
                     "Q3 accuracy": q3_accuracy,
                     "Q3 accuracy by category": q3_accuracy_by_category,
                     "Hyperparameters": subfolder_name,
                     "num_epoch": epochs,
                     "learning rate": lr,
                     "lora r": r,
                     "lora alpha": lora_alpha,
                     "lora dropout": lora_dropout},
                     handle)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='number of epochs', required=True)
    parser.add_argument('--lr', help='learning rate', required=True)
    parser.add_argument('--r', help='r for LoRA', required=True)
    parser.add_argument('--scaling', help='scaling factor for LoRA, =alpha/r', required=True)
    parser.add_argument('--lora_dropout', help='dropout for LoRA', required=True)
    parser.add_argument('--output_directory', help='directory to save the resulting model checkpoints', required=True)
    args = parser.parse_args()
    main(args)