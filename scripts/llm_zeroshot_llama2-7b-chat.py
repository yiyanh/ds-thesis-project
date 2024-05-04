import pandas as pd
import numpy as np
import re
import sys
from tqdm import tqdm

from datasets import Dataset
import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import random
import argparse
import time

SEED=123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MAX_SEQ_LENGTH = 4096
verbose = 0

chat_template_noeos = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' }}{% endif %}{% endfor %}"

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
'Provide your answer in two steps. ' + \
'First, provide reasoning for whether there is cancer (i.e. DCIS, IDC, ILC, other invasive tumors, adenocarcinoma or other cancer types) or high risk lesions (i.e. atypia, radial scar, complex sclerosing lesion, atypical ductal hyperplasia, lobular carcinoma in situ, atypical lobular hyperplasia, FEA, ALH, ADH) diagnosed, and which laterality (i.e. left or right) it belongs to. ' + \
'Then output a final answer in json format: {"left": __, "right": __}, where each blank is filled in by "1" if any cancer is diagnosed, "0" if the tissue is benign, or "2" if there are no cancers but high risk lesions. ' + \
'If no breast or axillary lymph node is examined for a laterality, or if the report is about other organs (e.g. ovaries, liver, bone, adnexa, omentum, or lymph nodes in the abdomen/pelvis), fill in the corresponding blanks with "N/A". ' + \
'Provide your answer in the required format. Do not provide additional explanations.'

question_3_l = 'Provide your answer in three steps. ' + \
'1) identify which laterality (left / right) of tissues are examined in this report. If left breast is not examined in this report, skip the remaining steps, output "N/A" and stop generating immediately. Proceed to the next step only if left breast is examined. ' + \
'2) extract the part of the report that is about the left breast, removing any information about right breast, any lymph nodes, or any other tissues. ' + \
'3) based on your extracted report from the second step, answer this question: Is cancer diagnosed in the left breast? Possible cancer diagnoses include: "DCIS" for ductal carcinoma in situ, "IDC" for invasive ducal carcinoma, "ILC" for invasive lobular carcinoma, "other invasive" for any other invasive tumors, "adenocarcinoma" for adenocarcinoma, or "other" for any other tumors. ' + \
'Provide your final answer following this format: "Cancer diagnosis for the left breast: Benign / DCIS / IDC / ILC / other invasive / adenocarcinoma / other" ' + \
'Do not provide further explanations. '

question_3_r = 'Provide your answer in three steps. ' + \
'1) identify which laterality (left / right) of tissues are examined in this report. If right breast is not examined in this report, skip the remaining steps, output "N/A" and stop generating immediately. Proceed to the next step only if right breast is examined. ' + \
'2) extract the part of the report that is about the right breast, removing any information about left breast, any lymph nodes, or any other tissues. ' + \
'3) based on your extracted report from the second step, answer this question: Is cancer diagnosed in the right breast? Possible cancer diagnoses include: "DCIS" for ductal carcinoma in situ, "IDC" for invasive ducal carcinoma, "ILC" for invasive lobular carcinoma, "other invasive" for any other invasive tumors, "adenocarcinoma" for adenocarcinoma, or "other" for any other tumors. ' + \
'Provide your final answer following this format: "Cancer diagnosis for the right breast: Benign / DCIS / IDC / ILC / other invasive / adenocarcinoma / other" ' + \
'Do not provide further explanations. '

response_start = "My answer is: "

def get_model_inputs(report_text, question, tokenizer, verbose=0):
    prompt = prompt_framework.format(report=report_text, question=question)
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response_start}]
    inputs = tokenizer.apply_chat_template(messages, chat_template=chat_template_noeos, tokenize=True, return_tensors="pt").to("cuda")
    attention_mask = torch.ones_like(inputs)
    #print(f"\nModel inputs token: \n{inputs}\nMask: {attention_mask}\n")
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
            # if none of the choices get matched in the response, parse the response as other.
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
            json_response = str(json_response).replace("'", '"')
            json_response = json.loads(json_response)
        except:
            #print(f"\nExtracted response cannot be parsed into json list: \n{json_response}\n")
            json_response = None
    return json_response

def parse_response_q3(response, laterality, verbose=False):
    """Parses model response into 0 if benign or if breast is not examined, and a string describing cancer type if there is cancer.
    If format is wrong, and cannot match n/a in the response, returns None. """
    response = response.replace(response_start, "").lower()
    print(f"\nRaw response: \n{response}")
    if laterality == "left":
        pattern = r"(3[.)]\s*cancer diagnosis for the left breast:)(.*)"
    elif laterality == "right":
        pattern = r"(3[.)]\s*cancer diagnosis for the right breast:)(.*)"

    final_response = re.search(pattern, response)
    if final_response:
        final_response = final_response.group(2).strip()
        print(f"\nParsed final response: {final_response}\n")
        if re.search("benign", final_response):
            final_response=0
    else:
        print(f"\nCannot find the third bullet point in the response.\n")
        if re.search("n/a", response):
            final_response = 0 # when the laterality is not examined or the report is not about breasts
    return final_response

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
    print(f"\nParsed answers: \n{answers}\n")
    
    if answers is None:
        print(f"Cannot parse response: {response}")

    return response, answers

def predict_q3(report_text, question, model, tokenizer, parse_function, laterality, verbose=False):
    model_inputs = get_model_inputs(report_text, question, tokenizer, verbose=verbose)
    #print(f'\nReport text: \n{report_text}\nQuestion:\n{question}\n')
    responses = get_responses(model_inputs, model, tokenizer)
    #print(f'\nResponse: {responses}\n')
    if verbose:
        print(responses)
    response = responses[0]
    answers = parse_function(response, laterality, verbose=verbose)
    print(f"\nParsed answers: \n{answers}\n")
    
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
        print(f"**Processing {accession_number}**")
        report_text = row["Preprocessed Report Text"]
        responses, answers = predict(report_text, question_1, model, tokenizer, parse_response_q1, verbose=verbose)
        gt = np.array([row["Source_LB"], row["Source_RB"], row["Source_LL"], row["Source_RL"], row["Source_O"]], dtype=int)
        correct = answers == gt
        print(f"correct: {correct}, answers: {answers}, gt: {gt}")
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
        print(f"Incorrect Accession #: \n{incorrect_row_access_numbers}\n-------------------------------------------------------\n-------------------------------------------------------\n-------------------------------------------------------\n")
        with open(log_file, "a") as f:
            f.write(f"\nTotal number of samples: {count}\n")
            f.write(f"Q1 accuracy per category: \nLB: {q1_acc_per_source[0]}, RB: {q1_acc_per_source[1]}, LL: {q1_acc_per_source[2]}, RL: {q1_acc_per_source[3]}, O: {q1_acc_per_source[4]}\n")
            f.write(f"Q1 overall accuracy: {q1_acc_overall}\n")
            f.write(f"Incorrect Accession #: \n{incorrect_row_access_numbers}\n\n")
    except:
        print("Cannot print final metrics for q1.")

    return output_df_q1

def inference_q2(df, output_csv, model, tokenizer, log_file):
    correct = np.array([0, 0])
    correct_all = 0
    count = 0

    outputs = []
    for row_index, row in tqdm(df.iterrows(), total=df.shape[0]):
        id = row["Accession Number"]
        print(f"**Processing {id}**")
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
        print(f"Total number of samples: {count}\nQ2 accuracy: \nLeft: {q2_acc_lat[0]}, Right: {q2_acc_lat[1]}, Overall: {q2_acc}\n-------------------------------------------------------\n-------------------------------------------------------\n-------------------------------------------------------\n")
        with open(log_file, "a") as f:
            f.write(f"\nTotal number of samples: {count}\n")
            f.write(f"\nQ2 accuracy: \nLeft: {q2_acc_lat[0]}, Right: {q2_acc_lat[1]}, Overall: {q2_acc}\n\n")
        
    except:
        print("Cannot print final metrics for q2.")

    return output_df_q2


def inference_q3(df, output_csv, model, tokenizer, log_file):
    diagnoses_regex = ["dcis", "idc", "ilc", "other invasive", "adenocarcinoma", r"other(?!\s*invasive)"]
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
            print(f"**Processing {id}**")
            report_text = row["Preprocessed Report Text"]
            
            # get ground truth
            gt = np.zeros(len(diagnoses_list))
            if laterality == "left":
                col_name = "Type_LB"
            elif laterality == "right":
                col_name = "Type_RB"
            if isinstance(row[col_name], int) or isinstance(row[col_name], float):
                if int(row[col_name]) > 0:
                    gt[int(row[col_name])-1] = 1
            elif isinstance(row[col_name], list):
                print("\nMultiple diagnoses in GT.\n")
                lst = list(map(int, row[col_name]))
                for diagnosis_idx, diagnosis in enumerate(diagnoses_list):
                    if diagnosis in lst: 
                        gt[diagnosis_idx] = 1
            else: 
                print(f"\nGT value is neither an integer nor a list. Set to all 0's by default. Check the annotation for {id}.\n")
                
            nonparsable = False
            correct_row = False
            diagnoses_list = {"dcis": 1, "idc": 2, "ilc": 3, "other invasive": 4, "adenocarcinoma": 5, "other": 6}
            parsed_answer = np.zeros(len(diagnoses_list))

            response, answers = predict_q3(report_text, question, model, tokenizer, parse_response_q3, laterality, verbose=verbose)
            if answers is None:
                print(f"Cannot parse response: {response}")
                nonparsable_row_access_numbers.append(id)  
                nonparsable = True
                correct_row = np.array([False] * len(diagnoses_list))
            else:
                if answers == 0: # model says laterality is benign or breast is not examined
                    parsed_answer = np.zeros(len(diagnoses_list))
                elif re.search("n/a", answers):
                    parsed_answer = np.zeros(len(diagnoses_list))
                else: 
                    for choice_index, choice_regex in enumerate(diagnoses_regex):
                        if re.search(choice_regex, answers):
                            parsed_answer[choice_index] = 1

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
                            "Q3 Response Right": responses_right,
                            "Q3 Answers Right": parsed_responses_right.tolist(),
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
        print(f"# Nonparsable: \n{sum(answer_not_parsable)}\n-------------------------------------------------------\n-------------------------------------------------------\n-------------------------------------------------------\n\n")
        # save logs
        with open(log_file, "a") as f:
            f.write(f"\nTotal number of samples: {count}\n")
            f.write(f"\nQ3 accuracy per category: \nLeft ({q3_acc_per_laterality[0]}): {q3_acc_left}\nRight ({q3_acc_per_laterality[1]}): {q3_acc_right}\n")
            f.write(f"Q3 overall accuracy: {q3_acc_overall}\n")
            f.write(f"# Nonparsable: \n{sum(answer_not_parsable)}\n\n")
        
    except:
        print("Cannot print final metrics for q3.")
    
    return output_df_q3


def main(args):
    output_directory = str(args.output_directory)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    test_data_path = "/home/yiyan_hao/data/split_50_25_25/all_test_50_25_25.csv"
    df = pd.read_csv(test_data_path)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    output_q1_csv = os.path.join(output_directory, "output_q1.csv")
    output_q2_csv = os.path.join(output_directory, "output_q2.csv") 
    output_q3_csv = os.path.join(output_directory, "output_q3.csv")
    final_output_csv = os.path.join(output_directory, "output_allq.csv")
    log_file = os.path.join(output_directory, "inference_log.txt")

    with open(log_file, "a") as f:
        f.write(f"Inference log\n")
        f.write(f"Model: \n{model}\n\n")
        f.write(f"Path to save output results, including output csv per question and inference log txt (this file): \n{output_directory}\n\n")
        f.write(f"Begin inference on 3 questions...\n\n")
    
    if not os.path.isfile(output_q1_csv):
        ### Show current memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")
        
        start_time = time.perf_counter()
        output_df_q1 = inference_q1(df, output_q1_csv, model, tokenizer, log_file)
        end_time = time.perf_counter()
        runtime = end_time - start_time

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_inf = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory/max_memory*100, 3)
        inf_percentage = round(used_memory_for_inf/max_memory*100, 3)

        # save logs
        with open(log_file, "a") as f:
            f.write(f"\nQuestion 1")
            f.write(f"\nInference runtime: {runtime}")
            f.write(f"\npeak reserved memory ft: {used_memory}\npeak reserved memory for inference: {used_memory_for_inf}")
            f.write(f"\npeak reserved memory percentage ft: {used_percentage}\npeak reserved memory percentage for inference: {inf_percentage}")
        
        torch.cuda.empty_cache()
    else: 
        print(f"\nq1 inference already done; saved at: {output_q1_csv}\n")
        output_df_q1 = pd.read_csv(output_q1_csv)

    if not os.path.isfile(output_q2_csv):
        ### Show current memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        start_time = time.perf_counter()
        output_df_q2 = inference_q2(df, output_q2_csv, model, tokenizer, log_file)
        end_time = time.perf_counter()
        runtime = end_time - start_time

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_inf = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory/max_memory*100, 3)
        inf_percentage = round(used_memory_for_inf/max_memory*100, 3)

        # save logs
        with open(log_file, "a") as f:
            f.write(f"\nQuestion 2")
            f.write(f"\nInference runtime: {runtime}")
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
        output_df_q3 = inference_q3(df, output_q3_csv, model, tokenizer, log_file)
        end_time = time.perf_counter()
        runtime = end_time - start_time

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_inf = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory/max_memory*100, 3)
        inf_percentage = round(used_memory_for_inf/max_memory*100, 3)

        # save logs
        with open(log_file, "a") as f:
            f.write(f"\nQuestion 3")
            f.write(f"\nInference runtime: {runtime}")
            f.write(f"\npeak reserved memory ft: {used_memory}\npeak reserved memory for inference: {used_memory_for_inf}")
            f.write(f"\npeak reserved memory percentage ft: {used_percentage}\npeak reserved memory percentage for inference: {inf_percentage}")
        
        torch.cuda.empty_cache()
    else: 
        print(f"\nq3 inference already done; saved at: {output_q3_csv}\n")
        output_df_q3 = pd.read_csv(output_q3_csv)

    #output_all = output_df_q1.merge(output_df_q2, on="Accession Number")
    #output_all = output_all.merge(output_df_q3, on="Accession Number")
    #output_all.to_csv(final_output_csv)

    print(f"\n*********************\nAll inference outputs saved to: {final_output_csv}\n*********************\n")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_directory', help='directory to store inference output csv', required=True)
    args = parser.parse_args()
    main(args)