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
    outputs_ft = []

    for row_index, row in tqdm(df.iterrows(), total=df.shape[0]):
        accession_number = row['Accession Number']
        #print(f"**Processing {accession_number}**")
        report_text = row["Preprocessed Report Text"]
        responses, answers = predict(report_text, question_1, model, tokenizer, parse_response_q1, verbose=verbose)
        
        outputs_ft.append({"Accession Number": accession_number,
                        "Q1 Response": responses,
                        "Q1 Parsed Response": answers,
                        })
    output_df_q1 = pd.DataFrame(outputs_ft)
    output_df_q1.to_csv(output_csv)

    return output_df_q1

def inference_q2(df, output_csv, model, tokenizer, log_file):
    correct = np.array([0, 0])
    correct_all = 0
    count = 0

    outputs = []
    for row_index, row in tqdm(df.iterrows(), total=df.shape[0]):
        id = row["Accession Number"]
        report_text = row["Preprocessed Report Text"]
        
        response, answer = predict(report_text, question_2, model, tokenizer, parse_response_q2, verbose=verbose)
        if not answer:
            print(f"Cannot parse response: {response}")
        
        outputs.append({"Accession Number": id,
                        "Q2 Response": response,
                        "Q2 Parsed Response": answer})
    output_df_q2 = pd.DataFrame(outputs)
    output_df_q2.to_csv(output_csv)
    
    return output_df_q2

def inference_q3(df, output_csv, model, tokenizer, log_file):
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


def main(args):
    model_path = str(args.model_directory)
    output_dir = str(args.output_directory)
    test_data_path = str(args.test_data_path)
    
    ### Inference
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_file_inf = os.path.join(output_dir, "inference_log.txt")

    df = pd.read_csv(test_data_path)

    output_q1_csv = os.path.join(output_dir, "output_q1.csv")
    output_q2_csv = os.path.join(output_dir, "output_q2.csv") 
    output_q3_csv = os.path.join(output_dir, "output_q3.csv")
    final_output_csv = os.path.join(output_dir, "output_allq.csv")
    

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
        output_df_q1 = inference_q1(df, output_q1_csv, model, tokenizer, log_file_inf)
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
        output_df_q2 = inference_q2(df, output_q2_csv, model, tokenizer, log_file_inf)
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
        output_df_q3 = inference_q3(df, output_q3_csv, model, tokenizer, log_file_inf)
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # default model: /mnt/shareddata/yiyan/grid_search/gs_042824_new/epoch_3_lr_0.0001_r_128_alpha_256_dropout_0.1
    # default test data path: /home/yiyan_hao/data/preprocessed_consolidated_all_051324.csv
    # default output directory: /mnt/shareddata/yiyan/eval_best_llm/data_051824
    parser.add_argument('--model_directory', help='directory to saved finetuned model', required=True)
    parser.add_argument('--output_directory', help='directory to save the resulting model checkpoints', required=True)
    parser.add_argument('--test_data_path', help='directory to the test data', required=True)
    args = parser.parse_args()
    main(args)