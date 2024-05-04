import argparse
import json
import random
import os, subprocess
from csv import DictWriter
import multiprocessing
import itertools
import pickle
import pandas as pd

def add_main_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/yiyan_hao/breast/scripts/grid_search_2.json",
        help="Location of config file"
    )

    parser.add_argument(
        '--result_path', 
        type=str, 
        default="/mnt/shareddata/yiyan/grid_search/gs_050224_more_hyperparams/grid_search.csv", 
        help="Path to store grid_search table. This is preferably on shared storage"
    )


    return parser

def get_experiment_list(config: dict) -> (list[dict]):
    '''
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item, but the config contains a list, this will return one job for each item in the list.
    :config - experiment_config

    returns: jobs - a list of dicts, each of which encapsulates one job.
        *Example: {learning_rate: 0.001 , batch_size: 16 ...}
    '''

    keys = list(config.keys())[:-1]
    combinations = itertools.product(*(config[key] for key in keys))
    jobs = [dict(zip(keys, combination)) for combination in combinations]
        
    return jobs

def worker(args: argparse.Namespace, job_queue: multiprocessing.Queue, done_queue: multiprocessing.Queue, gpu:int):
    '''
    Worker thread for each worker. Consumes all jobs and pushes results to done_queue.
    :args - command line args
    :job_queue - queue of available jobs.
    :done_queue - queue where to push results.
    '''
    while not job_queue.empty():
        params = job_queue.get()
        if params is None:
            return
        done_queue.put(
            launch_experiment(args, params, gpu))


def launch_experiment(args: argparse.Namespace, experiment_config: dict, gpu: int) ->  dict:
    '''
    Launch an experiment and direct logs and results to a unique filepath.
    :configs: flags to use for this model run. Will be fed into
    scripts/main.py

    returns: flags for this experiment as well as result metrics
    '''
    gpu_command = "CUDA_VISIBLE_DEVICES={} ".format(gpu)
    command = gpu_command + "python /home/yiyan_hao/breast/scripts/llm_combined_ft.py"
    
    for key, value in experiment_config.items():
        if key != "available_gpus":
            command += (f" --{key} ")
            command += str(value)
        if key == "output_dir":
            log_path = value
    print(f"\nexperiment config: {experiment_config}\ncommand:\n {command}\n")
    subprocess.call(command, shell=True)
    
    subfolder_name = "epoch_{}_lr_{}_r_{}_alpha_{}_dropout_{}".format(int(experiment_config.get("epochs")), 
                                                                      float(experiment_config.get("lr")), 
                                                                      int(experiment_config.get("r")),
                                                                      int(experiment_config.get("scaling")) * int(experiment_config.get("r")), 
                                                                      float(experiment_config.get("lora_dropout")))
    log_path = os.path.join(log_path, subfolder_name)
    output_dir_inf = os.path.join(log_path, "inference")
    accuracies_path = os.path.join(output_dir_inf, "accuracy_sum.pickle")
    
    return accuracies_path, log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_main_args(parser)
    args = parser.parse_args()
    return args

def update_summary_with_results(result_path, log_path, summary):
    assert result_path is not None
    print(f"\nresult_path: {result_path}\nlog_path: {log_path}\nsummary:{summary}\n")
    #try:
    result_dict = pickle.load(open(result_path, 'rb'))
    
    #except Exception as e:
    #    print("Experiment failed! Logs are located at: {}".format(log_path))
    #    return summary
    
    result_dict['log_path'] = log_path
    summary.append(result_dict)

    # Write summary to csv
    sum_df = pd.DataFrame(summary)
    if os.path.exists(args.result_path):
        past_df = pd.read_csv(args.result_path)
        new_df = pd.concat([past_df, sum_df], axis=0)
    else: 
        new_df = sum_df
    pd.to_csv(new_df, args.result_path)
    print(f"\nCurrent summary list: {summary}\n\n")
    return summary


def main(args: argparse.Namespace) -> dict:
    print(args)
    config = json.load(open(args.config_path, "r"))
    print("Starting grid search with the following config:")
    print(config)

    # TODO: From config, generate a list of experiments to run
    experiments = get_experiment_list(config)
    random.shuffle(experiments)

    job_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    for exper in experiments:
        job_queue.put(exper)

    print("Launching dispatcher with {} experiments and {} workers".format(len(experiments), len(config['available_gpus'])))

    # TODO: Define worker fn to launch an experiment as a separate process.
    #for _ in range(args.num_workers):
    for gpu in config['available_gpus']:
        print("Start gpu worker {}".format(gpu))
        multiprocessing.Process(target=worker, args=(args, job_queue, done_queue, gpu)).start()

    # Accumualte results into a list of dicts
    summary_list = []
    for i in range(len(experiments)):
        result_path, log_path = done_queue.get()
        summary_list = update_summary_with_results(result_path, log_path, summary_list)
        dump_result_string = "SUCCESS! Grid search results dumped to {}.".format(args.result_path)
        print("({}/{}) \t {}".format(i+1, len(experiments), dump_result_string))

    print("Done")

if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)
