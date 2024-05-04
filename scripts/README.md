Here, we provide the scripts used for pre-processing the dataset, performing baseline LLM evaluation, and fine-tuning. 

To run each script, first change the file path to the correct path corresponding your local directory containing data.

preprocess_breast_data.py performs necessary preprocessing steps to ensure the quality of the dataset. Run train_test_split.py to split the dataset into train / validation / test sets. You can specify the proportion of each subset in the script. You might want to try different random seeds to ensure the data distribution is balanced.

To evaluate zero-shot performance of baseline LLMs, modify llm_zeroshot_llama2-7b-chat.py with your model, prompts, and post-processing steps. Remember to check the chat template of each model. 

For fine-tuning, modify grid_search.json to adjust the hyperparameter search settings, and run dispatcher.py to run experiments. 